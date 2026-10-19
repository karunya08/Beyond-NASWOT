"""
Extract ground truth test accuracies from JAHS-Bench-201 surrogates.

Maps every NAS-Bench-201 architecture (15,625 UIDs) to JAHS-Bench-201 config,
queries the surrogate for ReLU / Hardswish / Mish test-acc at epoch 200,
and saves .npy files into ground_truth/.

Run with the .jahs_env Python 3.10 environment:
    .jahs_env\Scripts\python.exe extract_jahs_ground_truth.py

Output files (committed to repo -- grader never needs JAHS-Bench):
    ground_truth/jahs_relu_cifar10_accs.npy
    ground_truth/jahs_hardswish_cifar10_accs.npy
    ground_truth/jahs_mish_cifar10_accs.npy
    ground_truth/jahs_uids.npy              (just [0..15624])
"""

import os, re, time, json, warnings
import numpy as np

warnings.filterwarnings("ignore")

# ---------- JAHS-Bench-201 surrogate ----------
from jahs_bench import Benchmark

SAVE_DIR = "jahs_bench_data"
GROUND_TRUTH_DIR = "ground_truth"
ACTIVATIONS = ["ReLU", "Hardswish", "Mish"]
NEPOCHS = 200

# NAS-Bench-201 operation name -> JAHS Op index
# JAHS OP_NAMES = ['Identity', 'Zero', 'ConvBN3x3', 'ConvBN1x1', 'AvgPool1x1']
NB201_OP_TO_JAHS_IDX = {
    "skip_connect": 0,  # Identity
    "none":         1,  # Zero
    "nor_conv_3x3": 2,  # ConvBN3x3
    "nor_conv_1x1": 3,  # ConvBN1x1
    "avg_pool_3x3": 4,  # AvgPool1x1
}

# NAS-Bench-201 arch string -> list of 6 operation names
# Format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
# The 6 ops in string order correspond to edges:
#   (0->1), (0->2), (1->2), (0->3), (1->3), (2->3)
# JAHS EDGE_LIST = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4)) maps to Op1..Op6
# String pos -> JAHS Op:  pos0=Op1, pos1=Op2, pos2=Op4, pos3=Op3, pos4=Op5, pos5=Op6
STRING_POS_TO_JAHS_OP = [1, 2, 4, 3, 5, 6]  # pos_i -> Op number


def parse_nb201_arch(arch_str):
    """Parse NAS-Bench-201 arch string into JAHS Op1-Op6 dict."""
    # Extract operation names in order from the arch string
    ops = re.findall(r'(\w+)~\d', arch_str)
    assert len(ops) == 6, f"Expected 6 ops, got {len(ops)} from {arch_str}"

    jahs_config = {}
    for pos, op_name in enumerate(ops):
        jahs_op_num = STRING_POS_TO_JAHS_OP[pos]
        jahs_config[f"Op{jahs_op_num}"] = NB201_OP_TO_JAHS_IDX[op_name]
    return jahs_config


def main():
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

    # Load pre-extracted NAS-Bench-201 arch strings (avoids torch dependency)
    arch_file = os.path.join(GROUND_TRUTH_DIR, "nb201_arch_strings.json")
    with open(arch_file) as f:
        arch_strings = json.load(f)  # keys are string UIDs "0".."15624"
    n_archs = len(arch_strings)
    print(f"NAS-Bench-201: {n_archs} architectures (from {arch_file})")

    # Verify mapping with a quick sanity check
    arch0 = arch_strings["0"]
    ops0 = parse_nb201_arch(arch0)
    print(f"UID 0: {arch0}")
    print(f"  -> JAHS config: {ops0}")

    # Load JAHS surrogate (only test-acc to save memory)
    print("Loading JAHS-Bench-201 surrogate (test-acc only)...")
    bench = Benchmark(
        task="cifar10", kind="surrogate", download=False,
        save_dir=SAVE_DIR, metrics=("test-acc",)
    )
    print("Surrogate loaded.")

    # Base config (fixed hyperparameters -- standard training setup)
    base_config = {
        "Optimizer": "SGD",
        "LearningRate": 0.1,
        "WeightDecay": 5e-4,
        "TrivialAugment": False,
        "N": 5,
        "W": 16,
        "Resolution": 1.0,
    }

    # Allocate arrays
    uids = np.arange(n_archs, dtype=np.int32)
    accs = {act: np.zeros(n_archs, dtype=np.float32) for act in ACTIVATIONS}

    t0 = time.time()
    for uid in range(n_archs):
        arch_str = arch_strings[str(uid)]
        jahs_ops = parse_nb201_arch(arch_str)

        config = {**base_config, **jahs_ops}
        for act in ACTIVATIONS:
            config["Activation"] = act
            result = bench(config, nepochs=NEPOCHS)
            accs[act][uid] = result[NEPOCHS]["test-acc"]

        if (uid + 1) % 500 == 0 or uid == 0:
            elapsed = time.time() - t0
            rate = (uid + 1) / elapsed
            eta = (n_archs - uid - 1) / rate
            print(f"[{uid+1:5d}/{n_archs}] "
                  f"ReLU={accs['ReLU'][uid]:.2f}% "
                  f"Hardswish={accs['Hardswish'][uid]:.2f}% "
                  f"Mish={accs['Mish'][uid]:.2f}% "
                  f"| {rate:.1f} arch/s | ETA {eta/60:.1f}min")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    np.save(os.path.join(GROUND_TRUTH_DIR, "jahs_uids.npy"), uids)
    for act in ACTIVATIONS:
        fname = f"jahs_{act.lower()}_cifar10_accs.npy"
        np.save(os.path.join(GROUND_TRUTH_DIR, fname), accs[act])
        print(f"Saved {fname}: mean={accs[act].mean():.2f}%, "
              f"min={accs[act].min():.2f}%, max={accs[act].max():.2f}%")

    # Quick correlation check: do all three activations mostly agree on ranking?
    from scipy.stats import kendalltau
    for a, b in [("ReLU", "Hardswish"), ("ReLU", "Mish"), ("Hardswish", "Mish")]:
        tau, _ = kendalltau(accs[a], accs[b])
        print(f"Kendall-tau({a}, {b}) = {tau:.4f}")


if __name__ == "__main__":
    main()

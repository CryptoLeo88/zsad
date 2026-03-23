import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASET_SPECS = [
    {"name": "visa", "path": "./data/visa"},
    {"name": "mpdd", "path": "./data/MPDD"},
    {"name": "btad", "path": "./data/BTAD"},
    {"name": "sdd", "path": "./data/SDD"},
    {"name": "dagm", "path": "./data/DAGM"},
    {"name": "dtd", "path": "./data/DTD-Synthetic"},
]


TEST_SCRIPT_BY_MODEL = {
    "baseline": "test.py",
    "visual_prompt": "test_visual_prompt.py",
    "visual_prompt_gate": "test_visual_prompt_gate.py",
}


def build_command(args, dataset_spec):
    test_script = TEST_SCRIPT_BY_MODEL[args.model_type]
    data_path = dataset_spec["path"]
    save_path = os.path.join(args.results_root, dataset_spec["name"])

    cmd = [
        sys.executable,
        test_script,
        "--dataset", dataset_spec["name"],
        "--data_path", data_path,
        "--save_path", save_path,
        "--checkpoint_path", args.checkpoint_path,
        "--features_list", *[str(x) for x in args.features_list],
        "--image_size", str(args.image_size),
        "--depth", str(args.depth),
        "--n_ctx", str(args.n_ctx),
        "--t_n_ctx", str(args.t_n_ctx),
        "--metrics", args.metrics,
        "--sigma", str(args.sigma),
    ]

    if args.model_type in {"visual_prompt", "visual_prompt_gate"}:
        cmd.extend([
            "--visual_prompt_hidden_dim", str(args.visual_prompt_hidden_dim),
            "--visual_prompt_dropout", str(args.visual_prompt_dropout),
        ])

    if args.model_type == "visual_prompt":
        cmd.extend(["--cluster_alpha", str(args.cluster_alpha)])

    if args.model_type == "visual_prompt_gate":
        cmd.extend([
            "--gate_hidden_dim", str(args.gate_hidden_dim),
            "--gate_dropout", str(args.gate_dropout),
        ])

    return cmd


def main():
    parser = argparse.ArgumentParser("Run all non-MVTec evaluations")
    parser.add_argument("--model_type", type=str, default="visual_prompt_gate",
                        choices=["baseline", "visual_prompt", "visual_prompt_gate"])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint to evaluate")
    parser.add_argument("--results_root", type=str, default="./results/all_except_mvtec", help="root folder for logs")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--metrics", type=str, default="image-pixel-level")
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--visual_prompt_hidden_dim", type=int, default=0)
    parser.add_argument("--visual_prompt_dropout", type=float, default=0.1)
    parser.add_argument("--cluster_alpha", type=float, default=0.2)
    parser.add_argument("--gate_hidden_dim", type=int, default=0)
    parser.add_argument("--gate_dropout", type=float, default=0.1)
    args = parser.parse_args()

    Path(args.results_root).mkdir(parents=True, exist_ok=True)

    for dataset_spec in DATASET_SPECS:
        cmd = build_command(args, dataset_spec)
        print("=" * 80)
        print("Running:", " ".join(cmd))
        print("=" * 80)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

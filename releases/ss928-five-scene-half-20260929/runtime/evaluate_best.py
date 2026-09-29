"""Evaluate validation-selected checkpoints on frozen held-out test samples."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


RUNS = (
    "DAY-GPU1", "DAY-FULLNIGHT", "DAY-FULLNIGHT-LR1E4",
    "LIGHT-MEDIUM", "LIGHT-FULLNIGHT", "LIGHT-GATE01",
    "HEAVY-C32", "HEAVY-FULLNIGHT", "C32-SPECIALIST", "C32-FULLNIGHT",
    "C32-FULLNIGHT-LR1E4",
    "LIGHT-FULLNIGHT-WEAKMOTION", "HEAVY-FULLNIGHT-WEAKMOTION",
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=RUNS)
    parser.add_argument("--phase", choices=("12k", "20k"), default="20k")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code / "src"))
    from ir_sr.model import inference_model
    from ir_sr.training import evaluate

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    report = {}
    for name in args.runs:
        suffix = "-20K" if args.phase == "20k" else ""
        run = (args.root / "runs" /
               f"SS928-FIVE-SCENE-NINE-SCRATCH-20260929-{name}{suffix}")
        checkpoint = run / "checkpoints/best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = state["config"]
        assert not config.get("initialization_checkpoint")
        model = inference_model(config, state["model"]).eval().cuda()
        result = evaluate(model, config["data_root"], "test", "cuda",
                          args.out / name, state["progress"]["step"],
                          save_examples=True, scene_ids=config["scene_ids"],
                          config=config)
        report[name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "selected_step": state["progress"]["step"],
            "validation_best_psnr": state["best_val_macro_psnr"],
            "test_macro": result["macro"],
            "test_scene_metrics": result["scene_metrics"],
        }
        print(name, json.dumps(report[name], ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

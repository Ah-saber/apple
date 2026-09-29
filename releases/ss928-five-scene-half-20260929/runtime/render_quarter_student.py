"""Render complete test sequences and temporal metrics for a quarter student."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F


GROUP_SCENES = {
    "DAY": {"day_normal"},
    "DAY-FULLNIGHT": {"day_normal"},
    "DAY-FULLNIGHT-LR1E4": {"day_normal"},
    "LIGHT-MEDIUM": {"weather_light", "weather_medium"},
    "LIGHT-FULLNIGHT": {"weather_light", "weather_medium"},
    "LIGHT-GATE01": {"weather_light", "weather_medium"},
    "LIGHT-FULLNIGHT-WEAKMOTION": {"weather_light", "weather_medium"},
    "HEAVY-C32": {"weather_heavy", "weather_heavy_c32"},
    "HEAVY-FULLNIGHT": {"weather_heavy", "weather_heavy_c32"},
    "C32-SPECIALIST": {"weather_heavy_c32"},
    "C32-FULLNIGHT": {"weather_heavy_c32"},
    "C32-FULLNIGHT-LR1E4": {"weather_heavy_c32"},
    "HEAVY-FULLNIGHT-WEAKMOTION": {"weather_heavy", "weather_heavy_c32"},
}
BASELINE_RUNS = {
    "DAY": "SS928-QUALITY-PHASE-20260924-DAY-PHASE02-2K/checkpoints/step_000002000.pt",
    "DAY-FULLNIGHT": "SS928-QUALITY-PHASE-20260924-DAY-PHASE02-2K/checkpoints/step_000002000.pt",
    "DAY-FULLNIGHT-LR1E4": "SS928-QUALITY-PHASE-20260924-DAY-PHASE02-2K/checkpoints/step_000002000.pt",
    "LIGHT-MEDIUM": "SS928-QUALITY-V3-20260924-LIGHT_MEDIUM-ABSOLUTE-2K/checkpoints/step_000002000.pt",
    "LIGHT-FULLNIGHT": "SS928-QUALITY-V3-20260924-LIGHT_MEDIUM-ABSOLUTE-2K/checkpoints/step_000002000.pt",
    "LIGHT-GATE01": "SS928-QUALITY-V3-20260924-LIGHT_MEDIUM-ABSOLUTE-2K/checkpoints/step_000002000.pt",
    "LIGHT-FULLNIGHT-WEAKMOTION": "SS928-QUALITY-V3-20260924-LIGHT_MEDIUM-ABSOLUTE-2K/checkpoints/step_000002000.pt",
    "HEAVY-C32": "SS928-QUALITY-CROSS-20260923-HEAVY/checkpoints/step_000012000.pt",
    "HEAVY-FULLNIGHT": "SS928-QUALITY-CROSS-20260923-HEAVY/checkpoints/step_000012000.pt",
    "C32-SPECIALIST": "SS928-QUALITY-CROSS-20260923-HEAVY/checkpoints/step_000012000.pt",
    "C32-FULLNIGHT": "SS928-QUALITY-CROSS-20260923-HEAVY/checkpoints/step_000012000.pt",
    "C32-FULLNIGHT-LR1E4": "SS928-QUALITY-CROSS-20260923-HEAVY/checkpoints/step_000012000.pt",
    "HEAVY-FULLNIGHT-WEAKMOTION": "SS928-QUALITY-CROSS-20260923-HEAVY/checkpoints/step_000012000.pt",
}
LABELS = {
    "day_normal": "白天",
    "weather_light": "轻档",
    "weather_medium": "中档",
    "weather_heavy": "重档",
    "weather_heavy_c32": "重档 C32",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_video(path):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "5120x1080", "-r", "12", "-i", "-", "-an", "-c:v", "libx264",
         "-threads", "4", "-preset", "veryfast", "-crf", "21",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(path)],
        stdin=subprocess.PIPE,
    )


def gray8(value):
    return np.rint(np.clip(value, 0, 255)).astype(np.uint8)


def panel(raw, gt, baseline, prediction, scene, frame, font,
          repeat_current=False, gate_bias_offset=0.0, fast_fused=False):
    canvas = Image.new("RGB", (5120, 1080), "#161b22")
    draw = ImageDraw.Draw(canvas)
    for index, (value, title) in enumerate(
        zip((raw, gt, baseline, prediction),
            ("输入 RAW", "GT", "原单帧模型",
             "当前帧重复九次" if repeat_current else
             "紧凑九帧候选"))
    ):
        assert value.shape == (1024, 1280)
        canvas.paste(Image.fromarray(gray8(value)).convert("RGB"),
                     (index * 1280, 56))
        draw.text((index * 1280 + 16, 10),
                  f"{LABELS[scene]} | {title} | 第 {frame:03d} 帧",
                  font=font, fill="white")
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--detail-scale", type=float, default=1.0)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--night-runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=12000)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--repeat-current", action="store_true",
                        help="Replace all historical frames with the current one")
    parser.add_argument("--gate-bias-offset", type=float, default=0.0,
                        help="Add a constant to the learned motion-gate logit bias")
    parser.add_argument("--best", action="store_true",
                        help="Read the validation-selected best checkpoint from the 20k continuation")
    parser.add_argument("--fast-fused", action="store_true",
                        help="Use the prior night fused half-precision inference primitives")
    parser.add_argument("--runtime", type=Path,
                        help="Directory containing the prior night fused runtime primitives")
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUP_SCENES),
                        default=["HEAVY-C32", "LIGHT-MEDIUM", "DAY"])
    parser.add_argument("--scenes", nargs="+", choices=tuple(LABELS))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.fast_fused and not args.runtime:
        parser.error("--fast-fused requires --runtime")
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code / "src"))
    sys.path.insert(0, str(args.night_runtime))
    sys.path.insert(0, str(Path(__file__).parent))
    if args.runtime:
        sys.path.insert(0, str(args.runtime))
    if args.fast_fused:
        from fast_nine_inference import build_fast_model
    from ir_sr.model import inference_model
    from ir_sr.sequence_normalization import allowed_region
    from ir_sr.training import dataset_for_config
    from half_student import make_student
    from train_quarter_student import BoxQuarter, box_for

    catalog = json.loads((args.root / "data/manifests/dataset_d1/scene_catalog.json").read_text())
    frame_counts = {(s["scene_id"], s["sequence_id"]): s["frames"]
                    for s in catalog["sequences"]}
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 24)
    report = {"kind": "five_scene_quarter_student_full_sequence", "scenes": {}}

    for group in args.groups:
        suffix = "DAY-GPU1" if group == "DAY" else group
        if (args.best or args.checkpoint_step > 12000) and not group.endswith("-WEAKMOTION"):
            suffix += "-20K"
        run = args.root / "runs" / f"SS928-FIVE-SCENE-NINE-SCRATCH-20260929-{suffix}"
        checkpoint = (run / "checkpoints/best.pt" if args.best else
                      run / "checkpoints" / f"step_{args.checkpoint_step:09d}.pt")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = state["config"]
        assert not config.get("initialization_checkpoint")
        dataset = dataset_for_config(config, args.split)
        student_state = torch.load(args.student_checkpoint,
                                   map_location="cpu", weights_only=False)
        reference_state = (torch.load(args.reference_checkpoint,
                           map_location="cpu", weights_only=False)
                           if args.reference_checkpoint else state)
        reference = inference_model(reference_state["config"],
                                    reference_state["model"]).global_reference
        if student_state.get('format') == 'quarter_detail_refiner_v1':
            from refine_quarter_detail import DetailRefiner
            base_state = torch.load(student_state['base_checkpoint'],
                                    map_location='cpu', weights_only=False)
            base = BoxQuarter(make_student(reference,base_state),
                current_only=base_state.get('current_only', False))
            model = DetailRefiner(base)
            model.scale = args.detail_scale
        else:
            model = BoxQuarter(make_student(reference,student_state),
                current_only=student_state.get('current_only', False))
        model.load_state_dict(student_state["model"], strict=True)
        model = model.eval().cuda()
        baseline_checkpoint = args.root / "runs" / BASELINE_RUNS[group]
        baseline_state = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
        baseline_config = baseline_state["config"]
        baseline_dataset = dataset_for_config(baseline_config, args.split)
        baseline_model = inference_model(baseline_config,
                                         baseline_state["model"]).eval().cuda()
        selected = {}
        for row in sorted(dataset.records, key=lambda r: (r["scene_id"], r["sequence_id"], r["frame_id"])):
            if args.scenes is None or row["scene_id"] in args.scenes:
                selected.setdefault(row["scene_id"], row)
        assert set(selected) == (set(args.scenes) if args.scenes else GROUP_SCENES[group])

        for scene, record in selected.items():
            full_count = frame_counts[scene, record["sequence_id"]]
            count = min(full_count, args.max_frames) if args.max_frames else full_count
            video = args.out / f"{scene}__quarter__{args.split}__{record['sequence_id']}__full_comparison.mp4"
            writer = write_video(video)
            motion = []
            errors = []
            previous = None
            try:
                with torch.inference_mode():
                    for frame in range(count):
                        row = dict(record, frame_id=frame)
                        raw = dataset.normalized_stack(row, (0, 0, 1024, 1280))[None].cuda()
                        if args.repeat_current:
                            raw = raw[:, -1:].expand_as(raw)
                        rt, rl, rh, rw = allowed_region(row)
                        context, _ = dataset.context_for(row)
                        box = torch.tensor([-rt / rh, -rl / rw,
                                            (1024 - rt) / rh, (1280 - rl) / rw],
                                           dtype=torch.float32)
                        with torch.autocast('cuda', dtype=torch.float16):
                            prediction = model.native(raw, context[None].cuda(),
                                box_for(row).cuda()).float().clamp(0, 1)
                        assert tuple(prediction.shape) == (1, 1, 1024, 1280)
                        result = prediction[0, 0].cpu().numpy() * 255
                        baseline_norm = baseline_dataset.normalization_for(row)
                        baseline_raw = (baseline_dataset._raw(row).astype(np.float32)
                                        - baseline_norm["offset"]) / baseline_norm["scale"]
                        baseline_context, _ = baseline_dataset.context_for(row)
                        baseline_prediction = baseline_model(
                            torch.from_numpy(baseline_raw.copy())[None, None].cuda(),
                            context=baseline_context[None].cuda(),
                            context_box=box[None].cuda()).float().clamp(0, 1)
                        baseline_result = F.avg_pool2d(baseline_prediction, 3, 3)[0, 0].cpu().numpy() * 255
                        gt_path = (Path(config["data_root"]) / row["target"]["path"]).with_name(f"{frame:06d}.png")
                        gt = np.asarray(Image.open(gt_path), dtype=np.float32)
                        norm = dataset.normalization_for(row)
                        original = (dataset.base._raw(row).astype(np.float32) - norm["offset"]) / norm["scale"] * 255
                        writer.stdin.write(panel(original, gt, baseline_result,
                                                 result, scene, frame, font,
                                                 args.repeat_current,
                                                 args.gate_bias_offset,
                                                 args.fast_fused).tobytes())

                        y, x, h, w = row["eval_crop_tlhw"]
                        y, x, h, w = y + 3, x + 3, h - 6, w - 6
                        pred_roi = result[y:y+h, x:x+w]
                        baseline_roi = baseline_result[y:y+h, x:x+w]
                        gt_roi = gt[y:y+h, x:x+w]
                        errors.append({
                            "new": float(np.mean(np.abs(pred_roi - gt_roi))),
                            "baseline": float(np.mean(np.abs(baseline_roi - gt_roi))),
                            "bias_new": float(np.mean(pred_roi - gt_roi)),
                            "bias_baseline": float(np.mean(baseline_roi - gt_roi)),
                        })
                        if previous is not None:
                            prev_pred, prev_baseline, prev_gt = previous
                            spatial_gradient = np.maximum.reduce((
                                np.abs(gt_roi - np.roll(gt_roi, 1, axis=0)),
                                np.abs(gt_roi - np.roll(gt_roi, 1, axis=1)),
                                np.abs(gt_roi - np.roll(gt_roi, -1, axis=0)),
                                np.abs(gt_roi - np.roll(gt_roi, -1, axis=1)),
                            ))
                            gt_change = gt_roi - prev_gt
                            static = (np.abs(gt_change) <= 1.0) & (spatial_gradient <= 5.0)
                            if static.any():
                                frame_metrics = {
                                    "frame": frame,
                                    "static_pixel_fraction": float(static.mean()),
                                    "prediction_change_gray": float(np.abs(pred_roi - prev_pred)[static].mean()),
                                    "gt_change_gray": float(np.abs(gt_roi - prev_gt)[static].mean()),
                                    "residual_change_gray": float(np.abs((pred_roi-prev_pred) - (gt_roi-prev_gt))[static].mean()),
                                    "baseline_static_prediction_change_gray": float(np.abs(baseline_roi-prev_baseline)[static].mean()),
                                    "baseline_static_residual_change_gray": float(np.abs((baseline_roi-prev_baseline) - (gt_roi-prev_gt))[static].mean()),
                                }
                                moving = np.abs(gt_change) >= 3.0
                                if moving.any():
                                    denominator = float(np.mean(np.abs(gt_change[moving])))
                                    frame_metrics["motion_response_ratio"] = float(
                                        np.mean(np.sign(gt_change[moving]) *
                                                (pred_roi-prev_pred)[moving]) / denominator)
                                    frame_metrics["baseline_motion_response_ratio"] = float(
                                        np.mean(np.sign(gt_change[moving]) *
                                                (baseline_roi-prev_baseline)[moving]) / denominator)
                                weak_structure = ((np.abs(gt_change) >= 1.0) &
                                                  (np.abs(gt_change) < 3.0) &
                                                  (spatial_gradient >= 5.0))
                                frame_metrics["weak_structure_fraction"] = float(weak_structure.mean())
                                if weak_structure.any():
                                    denominator = float(np.mean(np.abs(gt_change[weak_structure])))
                                    frame_metrics["weak_structure_temporal_error_gray"] = float(
                                        np.mean(np.abs((pred_roi-prev_pred-gt_change)[weak_structure])))
                                    frame_metrics["baseline_weak_structure_temporal_error_gray"] = float(
                                        np.mean(np.abs((baseline_roi-prev_baseline-gt_change)[weak_structure])))
                                    frame_metrics["weak_structure_response_ratio"] = float(
                                        np.mean(np.sign(gt_change[weak_structure]) *
                                                (pred_roi-prev_pred)[weak_structure]) / denominator)
                                    frame_metrics["baseline_weak_structure_response_ratio"] = float(
                                        np.mean(np.sign(gt_change[weak_structure]) *
                                                (baseline_roi-prev_baseline)[weak_structure]) / denominator)
                                motion.append(frame_metrics)
                        previous = pred_roi.copy(), baseline_roi.copy(), gt_roi.copy()
                        if frame % 20 == 0:
                            print("FRAME", scene, frame, count, flush=True)
            finally:
                writer.stdin.close()
                if writer.wait() != 0:
                    raise RuntimeError(f"Video encoding failed: {video}")
            scene_report = {
                "sequence_id": record["sequence_id"], "model_group": group,
                "split": args.split, "repeat_current": args.repeat_current,
                "gate_bias_offset": args.gate_bias_offset,
                "fast_fused": args.fast_fused,
                "frames": count,
                "full_sequence_frames": full_count,
                "checkpoint": str(args.student_checkpoint),
                "checkpoint_sha256": sha(args.student_checkpoint),
                "checkpoint_step": student_state["step"],
                "baseline_checkpoint": str(baseline_checkpoint),
                "baseline_checkpoint_sha256": sha(baseline_checkpoint),
                "normalization_index": config["sequence_normalization_index"],
                "baseline_normalization_index": baseline_config["sequence_normalization_index"],
                "video": str(video), "video_sha256": sha(video),
                "mean_absolute_error_gray": float(np.mean([e["new"] for e in errors])),
                "baseline_mean_absolute_error_gray": float(np.mean([e["baseline"] for e in errors])),
                "mean_brightness_bias_gray": float(np.mean([e["bias_new"] for e in errors])),
                "baseline_mean_brightness_bias_gray": float(np.mean([e["bias_baseline"] for e in errors])),
                "std_frame_brightness_bias_gray": float(np.std([e["bias_new"] for e in errors])),
                "baseline_std_frame_brightness_bias_gray": float(np.std([e["bias_baseline"] for e in errors])),
                "mean_static_prediction_change_gray": float(np.mean([m["prediction_change_gray"] for m in motion])),
                "baseline_mean_static_prediction_change_gray": float(np.mean([m["baseline_static_prediction_change_gray"] for m in motion])),
                "mean_static_gt_change_gray": float(np.mean([m["gt_change_gray"] for m in motion])),
                "mean_static_residual_change_gray": float(np.mean([m["residual_change_gray"] for m in motion])),
                "baseline_mean_static_residual_change_gray": float(np.mean([m["baseline_static_residual_change_gray"] for m in motion])),
                "mean_weak_structure_fraction": float(np.mean([m["weak_structure_fraction"] for m in motion])),
                "mean_weak_structure_temporal_error_gray": (
                    float(np.mean([m["weak_structure_temporal_error_gray"] for m in motion
                                   if "weak_structure_temporal_error_gray" in m]))
                    if any("weak_structure_temporal_error_gray" in m for m in motion) else None),
                "baseline_mean_weak_structure_temporal_error_gray": (
                    float(np.mean([m["baseline_weak_structure_temporal_error_gray"] for m in motion
                                   if "baseline_weak_structure_temporal_error_gray" in m]))
                    if any("baseline_weak_structure_temporal_error_gray" in m for m in motion) else None),
                "mean_weak_structure_response_ratio": (
                    float(np.mean([m["weak_structure_response_ratio"] for m in motion
                                   if "weak_structure_response_ratio" in m]))
                    if any("weak_structure_response_ratio" in m for m in motion) else None),
                "baseline_mean_weak_structure_response_ratio": (
                    float(np.mean([m["baseline_weak_structure_response_ratio"] for m in motion
                                   if "baseline_weak_structure_response_ratio" in m]))
                    if any("baseline_weak_structure_response_ratio" in m for m in motion) else None),
                "mean_motion_response_ratio": (
                    float(np.mean([m["motion_response_ratio"] for m in motion if "motion_response_ratio" in m]))
                    if any("motion_response_ratio" in m for m in motion) else None),
                "baseline_mean_motion_response_ratio": (
                    float(np.mean([m["baseline_motion_response_ratio"] for m in motion
                                   if "baseline_motion_response_ratio" in m]))
                    if any("baseline_motion_response_ratio" in m for m in motion) else None),
                "temporal_frames": motion,
            }
            report["scenes"][scene + "__" + group] = scene_report
            (args.out / f"{scene}__{group.lower()}__report.json").write_text(
                json.dumps(scene_report, indent=2, ensure_ascii=False))
            print("SCENE_DONE", scene, json.dumps({k: v for k, v in scene_report.items() if k.startswith("mean_")}), flush=True)
        del model, baseline_model
        torch.cuda.empty_cache()

    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Evaluate RGB-D reprojection on the ECCV PhysAI iPhone validation bundle."""

import argparse
import csv
import math
from pathlib import Path
import re
import statistics
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity


METHODS = ("depth_gated_splat", "mesh_without_edge_gate", "mesh_with_edge_gate", "mesh_with_depth_repair")
VIDEO_SHAPE = (49, 288, 512, 3)


def read_video(ffmpeg, path):
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "pipe:1"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    if len(decoded) != math.prod(VIDEO_SHAPE):
        raise ValueError(f"Unexpected decoded RGB byte count for {path}")
    return np.frombuffer(decoded, np.uint8).reshape(VIDEO_SHAPE)


def prepare_source(bundle, sequence, trajectory, frames, path, stride):
    path.mkdir()
    count, rows, cols, _ = frames.shape
    with np.load(bundle / "source_depth" / sequence / f"{trajectory}.npz") as archive:
        depth = archive["depth_m"]
    if depth.shape != (count, rows, cols) or not np.isfinite(depth).all() or np.any(depth < 0) or depth.max() > 65.535:
        raise ValueError(f"Invalid LiDAR depth in {sequence}")
    depth_mm = np.rint(depth * 1000).astype(np.uint16)
    with np.load(bundle / "cameras" / sequence / f"{trajectory}.npz") as archive:
        intrinsics = [archive["source_K"], archive["target_K"]]
        poses = [archive["source_c2w_query"], archive["target_c2w_query"]]
    for k in intrinsics:
        if (k.shape != (count, 3, 3) or not np.isfinite(k).all()
                or not np.allclose(k[:, 2, :], [0, 0, 1])
                or not np.allclose(k[:, (0, 1), (1, 0)], 0)
                or (k[:, (0, 1), (0, 1)] <= 0).any()):
            raise ValueError(f"Invalid pinhole intrinsics in {sequence}")
    for pose in poses:
        if pose.shape != (count, 4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid camera-to-world shape in {sequence}")
        r = pose[:, :3, :3]
        if (not np.allclose(pose[:, 3, :], [0, 0, 0, 1])
                or not np.allclose(np.swapaxes(r, 1, 2) @ r, np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(r), 1, atol=1e-4)):
            raise ValueError(f"Non-rigid camera-to-world matrix in {sequence}")

    frames = frames[:, ::stride, ::stride]
    depth_mm = depth_mm[:, ::stride, ::stride]
    _, source_rows, source_cols, _ = frames.shape
    with (path / "pairs.txt").open("w", encoding="ascii") as manifest:
        print(source_cols, source_rows, cols, rows, count, file=manifest)
        for i in range(count):
            values = []
            for k, factor in zip(intrinsics, (stride, 1)):
                values.extend(k[i, (0, 1, 0, 1), (0, 1, 2, 2)] / factor)
            for pose in poses:
                values.extend(pose[i, :3].ravel())
            print(" ".join(f"{x:.12g}" for x in values), file=manifest)

            packed = np.empty((source_rows, source_cols, 5), np.uint8)
            packed[..., :3] = frames[i]
            packed[..., 3] = depth_mm[i] & 255
            packed[..., 4] = depth_mm[i] >> 8
            packed.tofile(path / f"{i}.rgbd")


def psnr(error, pixels):
    if pixels == 0:
        return float("nan")
    mse = error / pixels / 255**2
    return 100.0 if mse < 1e-10 else -10 * math.log10(mse)


def collect_metrics(sequence, predictions, truth, mask, times, methods, stride):
    per_frame, per_clip = [], []
    common = mask & np.all(predictions[..., 3] != 0, axis=1)
    count, rows, cols, _ = truth.shape
    full_pixels = rows * cols
    for m, method in enumerate(methods):
        totals = np.zeros(6, dtype=np.float64)
        for i, target in enumerate(truth):
            predicted = predictions[i, m, :, :, :3]
            error = np.square(predicted.astype(np.float64) - target).mean(axis=-1)
            masked = mask[i]
            shared = common[i]
            mask_count = int(masked.sum())
            shared_count = int(shared.sum())
            covered = int((masked & (predictions[i, m, :, :, 3] != 0)).sum())
            ssim = structural_similarity(
                target.astype(np.float64), predicted.astype(np.float64), channel_axis=-1, data_range=255,
                gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False,
            )
            splat_time = "scaled_splat_ms" if stride > 1 and m > 0 else "splat_ms"
            elapsed = times[i]["projection_ms"] + times[i][splat_time]
            if method.startswith("mesh_"):
                elapsed += times[i]["mesh_open_ms" if method == METHODS[1] else "mesh_gated_ms"]
            if method == METHODS[3]:
                elapsed += times[i]["repair_ms"]
            full_error, masked_error, shared_error = error.sum(), error[masked].sum(), error[shared].sum()
            totals += (full_error, masked_error, shared_error, covered, ssim, elapsed)
            per_frame.append(dict(
                sequence=sequence, method=method, frame=i, mask_pixels=mask_count,
                full_psnr_db=psnr(full_error, full_pixels), masked_psnr_db=psnr(masked_error, mask_count),
                shared_psnr_db=psnr(shared_error, shared_count), full_ssim=ssim,
                mask_coverage_pct=100 * covered / mask_count if mask_count else float("nan"),
                render_ms=elapsed,
            ))
        mask_pixels = int(mask.sum())
        if mask_pixels == 0 or not common.any():
            raise ValueError(f"No valid comparison pixels in {sequence}")
        per_clip.append(dict(
            sequence=sequence, method=method, frames=count, mask_pixels=mask_pixels,
            full_psnr_db=psnr(totals[0], full_pixels * count), masked_psnr_db=psnr(totals[1], mask_pixels),
            shared_psnr_db=psnr(totals[2], int(common.sum())), full_ssim=totals[4] / count,
            mask_coverage_pct=100 * totals[3] / mask_pixels, render_ms=totals[5] / count,
        ))
    return per_frame, per_clip


def evaluate_clip(args, sequence, trajectory):
    with TemporaryDirectory(prefix="frames-", dir=args.output) as scratch:
        work = Path(scratch)
        source_path = args.bundle / "sources" / sequence / trajectory / "source.mp4"
        source = read_video(args.ffmpeg, source_path)
        prepare_source(args.bundle, sequence, trajectory, source, work / "input", args.source_stride)
        # The C++ executable receives only source RGB-D and both camera calibrations.
        subprocess.run([str(args.executable), "--paired", "input", "rendered"], cwd=work, check=True)
        del source

        truth = read_video(args.ffmpeg, args.bundle / "gt" / sequence / trajectory / "gt.mp4")
        with np.load(args.bundle / "masks" / sequence / f"{trajectory}.npz") as masks:
            mask = masks["evaluation_mask"].astype(bool)
            if mask.shape != VIDEO_SHAPE[:-1] or not np.array_equal(mask, masks["valid"] & masks["covisible"]):
                raise ValueError(f"Invalid official mask in {sequence}")
        with (work / "rendered" / "timing.csv").open(newline="") as handle:
            times = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
        if [row["frame"] for row in times] != list(range(VIDEO_SHAPE[0])):
            raise ValueError(f"Missing renderer timing rows in {sequence}")
        count, h, w, _ = VIDEO_SHAPE
        out_shape = (count, len(args.methods), h, w, 4)
        file = work / "rendered" / "renders.rgba"
        if file.stat().st_size != math.prod(out_shape):
            raise ValueError(f"Unexpected renderer output size in {sequence}")
        predictions = np.fromfile(file, np.uint8).reshape(out_shape)
        fallback = 0 if args.source_stride == 1 else 1
        if np.any((predictions[:, fallback, ..., 3] != 0) & np.any(predictions[:, -3:, ..., 3] == 0, axis=1)):
            raise ValueError(f"A mesh variant lost fallback coverage in {sequence}")
        existing = predictions[:, -2, ..., 3] != 0
        if not np.array_equal(predictions[:, -2][existing], predictions[:, -1][existing]):
            raise ValueError(f"Depth repair changed covered pixels in {sequence}")
        frames, clips = collect_metrics(sequence, predictions, truth, mask, times, args.methods, args.source_stride)
        middle = count // 2
        panels = np.concatenate((truth[middle, None], predictions[middle, ..., :3]), axis=0)
        return frames, clips, panels


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, (float, np.floating)) else v for k, v in row.items()})


def preview(output, scenes, stride):
    image_width, image_height, gap, margin = 280, 158, 12, 22
    row_stride = image_height + 33
    labels = ["Captured target", "Depth-gated splat"]
    if stride > 1:
        labels.append("Scale-aware splat")
    labels.extend(("Mesh without edge gate", "Mesh with edge gate", "Mesh + depth repair"))
    cols = len(labels)
    canvas = Image.new("RGB", (2 * margin + cols * image_width + (cols - 1) * gap,
                               92 + row_stride * len(scenes)), "#f4f1ea")
    draw = ImageDraw.Draw(canvas)
    windows_font = Path("C:/Windows/Fonts/segoeui.ttf")
    font_path = str(windows_font) if windows_font.exists() else "DejaVuSans.ttf"
    title, regular = ImageFont.truetype(font_path, 23), ImageFont.truetype(font_path, 15)
    draw.text((margin, 10), "iPhone RGB-D novel-view rendering", font=title, fill="#172a2c")
    resolution = f"{VIDEO_SHAPE[2] // stride}x{VIDEO_SHAPE[1] // stride}"
    draw.text((margin, 42), f"{resolution} source. Frame 24 shown. Scores use all frames and the official mask.",
              font=regular, fill="#546461")
    for col, name in enumerate(labels):
        draw.text((margin + col * (image_width + gap), 68), name, font=regular, fill="#172a2c")
    for row, (seq, pictures, clips) in enumerate(scenes):
        top = 92 + row * row_stride
        for col, pixels in enumerate(pictures):
            left = margin + col * (image_width + gap)
            canvas.paste(Image.fromarray(pixels).resize((image_width, image_height), Image.Resampling.LANCZOS),
                         (left, top))
            label = seq if col == 0 else f"{clips[col - 1]['masked_psnr_db']:.2f} dB masked PSNR"
            draw.text((left, top + image_height + 3), label, font=regular, fill="#116863")
    canvas.save(output / "preview.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True, help="Directory containing test_pairs.csv")
    p.add_argument("--executable", type=Path, required=True, help="Compiled rgbd_demo binary")
    p.add_argument("--output", type=Path, default=Path("results/iphone"))
    p.add_argument("--ffmpeg", default="ffmpeg", help="Path to FFmpeg or its name on PATH")
    p.add_argument("--source-stride", type=int, default=1, choices=(1, 2),
                   help="Use 2 for the half-resolution source stress test")
    args = p.parse_args()
    args.methods = METHODS if args.source_stride == 1 else (METHODS[0], "scale_aware_splat", *METHODS[1:])
    args.bundle = args.bundle.resolve(strict=True)
    args.executable = args.executable.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.bundle / "test_pairs.csv").open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    if not pairs or len({r["video"] for r in pairs}) != len(pairs):
        raise ValueError("Expected unique clips in test_pairs.csv")

    frames, clips, examples = [], [], []
    for row in pairs:
        seq, trajectory = row["video"], row["trajectory"]
        if any(re.fullmatch(r"[A-Za-z0-9_-]+", part) is None for part in (seq, trajectory)):
            raise ValueError("Invalid clip name")
        if tuple(int(row[k]) for k in ("num_frames", "height", "width")) != VIDEO_SHAPE[:-1]:
            raise ValueError(f"Unexpected video format for {seq}")
        frame_rows, clip_rows, images = evaluate_clip(args, seq, trajectory)
        frames.extend(frame_rows)
        clips.extend(clip_rows)
        examples.append((seq, images, clip_rows))
        print(seq + " " + " ".join(f"{r['masked_psnr_db']:.4f}" for r in clip_rows), flush=True)

    for method in args.methods:
        group = [r for r in clips if r["method"] == method]
        mean = dict(sequence="mean", method=method, frames=sum(r["frames"] for r in group),
                    mask_pixels=sum(r["mask_pixels"] for r in group))
        mean.update({k: statistics.mean(r[k] for r in group) for k in list(group[0])[4:]})
        clips.append(mean)
        print(f"{method} mean masked PSNR = {mean['masked_psnr_db']:.4f} dB", flush=True)
    write_csv(args.output / "per_frame.csv", frames)
    write_csv(args.output / "per_clip.csv", clips)
    preview(args.output, examples, args.source_stride)


if __name__ == "__main__":
    main()

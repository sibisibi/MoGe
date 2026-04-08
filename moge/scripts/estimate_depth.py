"""Stage 1: Extract frames from video + run MoGe depth/FoV estimation per frame.

Outputs per-frame depth.npy and fov.json in a layout compatible with
VITRA's process_video.py (stage 2).

Output layout:
    {output_dir}/{filename}/rgb_frames/{frame_idx:06d}.jpg
    {output_dir}/{filename}/moge/{frame_idx:06d}/depth.npy
    {output_dir}/{filename}/moge/{frame_idx:06d}/fov.json
"""

import argparse
import itertools
import json
import os
from os.path import join
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from moge.model import import_model_class_by_version
import utils3d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Root directory containing source videos: {data_dir}/{filename}.mp4")
    parser.add_argument("--index_path", required=True,
                        help="Path to index.csv")
    parser.add_argument("--output_dir", required=True,
                        help="Output root: {output_dir}/{filename}/rgb_frames/ and moge/")
    parser.add_argument("--index", type=int, required=True,
                        help="Row index in index.csv to process")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resolution_level", type=int, default=9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained", type=str, default="Ruicheng/moge-2-vitl-normal")
    return parser.parse_args()


def extract_frames(video_path, output_dir):
    """Extract all frames from mp4 to jpg. Returns list of saved paths."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        path = join(output_dir, f"{idx:06d}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
        idx += 1
    cap.release()
    print(f"  Extracted {idx} frames at {fps} fps")
    return paths, fps


def main():
    args = parse_args()

    row = pd.read_csv(args.index_path).iloc[args.index]
    filename = row["filename"]
    print(f"Index {args.index}: filename={filename}")

    video_path = join(args.data_dir, f"{filename}.mp4")
    rgb_dir = join(args.output_dir, filename, "rgb_frames")
    moge_dir = join(args.output_dir, filename, "moge")

    # Extract frames (or use existing)
    existing = sorted(Path(rgb_dir).glob("*.jpg")) if Path(rgb_dir).is_dir() else []
    if existing:
        print(f"  Using {len(existing)} existing frames in {rgb_dir}")
        image_paths = [str(p) for p in existing]
    else:
        image_paths, fps = extract_frames(video_path, rgb_dir)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No frames found/extracted for {filename}")

    # Load model
    device = torch.device(args.device)
    model = import_model_class_by_version("v2").from_pretrained(args.pretrained).to(device).eval()
    if args.fp16:
        model.half()

    # Run MoGe per frame
    for image_path in tqdm(image_paths, desc="MoGe"):
        stem = Path(image_path).stem
        save_dir = join(moge_dir, stem)
        os.makedirs(save_dir, exist_ok=True)

        # Skip if already done
        if os.path.exists(join(save_dir, "fov.json")) and os.path.exists(join(save_dir, "depth.npy")):
            continue

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        output = model.infer(image_tensor, resolution_level=args.resolution_level, use_fp16=args.fp16)
        depth = output['depth'].cpu().numpy()
        intrinsics = output['intrinsics'].cpu().numpy()
        normal = output['normal'].cpu().numpy() if 'normal' in output else None

        # Save depth
        np.save(join(save_dir, "depth.npy"), depth.astype(np.float16))

        # Save fov
        fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
        with open(join(save_dir, "fov.json"), 'w') as f:
            json.dump({
                "fov_x": round(float(np.rad2deg(fov_x)), 2),
                "fov_y": round(float(np.rad2deg(fov_y)), 2),
            }, f)

    print(f"  Done: {moge_dir}")


if __name__ == "__main__":
    main()

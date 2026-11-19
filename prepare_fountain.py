"""Pack the Open3D Fountain sample for the C++ multiview runner."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(source, output):
    colors = sorted((source / "image").glob("*.jpg"))
    depths = sorted((source / "depth").glob("*.png"))
    rows = (source / "scene" / "key.log").read_text().split()
    if len(colors) != 33 or len(depths) != len(colors) or len(rows) != 19 * len(colors):
        raise ValueError("Expected the 33 color/depth/pose triplets in the Open3D Fountain sample")
    output.mkdir(parents=True, exist_ok=True)
    manifest = ["640 480 525.0 525.0 319.5 239.5"]
    for i, (color_path, depth_path) in enumerate(zip(colors, depths)):
        offset = i * 19 + 3  # Three metadata integers precede each 4x4 pose.
        if tuple(map(int, rows[offset - 3:offset])) != (i, i, i + 1):
            raise ValueError(f"Unexpected pose metadata at frame {i}")
        pose = np.array(rows[offset:offset + 16], dtype=float).reshape(4, 4)
        rotation = pose[:3, :3]
        if (not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
            raise ValueError(f"Invalid pose {i}")
        # The text log rounds rotations. Restore orthogonality before using R transpose.
        u, _, vh = np.linalg.svd(rotation)
        pose[:3, :3] = u @ vh
        manifest.append(" ".join(f"{value:.12g}" for value in pose[:3].ravel()))

        with Image.open(color_path) as color_image, Image.open(depth_path) as depth_image:
            rgb = np.asarray(color_image.convert("RGB"))
            depth = np.asarray(depth_image)
        if rgb.shape != (480, 640, 3) or depth.shape != (480, 640):
            raise ValueError(f"Unexpected image dimensions at frame {i}")
        if depth.min() < 0 or depth.max() > 65535:
            raise ValueError(f"Out-of-range depth at frame {i}")
        packed = np.empty((480, 640, 5), dtype=np.uint8)
        packed[..., :3] = rgb
        packed[..., 3] = depth & 255
        packed[..., 4] = depth >> 8
        packed.tofile(output / f"{i}.rgbd")
    (output / "cameras.txt").write_text("\n".join(manifest) + "\n", encoding="ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted SampleFountainRGBDImages directory")
    parser.add_argument("output", type=Path, help="Directory for packed frames")
    args = parser.parse_args()
    prepare(args.source, args.output)

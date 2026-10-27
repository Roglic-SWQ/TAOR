from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build M-Hyper features for an image-filled MKG-W dataset while keeping the "
            "official MKG-W visual/textual feature dimensionality."
        )
    )
    parser.add_argument("--base-mhyper-dataset-dir", required=True)
    parser.add_argument("--filled-mhyper-dataset-dir", required=True)
    parser.add_argument("--filled-clip-image-features", required=True)
    parser.add_argument("--generated-manifest", required=True)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    return parser.parse_args()


def entity_key_to_id(entity2id_path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with entity2id_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            key = parts[0].rstrip("/").rsplit("/", 1)[-1]
            mapping[key] = int(parts[1])
    return mapping


def manifest_entity_ids(manifest_path: Path, key_to_id: dict[str, int]) -> list[int]:
    ids: list[int] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            entity_key = row["entity_key"]
            ids.append(key_to_id[entity_key])
    return sorted(set(ids))


def fit_ridge_map(x: torch.Tensor, y: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
    x_aug = torch.cat([x, ones], dim=1)
    lhs = x_aug.T @ x_aug
    lhs += torch.eye(lhs.shape[0], dtype=x.dtype) * ridge_lambda
    rhs = x_aug.T @ y
    return torch.linalg.solve(lhs, rhs)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_mhyper_dataset_dir)
    filled_dir = Path(args.filled_mhyper_dataset_dir)
    filled_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train", "valid", "test", "entity2id.txt", "relation2id.txt"):
        shutil.copy2(base_dir / name, filled_dir / name)

    official_visual = torch.load(base_dir / "MKG-W-visual.pth", map_location="cpu").float()
    official_textual = torch.load(base_dir / "MKG-W-textual.pth", map_location="cpu").float()
    clip_visual = torch.load(args.filled_clip_image_features, map_location="cpu").float()

    key_to_id = entity_key_to_id(base_dir / "entity2id.txt")
    replacement_ids = manifest_entity_ids(Path(args.generated_manifest), key_to_id)
    replacement_mask = torch.zeros(official_visual.shape[0], dtype=torch.bool)
    replacement_mask[torch.tensor(replacement_ids, dtype=torch.long)] = True

    official_nonzero = official_visual.abs().sum(dim=1) != 0
    clip_nonzero = clip_visual.abs().sum(dim=1) != 0
    fit_mask = official_nonzero & clip_nonzero & ~replacement_mask

    weights = fit_ridge_map(clip_visual[fit_mask], official_visual[fit_mask], args.ridge_lambda)
    x_replace = clip_visual[replacement_mask]
    x_replace_aug = torch.cat([x_replace, torch.ones((x_replace.shape[0], 1), dtype=x_replace.dtype)], dim=1)
    predicted_visual = x_replace_aug @ weights

    hybrid_visual = official_visual.clone()
    hybrid_visual[replacement_mask] = predicted_visual

    torch.save(hybrid_visual, filled_dir / "img_features.pth")
    torch.save(official_textual, filled_dir / "text_features.pth")

    summary = {
        "base_mhyper_dataset_dir": str(base_dir),
        "filled_mhyper_dataset_dir": str(filled_dir),
        "filled_clip_image_features": str(Path(args.filled_clip_image_features)),
        "generated_manifest": str(Path(args.generated_manifest)),
        "ridge_lambda": args.ridge_lambda,
        "official_visual_shape": list(official_visual.shape),
        "official_textual_shape": list(official_textual.shape),
        "clip_visual_shape": list(clip_visual.shape),
        "fit_rows": int(fit_mask.sum().item()),
        "replacement_rows": int(replacement_mask.sum().item()),
        "official_visual_nonzero_rows": int(official_nonzero.sum().item()),
        "clip_visual_nonzero_rows": int(clip_nonzero.sum().item()),
        "hybrid_visual_zero_rows": int((hybrid_visual.abs().sum(dim=1) == 0).sum().item()),
        "text_features_source": str(base_dir / "MKG-W-textual.pth"),
        "image_features_policy": (
            "official visual features copied for unchanged entities; generated-image rows "
            "mapped from CLIP512 into official 383-d visual space with ridge regression"
        ),
    }
    (filled_dir / "official_space_feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

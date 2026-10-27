from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate M-Hyper on a subset of test triples.")
    parser.add_argument("--mhyper-src-dir", default="M-Hyper/src")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="M_Hyper_B")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--init", type=float, default=1e-3)
    return parser.parse_args()


def avg_both(mrrs: dict[str, float], hits: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "MRR": (mrrs["lhs"] + mrrs["rhs"]) / 2.0,
        "hits@[1,3,10]": ((hits["lhs"] + hits["rhs"]) / 2.0).tolist(),
    }


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


def manifest_entity_keys(manifest_path: Path) -> list[str]:
    keys: list[str] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                keys.append(json.loads(line)["entity_key"])
    return keys


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    mhyper_src_dir = root / args.mhyper_src_dir
    sys.path.insert(0, str(mhyper_src_dir))

    from datasets import Dataset  # type: ignore
    from models import M_Hyper_B  # type: ignore

    data_root = mhyper_src_dir.parent / "data"
    feature_root = mhyper_src_dir.parent / "datasets" / args.dataset

    dataset = Dataset(str(data_root), args.dataset)
    key_to_id = entity_key_to_id(feature_root / "entity2id.txt")
    selected_ids = {key_to_id[key] for key in manifest_entity_keys(root / args.manifest)}

    test = dataset.get_examples("test")
    subset_mask = [(int(row[0]) in selected_ids) or (int(row[2]) in selected_ids) for row in test]
    subset = test[subset_mask]
    dataset.data["test_selected"] = subset

    img_emb = torch.load(feature_root / "img_features.pth", map_location="cpu")
    text_emb = torch.load(feature_root / "text_features.pth", map_location="cpu")
    model = M_Hyper_B(dataset.get_shape(), args.rank, args.init, img_emb, text_emb)
    state = torch.load(root / args.checkpoint, map_location="cuda:0")
    model.load_state_dict(state)
    model.cuda()

    os.chdir(mhyper_src_dir)
    full = avg_both(*dataset.eval(model, "test", -1))
    selected = avg_both(*dataset.eval(model, "test_selected", -1))
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "checkpoint": args.checkpoint,
                "manifest": args.manifest,
                "manifest_entities": len(selected_ids),
                "test_triples": int(test.shape[0]),
                "selected_test_triples": int(subset.shape[0]),
                "selected_test_triple_ratio": float(subset.shape[0] / test.shape[0]) if test.shape[0] else 0.0,
                "full_test": full,
                "selected_test": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

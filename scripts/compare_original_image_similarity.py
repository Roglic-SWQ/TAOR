#!/usr/bin/env python3
"""Compare the original image with each method's selected best image."""

import csv
import json
from pathlib import Path

import clip
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
COMPARISON_DIR = ROOT / "MMKGC/outputs/mkgw_image_comparison_50"
MODEL_PATH = ROOT / "MMKGC/CLIP/ViT-B-32.pt"
OUTPUT = COMPARISON_DIR / "original_vs_best_clip_similarity.csv"


def image_tensor(path, preprocess):
    with Image.open(path) as image:
        return preprocess(image.convert("RGB"))


def main():
    manifest = json.loads((COMPARISON_DIR / "comparison_manifest.json").read_text(encoding="utf-8"))
    model, preprocess = clip.load(str(MODEL_PATH), device="cpu", jit=False)
    model.eval()
    rows = []

    with torch.no_grad():
        for entity in manifest["entities"]:
            original = Path(entity["original"]["copied"])
            mmkgc = Path(entity["mmkgc"]["copied"])
            legacy = Path(entity["legacy_knowledge_bridger"]["copied"])
            batch = torch.stack([
                image_tensor(original, preprocess),
                image_tensor(mmkgc, preprocess),
                image_tensor(legacy, preprocess),
            ])
            features = model.encode_image(batch).float()
            features = features / features.norm(dim=-1, keepdim=True)
            scores = (features @ features[0]).tolist()
            mmkgc_score = float(scores[1])
            legacy_score = float(scores[2])
            if abs(mmkgc_score - legacy_score) < 0.01:
                winner = "tie"
            elif mmkgc_score > legacy_score:
                winner = "MMKGC"
            else:
                winner = "legacy Knowledge-Bridger"
            rows.append({
                "idx": entity["idx"],
                "entity_key": entity["entity_key"],
                "label": entity["label"],
                "original_vs_mmkgc_clip_cosine": round(mmkgc_score, 6),
                "original_vs_legacy_clip_cosine": round(legacy_score, 6),
                "winner": winner,
                "mmkgc_candidate_id": entity["mmkgc"]["candidate_id"],
                "legacy_candidate_index": entity["legacy_knowledge_bridger"]["candidate_index"],
            })

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    mmkgc_wins = sum(row["winner"] == "MMKGC" for row in rows)
    legacy_wins = sum(row["winner"] == "legacy Knowledge-Bridger" for row in rows)
    ties = sum(row["winner"] == "tie" for row in rows)
    print(f"rows={len(rows)}")
    print(f"mmkgc_mean={sum(r['original_vs_mmkgc_clip_cosine'] for r in rows) / len(rows):.6f}")
    print(f"legacy_mean={sum(r['original_vs_legacy_clip_cosine'] for r in rows) / len(rows):.6f}")
    print(f"wins_mmkgc={mmkgc_wins} wins_legacy={legacy_wins} ties={ties}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()

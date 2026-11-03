"""Evaluate a saved CCGL teacher or student on the prepared test split."""

import argparse
import json
from pathlib import Path

import torch

from .data import padded
from .losses import reported_msle
from .model import Predictor
from .train import configure, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    configure()
    test = torch.load(args.data, weights_only=True, map_location="cpu", mmap=True)["test"]
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    depth = sum(key.startswith("projection.") and key.endswith(".weight") for key in state)
    model = Predictor(depth).to(args.device)
    model.load_state_dict(state)
    prediction = predict(model, padded(test, args.device), args.batch_size)
    metrics = {"msle": float(reported_msle(prediction, test["labels"].to(args.device)))}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    torch.save({"predictions": prediction.cpu(), "labels": test["labels"], "ids": test["ids"]},
               args.output / "predictions.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

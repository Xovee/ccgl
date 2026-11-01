"""Pre-train, fine-tune and self-distill CCGL on the three label fractions."""

import argparse
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from .data import padded
from .losses import contrastive_loss, reported_msle, training_loss
from .model import ContrastiveModel, Predictor


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def checkpoint(model, path):
    temporary = path.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


@torch.no_grad()
def predict(model, x, batch_size):
    model.eval()
    return torch.cat([model(batch) for batch in x.split(batch_size)])


def record(path, **values):
    print(json.dumps(values), flush=True)
    with path.open("a") as log:
        log.write(json.dumps(values) + "\n")


def pretrain(model, first, second, args, directory):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, eps=1e-7)
    (directory / "epochs.jsonl").write_text("")
    best = float("inf")
    for epoch in range(args.pretrain_epochs):
        started = time.monotonic()
        permutation = torch.randperm(len(first), device=first.device)
        total = torch.zeros((), device=first.device)
        batches = 0
        model.train()
        for index in permutation.split(args.batch_size):
            if len(index) < args.batch_size:
                break  # Match the original complete-batch contrastive sampling.
            optimizer.zero_grad(set_to_none=True)
            z = model(torch.cat((first[index], second[index])))
            loss = contrastive_loss(z)
            loss.backward()
            optimizer.step()
            total += loss.detach()
            batches += 1
        mean = float(total / batches)
        record(directory / "epochs.jsonl", stage="pretrain", epoch=epoch + 1,
               loss=mean, seconds=time.monotonic() - started)
        if mean < best:
            best = mean
            checkpoint(model, directory / "pretrained.pt")
    model.load_state_dict(torch.load(directory / "pretrained.pt", weights_only=True))


def fit(model, x, y, validation, validation_y, args, directory, stage):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, eps=1e-7)
    best, stale = float("inf"), 0
    for epoch in range(args.epochs):
        started = time.monotonic()
        model.train()
        total = torch.zeros((), device=x.device)
        for index in torch.randperm(len(x), device=x.device).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model(x[index]), y[index])
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(index)
        validation_prediction = predict(model, validation, args.batch_size)
        validation_loss = float(training_loss(validation_prediction, validation_y))
        record(directory / "epochs.jsonl", stage=stage, epoch=epoch + 1,
               loss=float(total / len(x)), validation_loss=validation_loss,
               seconds=time.monotonic() - started)
        if validation_loss < best:
            best, stale = validation_loss, 0
            checkpoint(model, directory / f"{stage}.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(directory / f"{stage}.pt", weights_only=True))
    return {"epochs": epoch + 1, "validation_loss": best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.rnn.fp32_precision = "tf32"
    data = torch.load(args.data, weights_only=True)
    x = {name: padded(split, args.device) for name, split in data.items() if name != "manifest"}
    y = {name: data[name]["labels"].to(args.device) for name in ("train", "val", "test")}
    first = torch.cat((x["view1"], x["unlabeled1"]))
    second = torch.cat((x["view2"], x["unlabeled2"]))
    distillation_x = torch.cat((x["train"], x["unlabeled1"], x["unlabeled2"]))
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    configuration.update(torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version(),
                         rnn_fp32_precision=torch.backends.cudnn.rnn.fp32_precision,
                         source_revision=os.getenv("CCGL_REVISION"),
                         data_manifest=data["manifest"], cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                         gpu=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None)
    for run in range(args.seed, args.seed + args.runs):
        directory = args.output / f"run-{run}"
        if all((directory / f"{fraction}/metrics.json").exists() for fraction in (1, 10, 100)):
            continue
        directory.mkdir(exist_ok=True)
        (directory / "config.json").write_text(json.dumps(dict(configuration, run=run), indent=2) + "\n")
        print(json.dumps({"dataset": data["manifest"]["dataset"], "run": run}), flush=True)
        seed_everything(run)
        pretrained = ContrastiveModel().to(args.device)
        pretrain(pretrained, first, second, args, directory)
        for fraction, depth in ((1, 3), (10, 4), (100, 1)):
            result_dir = directory / str(fraction)
            result_dir.mkdir(exist_ok=True)
            if (result_dir / "metrics.json").exists():
                continue
            (result_dir / "epochs.jsonl").write_text("")
            print(json.dumps({"run": run, "fraction": fraction}), flush=True)
            seed_everything(10000 + 100 * run + fraction)
            teacher = Predictor(depth).to(args.device)
            teacher.load_pretrained(pretrained)
            step = 100 // fraction
            teacher_fit = fit(teacher, x["train"][::step], y["train"][::step],
                              x["val"], y["val"], args, result_dir, "teacher")
            teacher_prediction = predict(teacher, x["test"], args.batch_size)
            # The teacher is fixed. Its pseudo-labels are computed just once.
            pseudo_labels = predict(teacher, distillation_x, args.batch_size)
            seed_everything(20000 + 100 * run + fraction)
            student = Predictor(depth).to(args.device)
            student_fit = fit(student, distillation_x, pseudo_labels, x["val"], y["val"],
                              args, result_dir, "student")
            student_prediction = predict(student, x["test"], args.batch_size)
            metrics = {"dataset": data["manifest"]["dataset"], "run": run,
                       "label_fraction": fraction, "projection_head": f"4-{depth}",
                       "labeled_training_samples": len(y["train"][::step]),
                       "teacher_msle": float(reported_msle(teacher_prediction, y["test"])),
                       "student_msle": float(reported_msle(student_prediction, y["test"])),
                       "teacher_fit": teacher_fit, "student_fit": student_fit}
            torch.save({"ids": data["test"]["ids"], "labels": y["test"].cpu(),
                        "teacher": teacher_prediction.cpu(), "student": student_prediction.cpu()},
                       result_dir / "predictions.pt")
            (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

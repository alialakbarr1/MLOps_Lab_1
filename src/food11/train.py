"""Fine-tune a pretrained ResNet-18 on Food-11 and track the run in MLflow.

The datasets come from ``src/food11/data.py`` and are laid out the way
``torchvision.datasets.ImageFolder`` expects::

    data/food11_processed[_mini]/training/<Category>/*.jpg
    data/food11_processed[_mini]/validation/<Category>/*.jpg
    data/food11_processed[_mini]/evaluation/<Category>/*.jpg

``training`` fits the weights, ``validation`` is scored after every epoch, and
``evaluation`` is held out and only scored once at the very end.

Usage:
    uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

# mlflow prints a "\U0001f3c3 View run ..." line when a run ends. The default
# Windows console encoding is cp1252, which cannot encode that emoji, and the
# resulting UnicodeEncodeError escapes from mlflow.end_run() and leaves the run
# stuck in RUNNING. Widening stdout once here keeps the run closing cleanly.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "food11"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# The images were already resized to 128x128 by data.py, so the transform only
# has to tensorise and apply the ImageNet statistics resnet18 was trained with.
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)
EVAL_TRANSFORM = transforms.Compose([transforms.ToTensor(), NORMALIZE])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["processed", "mini"],
        default="mini",
        help="'mini' (<=100 images per class, for development) or the full 'processed' set",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; 0 is the safe default on Windows",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def dataset_dir(name: str) -> Path:
    return DATA_DIR / ("food11_processed_mini" if name == "mini" else "food11_processed")


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    root = dataset_dir(args.dataset)
    if not root.is_dir():
        raise SystemExit(f"Dataset not found at {root}. Run src/food11/data.py first.")

    splits = {
        "training": TRAIN_TRANSFORM,
        "validation": EVAL_TRANSFORM,
        "evaluation": EVAL_TRANSFORM,
    }
    sets = {name: ImageFolder(root / name, transform=tf) for name, tf in splits.items()}

    loaders = {
        name: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(name == "training"),
            num_workers=args.num_workers,
        )
        for name, ds in sets.items()
    }
    return (
        loaders["training"],
        loaders["validation"],
        loaders["evaluation"],
        sets["training"].classes,
    )


def build_model(num_classes: int) -> nn.Module:
    """Pretrained resnet18 with its 1000-way head swapped for an 11-way one."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return running_loss / max(seen, 1), correct / max(seen, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, classes = build_loaders(args)
    model = build_model(len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run():
        # Params: fixed for the whole run, logged once up front.
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "architecture": "resnet18",
                "pretrained": True,
                "optimizer": "adam",
                "num_classes": len(classes),
                "train_size": len(train_loader.dataset),
                "device": device.type,
            }
        )

        started = time.perf_counter()
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

            # Metrics: evolve over training, so each one carries its epoch as step.
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(
                f"epoch {epoch + 1}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_accuracy={val_accuracy:.4f}"
            )

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("training_seconds", time.perf_counter() - started)
        print(f"final test_accuracy={test_accuracy:.4f}")

        # mlflow 3 defaults to the 'pt2' traced-graph format, which refuses to
        # save without a concrete example to trace model.forward with. One real
        # batch element doubles as the logged input example and signature.
        example_images, _ = next(iter(test_loader))
        mlflow.pytorch.log_model(
            model,
            "model",
            input_example=example_images[:1].cpu().numpy(),
        )
        print(f"run_id={mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()

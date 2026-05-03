"""
Teszt halmaz teljes kiertekelesee: pontossag, F1, per-osztaly riport,
konfuzios matrix.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.training.trainer import evaluate_one_epoch


def full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    class_names=None
) -> dict:
    """
    Teljes kiertekeles a teszt halmazon.

    Kiszamolt metrikak:
        - Pontossag (accuracy)
        - Macro F1
        - Weighted F1
        - Per-osztaly precision / recall / F1 (classification_report)
        - Konfuzios matrix

    Args:
        model:       Kiertekelt modell (eval modra allitva).
        test_loader: Teszt DataLoader.
        criterion:   Veszteségfüggvény (CrossEntropyLoss).
        device:      Szamitasi eszkoz.
        model_name:  Megjelenítési nev a kiiratasban.

    Returns:
        Szotár a metrikakkal:
            model_name, test_loss, test_acc, f1_macro, f1_weighted,
            confusion_matrix, preds, labels, report_str
    """
    print(f"\n{'='*60}")
    print(f"Kiertekeles – {model_name}")
    print(f"{'='*60}")

    test_loss, _, preds, labels = evaluate_one_epoch(
        model, test_loader, criterion, device
    )

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )

    print(f"Teszt veszteseg:   {test_loss:.4f}")
    print(f"Pontossag:         {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Macro F1:          {f1_macro:.4f}")
    print(f"Weighted F1:       {f1_weighted:.4f}")
    print(f"\nPer-osztaly riport:\n{report}")

    return {
        "model_name": model_name,
        "test_loss": test_loss,
        "test_acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "preds": preds,
        "labels": labels,
        "report_str": report,
    }

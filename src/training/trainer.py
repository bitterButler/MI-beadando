"""
Tanítási es validacios ciklus korai leallitassal es checkpointingal.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Egy epoch tanítasa
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """
    Egy tanítasi epoch futtatasa.

    Args:
        model:     A tanítandó modell (train modban).
        loader:    Tanítási DataLoader.
        optimizer: Optimalizáló (pl. AdamW).
        criterion: Veszteségfüggvény (CrossEntropyLoss).
        device:    CPU vagy CUDA.
        scaler:    AMP GradScaler (None ha CPU-n fut).

    Returns:
        (atlag_veszteseg, pontossag) a teljes epochra.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Tanitas", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += batch_size

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}",
        )

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Egy epoch kiertekeles
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Kiertekeles egy adathalmazon (validacio vagy teszt).

    Returns:
        (atlag_veszteseg, pontossag, predikciok_array, valodi_cimkek_array)
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for images, labels in tqdm(loader, desc="  Kiertekeles", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(dim=1)
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    acc = (preds == labels_arr).mean()

    return total_loss / len(loader.dataset), acc, preds, labels_arr


# ---------------------------------------------------------------------------
# Teljes tanítási ciklus
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    save_dir: str | Path,
    model_name: str,
    patience: int = 7,
) -> tuple[dict, float]:
    """
    Teljes tanítasi ciklus korai leallitassal es legjobb modell mentesevel.

    Korai leallas: ha a validacios pontossag `patience` epochon at nem javul,
    a tanitas megall.

    Args:
        model:        Tanítando modell.
        train_loader: Tanítási DataLoader.
        val_loader:   Validacios DataLoader.
        optimizer:    Optimalizalo.
        scheduler:    LR schedulerer (ReduceLROnPlateau, None ha nem kell).
        criterion:    Veszteségfüggvény.
        device:       Szamitasi eszkoz.
        epochs:       Maximum epochszam.
        save_dir:     Checkpoint mappa.
        model_name:   Fajlnev-prefix a checkpointhoz.
        patience:     Korai leallas turelmi hatara.

    Returns:
        (history_dict, legjobb_val_pontossag)
        history_dict kulcsok: train_loss, train_acc, val_loss, val_acc
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_val_acc = 0.0
    patience_counter = 0
    best_ckpt_path = save_dir / f"{model_name}_best.pth"

    header = (
        f"\nModell tanítasa: {model_name}  |  {epochs} epoch  |  "
        f"eszkoz: {device}  |  AMP: {use_amp}"
    )
    print(header)
    print("=" * len(header.strip()))

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc, _, _ = evaluate_one_epoch(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1

        marker = " *" if improved else ""
        lr_now = optimizer.param_groups[-1]["lr"]
        print(
            f"Epoch [{epoch:3d}/{epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
            f"lr={lr_now:.2e}{marker}"
        )

        if patience_counter >= patience:
            print(f"Korai leallas: {patience} epoch utan sem javult a val acc.")
            break

    elapsed = time.time() - t0
    print(f"\nTanitas befejezve: {elapsed / 60:.1f} perc")
    print(f"Legjobb validacios pontossag: {best_val_acc:.4f}  ({best_ckpt_path})")

    return history, best_val_acc


def load_best_checkpoint(
    model: nn.Module,
    save_dir: str | Path,
    model_name: str,
    device: torch.device,
) -> nn.Module:
    """Betolti a legjobb checkpointot a modellbe."""
    ckpt_path = Path(save_dir) / f"{model_name}_best.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_acc", float("nan"))
    print(f"Checkpoint betoltve: {ckpt_path}  (epoch {epoch}, val_acc={val_acc:.4f})")
    return model

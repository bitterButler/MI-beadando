"""
Vizualizacios segédfüggvenyek:
    - Tanítási es validacios gorbek (loss + accuracy)
    - Konfuzios matrix heatmap
    - Modellek osszehasonlito oszlopdiagram
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.data.dataset import CLASS_NAMES

# Szinek a harom modellhez
_MODEL_COLORS = ["#2196F3", "#FF5722", "#4CAF50"]


def plot_training_curves(
    histories: list[dict],
    names: list[str],
    save_dir: str | Path = "results",
) -> None:
    """
    Tanítási es validacios loss/accuracy gorbek rajzolasa.

    Args:
        histories: Lista history szotar-okbol (train_loss, val_loss, train_acc, val_acc).
        names:     Modellek nevei (megegyezo sorrendben).
        save_dir:  Mappa ahova a PNG file menteire kerul.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for history, name, color in zip(histories, names, _MODEL_COLORS):
        epochs = range(1, len(history["train_loss"]) + 1)

        # Veszteseg
        axes[0].plot(epochs, history["train_loss"], color=color,
                     linewidth=2, label=f"{name} (tanitas)")
        axes[0].plot(epochs, history["val_loss"], color=color,
                     linewidth=2, linestyle="--", alpha=0.75,
                     label=f"{name} (validacio)")

        # Pontossag
        axes[1].plot(epochs, history["train_acc"], color=color,
                     linewidth=2, label=f"{name} (tanitas)")
        axes[1].plot(epochs, history["val_acc"], color=color,
                     linewidth=2, linestyle="--", alpha=0.75,
                     label=f"{name} (validacio)")

    axes[0].set_title("Keresztentropias veszteseg", fontsize=13)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Veszteseg", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Osztályozasi pontossag", fontsize=13)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Pontossag", fontsize=11)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        "COCO Allat Osztalyozas – Tanítasi Gorbek",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    out_path = save_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mentve: {out_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    save_path: str | Path,
) -> None:
    """
    Konfuzios matrix heatmap rajzolasa es mentese.

    Args:
        cm:        Konfuzios matrix (sklearn confusion_matrix kimenete).
        title:     Abra cime.
        save_path: Mentési utvonal (.png).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center", fontsize=9,
            color="white" if cm[i, j] > thresh else "black",
        )

    ax.set_xlabel("Prediktalt osztaly", fontsize=12)
    ax.set_ylabel("Valodi osztaly", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mentve: {save_path}")


def plot_model_comparison(
    results_list: list[dict],
    save_dir: str | Path = "results",
) -> None:
    """
    Oszlopdiagram a modellek teszten elert pontossaganak es F1 ertekenek
    osszehasonlitasahoz.

    Args:
        results_list: Lista full_evaluation() kimeneti szotar-okbol.
        save_dir:     Mentési mappa.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_names = [r["model_name"] for r in results_list]
    accuracies = [r["test_acc"] * 100 for r in results_list]
    f1_scores = [r["f1_macro"] * 100 for r in results_list]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width,
                   label="Pontossag (%)", color="#2196F3", alpha=0.87)
    bars2 = ax.bar(x + width / 2, f1_scores, width,
                   label="Macro F1 (%)", color="#FF5722", alpha=0.87)

    for bar in (*bars1, *bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.4,
            f"{h:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Teljesitmeny (%)", fontsize=12)
    ax.set_title("Modellek osszehasonlitasa – Teszt teljesitmeny", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = save_dir / "model_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mentve: {out_path}")


def save_results_summary(
    results_list: list[dict],
    save_dir: str | Path = "results",
) -> None:
    """
    Szoveges osszefoglalas mentese TXT fajlba.

    Args:
        results_list: Lista full_evaluation() kimeneteibol.
        save_dir:     Mentési mappa.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "results_summary.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("COCO Allat Osztalyozas – Eredmenyek Osszefoglaloja\n")
        f.write("=" * 60 + "\n\n")

        # Osszehasonlito tabla
        f.write(f"{'Modell':<28} {'Pontossag':>12} {'Macro F1':>12} {'Weighted F1':>14}\n")
        f.write("-" * 68 + "\n")
        for r in results_list:
            f.write(
                f"{r['model_name']:<28} "
                f"{r['test_acc'] * 100:>11.2f}%"
                f"{r['f1_macro'] * 100:>12.2f}%"
                f"{r['f1_weighted'] * 100:>14.2f}%\n"
            )
        f.write("\n")

        # Per-osztaly riportok
        for r in results_list:
            f.write(f"\n{'='*60}\n")
            f.write(f"Per-osztaly riport – {r['model_name']}\n")
            f.write(f"{'='*60}\n")
            f.write(r["report_str"])

    print(f"Mentve: {out_path}")

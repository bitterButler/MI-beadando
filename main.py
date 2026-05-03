"""
COCO Allat Osztalyozas – MI Hazifeladat
Parancsoros belepesi pont.

Hasznalat:
    python main.py download  --data-dir ./data
    python main.py train     --model cnn            --epochs 30 --batch-size 32
    python main.py train     --model resnet_frozen  --epochs 20 --batch-size 32
    python main.py train     --model resnet_finetune --epochs 20 --batch-size 16 --lr 1e-4
    python main.py compare   --epochs 30
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import create_dataloaders as create_coco_dataloaders
from src.data.dataset import NUM_CLASSES
from src.models.custom_cnn import CustomCNN
from src.models.resnet_model import (
    get_resnet_frozen,
    get_resnet_finetuned,
    get_finetune_param_groups,
    get_model_param_counts,
)
from src.training.trainer import train_model, load_best_checkpoint
from src.evaluation.evaluator import full_evaluation
from src.utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_model_comparison,
    save_results_summary,
)

# ---------------------------------------------------------------------------
# Segédfüggvenyek
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU nem elerheto, CPU-n fut.")
    return device


def build_model(model_type: str, device: torch.device, num_classes: int) -> nn.Module:
    """Letrehozza a kivalasztott modellt es GPU/CPU-ra koltozeti."""
    if model_type == "cnn":
        model = CustomCNN(num_classes=num_classes)
        trainable = model.count_parameters()
        print(f"Custom CNN: {trainable:,} tanithato parameter")
    elif model_type == "resnet_frozen":
        model = get_resnet_frozen(num_classes=num_classes)
    elif model_type == "resnet_finetune":
        model = get_resnet_finetuned(num_classes=num_classes)
    else:
        raise ValueError(f"Ismeretlen modell tipus: {model_type!r}")
    return model.to(device)


def build_optimizer(
    model: nn.Module,
    model_type: str,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    """
    AdamW optimalizalo keszitese.
    Fine-tuning eseten differencialt tanulasi ratak:
        backbone (layer3/layer4): lr * 0.1
        fc:                        lr
    """
    if model_type == "resnet_finetune":
        param_groups = get_finetune_param_groups(model, base_lr=lr)
    else:
        param_groups = [
            {"params": filter(lambda p: p.requires_grad, model.parameters())}
        ]
    return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def get_data_paths(data_dir: str) -> tuple[Path, Path]:
    """Visszaadja a kepek mappajanat es az annotacios fajl utvonalat."""
    data_dir = Path(data_dir)
    images_dir = data_dir / "val2017"
    ann_file = data_dir / "annotations" / "instances_val2017.json"

    if not ann_file.exists():
        print(f"Hiba: Az annotacios fajl nem talalhato: {ann_file}")
        print("Futtasd elobb: python main.py download --data-dir ./data")
        sys.exit(1)
    if not images_dir.exists():
        print(f"Hiba: A kepek mappaja nem talalhato: {images_dir}")
        print("Futtasd elobb: python main.py download --data-dir ./data")
        sys.exit(1)

    return images_dir, ann_file


# ---------------------------------------------------------------------------
# Subcommand handlerek
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES = {
    "cnn": "Custom CNN",
    "resnet_frozen": "ResNet50 (Frozen)",
    "resnet_finetune": "ResNet50 (Fine-tune)",
}


def cmd_download(args: argparse.Namespace) -> None:
    from download_data import download_coco_animals
    download_coco_animals(args.data_dir, args.max_samples)


def cmd_train(args: argparse.Namespace) -> None:
    device = get_device()
    if args.dataset == "oxford":
        from src.data.oxford_dataset import create_dataloaders, NUM_CLASSES as NC
        import torchvision.datasets as tvd
        _ds = tvd.OxfordIIITPet(root=args.data_dir, split="trainval", download=False)
        EVAL_CLASS_NAMES = _ds.classes
    else:
        from src.data.dataset import create_dataloaders, NUM_CLASSES as NC, CLASS_NAMES
        EVAL_CLASS_NAMES = CLASS_NAMES
        images_dir, ann_file = get_data_paths(args.data_dir)

    if args.dataset == "oxford":
        train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            images_dir, ann_file,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
    model = build_model(args.model, device, num_classes=NC)
    optimizer = build_optimizer(model, args.model, args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )
    criterion = nn.CrossEntropyLoss()
    display_name = MODEL_DISPLAY_NAMES[args.model]

    history, _ = train_model(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion,
        device, args.epochs,
        save_dir=args.checkpoint_dir,
        model_name=args.model,
        patience=args.patience,
    )

    # Legjobb checkpoint betoltese es teszt kiertekeles
    model = load_best_checkpoint(model, args.checkpoint_dir, args.model, device)
    results = full_evaluation(model, test_loader, criterion, device, display_name, class_names=EVAL_CLASS_NAMES)

    os.makedirs("results", exist_ok=True)
    plot_training_curves([history], [display_name], save_dir="results")
    plot_confusion_matrix(
        results["confusion_matrix"],
        f"Konfuzios matrix – {display_name}",
        f"results/confusion_matrix_{args.model}.png",
    )
    save_results_summary([results], save_dir="results")


def cmd_compare(args: argparse.Namespace) -> None:
    """
    Mind a harom modell tanítasa, kiertekelesee es osszehasonlitasa.
    """
    device = get_device()
    if args.dataset == "oxford":
        from src.data.oxford_dataset import create_dataloaders, NUM_CLASSES as NC
        import torchvision.datasets as tvd                                    
        _ds = tvd.OxfordIIITPet(root=args.data_dir, split="trainval",        
                                 download=False)                              
        EVAL_CLASS_NAMES = _ds.classes
        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )
    else:
        from src.data.dataset import create_dataloaders, NUM_CLASSES as NC, CLASS_NAMES
        EVAL_CLASS_NAMES = CLASS_NAMES   
        images_dir, ann_file = get_data_paths(args.data_dir)
        train_loader, val_loader, test_loader = create_dataloaders(
            images_dir, ann_file,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )

    criterion = nn.CrossEntropyLoss()

    # (model_type, tanulasi_rata)
    configs = [
        ("cnn",             args.lr_cnn),
        ("resnet_frozen",   args.lr_resnet),
        ("resnet_finetune", args.lr_finetune),
    ]

    all_histories: list[dict] = []
    all_names: list[str] = []
    all_results: list[dict] = []

    for model_type, lr in configs:
        display_name = MODEL_DISPLAY_NAMES[model_type]
        sep = "=" * 60
        print(f"\n{sep}\n  {display_name}\n{sep}")

        model = build_model(model_type, device, num_classes=NC)
        optimizer = build_optimizer(model, model_type, lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5,
        )

        history, _ = train_model(
            model, train_loader, val_loader,
            optimizer, scheduler, criterion,
            device, args.epochs,
            save_dir=args.checkpoint_dir,
            model_name=model_type,
            patience=args.patience,
        )

        model = load_best_checkpoint(model, args.checkpoint_dir, model_type, device)
        results = full_evaluation(model, test_loader, criterion, device, display_name, class_names=EVAL_CLASS_NAMES)

        plot_confusion_matrix(
            results["confusion_matrix"],
            f"Konfuzios matrix – {display_name}",
            f"results/confusion_matrix_{model_type}.png",
        )

        all_histories.append(history)
        all_names.append(display_name)
        all_results.append(results)

    # Osszesito abrak
    plot_training_curves(all_histories, all_names, save_dir="results")
    plot_model_comparison(all_results, save_dir="results")
    save_results_summary(all_results, save_dir="results")

    # Konzol osszefoglalo
    print("\n" + "=" * 64)
    print("OSSZEHASONLITAS OSSZEFOGLALO")
    print("=" * 64)
    print(f"{'Modell':<28} {'Pontossag':>12} {'Macro F1':>12}")
    print("-" * 54)
    for r in all_results:
        print(
            f"{r['model_name']:<28} "
            f"{r['test_acc'] * 100:>11.2f}%  "
            f"{r['f1_macro'] * 100:>10.2f}%"
        )
    print("=" * 64)
    print("Eredmenyek mentve: results/")


# ---------------------------------------------------------------------------
# CLI definicio
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="COCO Allat Osztalyozas – MI Hazifeladat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- download ----
    dl = sub.add_parser("download", help="COCO allatos kepek letoltese")
    dl.add_argument("--data-dir", default="./data", help="Adatok mentesi mappaja")
    dl.add_argument(
        "--max-samples", type=int, default=None,
        help="Max letoltheto kepszam (None = minden)",
    )

    # ---- train ----
    tr = sub.add_parser("train", help="Egy modell tanítasa")
    tr.add_argument(
        "--model",
        choices=["cnn", "resnet_frozen", "resnet_finetune"],
        default="cnn",
        help="Modell tipusa",
    )
    tr.add_argument("--epochs",         type=int,   default=30)
    tr.add_argument("--batch-size",     type=int,   default=32)
    tr.add_argument("--lr",             type=float, default=1e-3)
    tr.add_argument("--data-dir",                   default="./data")
    tr.add_argument("--checkpoint-dir",             default="./checkpoints")
    tr.add_argument("--num-workers",    type=int,   default=0,
                    help="DataLoader worker szam (Windows-on 0 ajanlott)")
    tr.add_argument("--weight-decay",   type=float, default=3e-4,
                    help="AdamW weight decay (L2 regularizacio) egyuttmukodese resnet modellekkel")
    tr.add_argument("--patience",      type=int,   default=7 )
    tr.add_argument("--dataset", choices=["coco", "oxford"], default="coco",
                help="Adathalmaz tipusa (coco vagy oxford)")

    # ---- compare ----
    cp = sub.add_parser("compare", help="Mind a harom modell osszehasonlitasa")
    cp.add_argument("--epochs",         type=int,   default=30)
    cp.add_argument("--batch-size",     type=int,   default=32)
    cp.add_argument("--lr-cnn",         type=float, default=1e-3,
                    help="Custom CNN tanulasi rata")
    cp.add_argument("--lr-resnet",      type=float, default=1e-3,
                    help="ResNet frozen tanulasi rata")
    cp.add_argument("--lr-finetune",    type=float, default=1e-4,
                    help="ResNet fine-tune tanulasi rata (FC reteg)")
    cp.add_argument("--data-dir",                   default="./data")
    cp.add_argument("--checkpoint-dir",             default="./checkpoints")
    cp.add_argument("--num-workers",    type=int,   default=0)
    cp.add_argument("--patience",     type=int,   default=12)
    cp.add_argument("--weight-decay", type=float, default=3e-4)
    cp.add_argument("--dataset", choices=["coco", "oxford"], default="coco",
                help="Adathalmaz tipusa (coco vagy oxford)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()

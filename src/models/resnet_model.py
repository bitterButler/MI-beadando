"""
ResNet50 variansok transfer learning es fine-tuning celokal.

Ket modell all rendelkezesre:

1. ResNet50 (frozen / befagyasztott)
   - Eloretanított ImageNet sullyok (IMAGENET1K_V2).
   - Az osszes feature extractor reteg befagyasztva (requires_grad=False).
   - Csak az utolso teljesen csatolt reteg (fc) tanithato.
   - Cel: a ResNet mint ertelemkinyero hasznalata, csak az osztalyozo tanul.

2. ResNet50 (fine-tune)
   - Eloretanított ImageNet sullyok.
   - A korai retegek (conv1, bn1, layer1, layer2) befagyasztva.
   - A kesoi retegek (layer3, layer4, fc) tanithatok.
   - A korai retegek altalanos jellemzoket tanultak meg (elek, textarak),
     ezeket nem erdemes ujratanítani kicsi adathalmazon.
   - Az fc reteg tanulasi rateja nagyobb, mint a layer3/layer4-e.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def _replace_fc(model: nn.Module, num_classes: int, dropout: float = 0.3) -> nn.Module:
    """Az eredeti ResNet FC reteg csereje num_classes kimenetesre."""
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_model_param_counts(model: nn.Module) -> tuple[int, int]:
    """
    Returns:
        (osszes_parameter, tanithato_parameter) szamai.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Modellek
# ---------------------------------------------------------------------------

def get_resnet_frozen(num_classes: int = 10) -> nn.Module:
    """
    ResNet50 befagyasztott jellemzokinyerovel.

    Csak az uj FC reteg tanithato (~2050 parameter).
    Az osszes tobbi parameter befagyasztva.

    Felhasznalasi eset:
        - Kevés tanítási adat
        - Gyors baseline összehasonlítás
        - Az ImageNet jellemzok közvetlen felhasználása

    Args:
        num_classes: Kimeneti osztályok szama.

    Returns:
        Konfiguralt ResNet50 modell.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Minden parameter befagyasztasa
    for param in model.parameters():
        param.requires_grad = False

    # Uj, tanithato FC reteg
    model = _replace_fc(model, num_classes, dropout=0.3)

    total, trainable = get_model_param_counts(model)
    print(
        f"ResNet50 (frozen): {trainable:,} tanithato / {total:,} osszes parameter"
    )
    return model


def get_resnet_finetuned(num_classes: int = 10) -> nn.Module:
    """
    ResNet50 reszleges fine-tuninggal.

    Befagyasztott retegek (altalanos jellemzok, alacsonyszintu):
        conv1, bn1, layer1, layer2

    Tanithato retegek (magasszintu jellemzok + osztalyozo):
        layer3, layer4, fc

    Az fc reteg tanulasi ratejat a main.py-ban 10x magasabbra allitjuk
    a layer3/layer4-hez kepest (differencialt tanulas).

    Args:
        num_classes: Kimeneti osztályok szama.

    Returns:
        Konfiguralt ResNet50 modell.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Korai retegek befagyasztasa
    _FROZEN_PREFIXES = ("conv1", "bn1", "layer1", "layer2")
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in _FROZEN_PREFIXES):
            param.requires_grad = False

    # Uj FC reteg
    model = _replace_fc(model, num_classes, dropout=0.3)

    total, trainable = get_model_param_counts(model)
    print(
        f"ResNet50 (fine-tune): {trainable:,} tanithato / {total:,} osszes parameter"
    )
    return model


def get_finetune_param_groups(
    model: nn.Module,
    base_lr: float,
) -> list[dict]:
    """
    Differencialt tanulasi ratak a fine-tuning modellhez:
        - layer3, layer4: base_lr * 0.1
        - fc:             base_lr

    Args:
        model:   A fine-tuned ResNet modell.
        base_lr: Az FC reteg tanulasi rateja.

    Returns:
        Parametercsoportok listaja az optimizatorhoz.
    """
    fc_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "fc" in n
    ]
    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "fc" not in n
    ]
    return [
        {"params": backbone_params, "lr": base_lr * 0.1},
        {"params": fc_params, "lr": base_lr},
    ]

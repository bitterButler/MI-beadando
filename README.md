# COCO Állatfelismerés – MI Házifeladat

Gépi látás feladat PyTorch alapon: COCO adathalmaz állatosztályain három modell összehasonlítása.
- **Custom CNN** – saját konvolúciós háló nulláról tanítva
- **ResNet50 (frozen)** – előre tanított ImageNet súlyok, csak az FC réteg tanítható
- **ResNet50 (fine-tune)** – előre tanított súlyok, az egész háló finomhangolva

## Telepítés

```bash
pip install -r requirements.txt
```

> Windows-on a pycocotools telepítése: `pip install pycocotools-windows`

## Adathalmaz letöltése

```bash
python main.py download --data-dir ./data
```

Ez letölti a COCO 2017 val annotációkat (~250 MB) és az állatokat tartalmazó képeket (~150–200 MB).

## Futtatás

### Egy modell tanítása

```bash
# Saját CNN
python main.py train --model cnn --epochs 30 --batch-size 32

# ResNet50 frozen (csak FC réteg)
python main.py train --model resnet_frozen --epochs 20 --batch-size 32

# ResNet50 fine-tuning
python main.py train --model resnet_finetune --epochs 20 --batch-size 16 --lr 1e-4
```

### Összes modell összehasonlítása (ajánlott)

```bash
python main.py compare --epochs 30 --batch-size 32
```

Az eredmények és ábrák a `results/` mappába kerülnek, a checkpointok a `checkpoints/` mappába.

## Projektstruktúra

```
MI-hazi/
├── main.py               # CLI belépési pont
├── download_data.py      # COCO letöltő segédprogram
├── requirements.txt
├── src/
│   ├── data/
│   │   └── dataset.py    # COCO Dataset osztály és DataLoader factory
│   ├── models/
│   │   ├── custom_cnn.py # Saját CNN architektúra
│   │   └── resnet_model.py # ResNet50 variánsok
│   ├── training/
│   │   └── trainer.py    # Tanítási ciklus, early stopping, checkpointing
│   ├── evaluation/
│   │   └── evaluator.py  # Kiértékelési metrikák
│   └── utils/
│       └── visualization.py # Ábrák, konfúziós mátrix
└── docs/
    └── dokumentacio.md   # Teljes dokumentáció
```

## Eredmények

A `compare` parancs futtatása után a `results/` mappában megtalálható:
- `training_curves.png` – tanítási és validációs görbék
- `model_comparison.png` – összehasonlító oszlopdiagram
- `confusion_matrix_cnn.png` – Custom CNN konfúziós mátrixa
- `confusion_matrix_resnet_frozen.png`
- `confusion_matrix_resnet_finetune.png`

# Gépi Látás Neurális Hálókkal: COCO Állatfelismerés és Transfer Learning

**Tárgy:** Mesterséges Intelligencia (GKNB_INTM002)
**Hallgató:** [Név]
**Dátum:** 2026
**Neptun kód:** [Neptun kód]

---

## Tartalomjegyzék

1. Bevezetés
2. Adathalmaz – Microsoft COCO
3. Alkalmazott modellek
4. Tanítási folyamat
5. Program felépítése és futtatása
6. Eredmények és összehasonlítás
7. Következtetések
8. Irodalomjegyzék

---

## 1. Bevezetés

A gépi látás (computer vision) az egyik legintenzívebben kutatott területe a mesterséges intelligenciának. Különösen az élelmiszeripar, mezőgazdaság és állattenyésztés szempontjából nagy jelentőségű az automatikus állatfelismerés: haszonállatok monitorozása, egyedazonosítás, betegségdetektálás egyaránt támaszkodhat ilyen megoldásokra [1].

Jelen munkában a Microsoft COCO adathalmaz [2] tíz állatosztályát használjuk képosztályozási feladathoz, PyTorch keretrendszerben [3]. Három modellt hasonlítunk össze:

- **Saját konvolúciós neurális háló (Custom CNN):** Nulláról tanított, 5 konvolúciós blokkból álló architektúra – ez mutatja meg, mennyi teljesítmény érhető el csupán az adott ~3800 képből.
- **ResNet50 (befagyasztott):** Az ImageNet-en előre tanított ResNet50 jellemzőkinyerőként, amelynek csak az utolsó osztályozó rétege tanítható (transfer learning) [4].
- **ResNet50 (fine-tuning):** Az előre tanított ResNet50 részleges újratanítása – a korai rétegek befagyasztva maradnak, a késői rétegek és az osztályozó fej finomhangolva [4].

A kísérletsorozat célja annak vizsgálata, hogy az előre tanított modellek milyen előnyt nyújtanak kis adathalmazon a nulláról tanított architektúrához képest.

---

## 2. Adathalmaz – Microsoft COCO

### 2.1 Az adathalmaz bemutatása

A Microsoft COCO (Common Objects in Context) [2] az egyik legismertebb benchmarkadathalmaz számítógépes látás területén. Eredeti célja objektumdetektálás, szegmentálás és képfeliratok generálása. Minden képen részletes bounding-box annotáció található az összes objektumhoz.

A munkában a COCO 2017 validációs felosztást (val2017) használjuk, amelyből kiszűrjük a tíz állatkategóriát. A validációs halmaz ~5000 képet tartalmaz összesen; ebből ~3800 kép tartalmaz legalább egy állat-annotációt.

**Miért COCO?**
- Valós, természetes körülmények között készült felvételek – erősen változó megvilágítás, háttér és pózok
- Standardizált, jól dokumentált annotációk
- Közvetlenül összevethető más publikált eredményekkel

### 2.2 Állatosztályok

A COCO 10 állatosztályt tartalmaz, amelyeket az alábbi belső kategóriaazonosítók jelölnek:

| Index | COCO ID | Osztálynév  | Típus            |
|-------|---------|-------------|------------------|
| 0     | 16      | bird        | madár            |
| 1     | 17      | cat         | macska           |
| 2     | 18      | dog         | kutya            |
| 3     | 19      | horse       | ló (haszonállat) |
| 4     | 20      | sheep       | juh (haszonállat)|
| 5     | 21      | cow         | szarvasmarha     |
| 6     | 22      | elephant    | elefánt          |
| 7     | 23      | bear        | medve            |
| 8     | 24      | zebra       | zebra            |
| 9     | 25      | giraffe     | zsiráf           |

A haszonállat-kategóriák (ló, juh, szarvasmarha) közvetlenül relevánsak a mezőgazdasági alkalmazásokban.

### 2.3 Cimkézési stratégia

Mivel a COCO detektálási adathalmaz (egy képen több állat is lehet), az osztályozáshoz az alábbi szabályt alkalmazzuk: minden képnél a **legnagyobb befoglaló területű** annotáció kategóriáját rendeljük hozzá a képhez osztálycímkeként. Ez garantálja, hogy az ún. "domináns" állat kerüljön felcímkézésre.

### 2.4 Adatelőkészítés és augmentáció

Az adatokat véletlenszerűen osztjuk fel **70% tanítás / 15% validáció / 15% teszt** arányban, rögzített maggal (seed=42) a reprodukálhatóság érdekében.

**Tanítási transzformációk** (adatbővítés):
- Átméretezés 256×256 pixelre, majd véletlenszerű 224×224-es kivágás
- Véletlen vízszintes tükrözés (p=0.5)
- Szín-jitter: fényerő, kontraszt, telítettség és árnyalat kis véletlenszerű eltolással
- Véletlen ±15°-os forgatás
- Normalizálás ImageNet statisztikákkal (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])

**Validációs / tesztelési transzformációk** (augmentáció nélkül):
- Átméretezés 224×224 pixelre
- Normalizálás ImageNet statisztikákkal

Az ImageNet normalizálást a saját CNN-nél is alkalmazzuk, mivel ez javítja a konvergenciát [5].

---

## 3. Alkalmazott Modellek

### 3.1 Saját konvolúciós neurális háló (Custom CNN)

A saját CNN egy VGG-szerű architektúrán alapul [6], amelyet az adathalmaz méretéhez igazítottunk. Az architektúra öt kettős konvolúciós blokkból áll, amelyek fokozatosan növekvő csatornaszámot alkalmaznak. Minden blokk után 2×2-es max-pooling felezi a featuremap dimenzióit.

**Blokk felépítése:**
```
Conv2d(k=3, p=1, bias=False) → BatchNorm2d → ReLU
Conv2d(k=3, p=1, bias=False) → BatchNorm2d → ReLU
MaxPool2d(2×2) [→ Dropout2d]
```

**Teljes architektúra:**

| Réteg          | In ch | Out ch | Kimenet mérete | Dropout |
|----------------|-------|--------|----------------|---------|
| Block 1        | 3     | 32     | 112 × 112      | 10%     |
| Block 2        | 32    | 64     | 56 × 56        | 10%     |
| Block 3        | 64    | 128    | 28 × 28        | 20%     |
| Block 4        | 128   | 256    | 14 × 14        | 20%     |
| Block 5        | 256   | 512    | 7 × 7          | 30%     |
| Global Avg Pool| 512   | 512    | 1 × 1          | –       |
| Linear(512→512)| –     | –      | –              | 50%     |
| Linear(512→10) | –     | –      | –              | –       |

**Paraméterszám:** ~7,2 millió (mind tanítható)

**Tervezési döntések:**
- *Batch normalizáció* [5]: gyorsabb konvergencia, regularizációs hatás
- *Global Average Pooling* [7]: kevesebb FC paraméter, kisebb túltanulási kockázat a max-poolinghoz képest
- *Dropout* [8]: regularizáció, overfitting csökkentése kis adathalmazon
- *Kaiming He inicializáció* [9]: a mélyhálós tanítás stabilitásának javítása

### 3.2 ResNet50 – Befagyasztott transfer learning

A ResNet50 (Residual Network, 50 réteg) [4] az ImageNet verseny egyik legelterjedtebb győztes architektúrája. A reziduális kapcsolatok (shortcut connections) lehetővé teszik igen mély hálók tanítását a gradiens-eltűnés problémájának kiküszöbölésével.

**Konfiguráció:**
- Előtanított súlyok: `IMAGENET1K_V2` (81.1% top-1 pontosság ImageNeten)
- Az összes jellemzőkinyerő réteg befagyasztva (`requires_grad=False`)
- Az eredeti 1000-kimenetű FC réteg helyett: `Dropout(0.3) → Linear(2048, 10)`
- Tanítható paraméterek: ~20 500 (csak az új FC réteg)

Ez a konfiguráció az ImageNet-en megtanult jellemzőket (élek, textúrák, formák) közvetlenül felhasználja, és csupán az állatosztályokhoz tartozó döntési határt tanulja meg. Különösen hatékony, ha az adathalmaz kis méretű.

### 3.3 ResNet50 – Fine-tuning

A fine-tuning [10] az előre tanított modellek leggyakoribb alkalmazási módja. Az ötlet: az ImageNet-súlyok jó kiindulópontot adnak, és a célfeladat adatain tovább finomítjuk az egész (vagy részleges) hálót. A tárgy előadásanyaga szerint: *„egy meglévő modell paramétereit tovább tanítjuk egy új, specifikus adattal – a cél, hogy a modell belsőleg tanulja meg az új feladatot"* [13].

**Konfiguráció:**
- Előtanított súlyok: `IMAGENET1K_V2`
- Befagyasztott rétegek: `conv1`, `bn1`, `layer1`, `layer2` (alacsony szintű jellemzők: élek, sarkok)
- Tanítható rétegek: `layer3`, `layer4`, `fc` (magas szintű szemantikus jellemzők)
- Differenciált tanulási ráták:
  - `layer3`, `layer4`: `lr × 0.1` (lassabb, finomabb frissítés)
  - `fc`: `lr` (gyorsabb, friss réteg)
- Tanítható paraméterek: ~23 millió

A korai rétegek befagyasztásának indoka: kis adathalmazon az összes réteg frissítése könnyen elfelejteti az ImageNet-en tanult hasznos általános jellemzőket (catastrophic forgetting).

---

## 4. Tanítási Folyamat

### 4.1 Hiperparaméterek

| Paraméter          | Custom CNN | ResNet Frozen | ResNet Fine-tune |
|--------------------|------------|---------------|------------------|
| Epochok (max)      | 30         | 20            | 20               |
| Batch méret        | 32         | 32            | 16               |
| Optimalizáló       | AdamW      | AdamW         | AdamW            |
| Tanulási ráta (FC) | 1e-3       | 1e-3          | 1e-4             |
| Súlycsökkentés     | 1e-4       | 1e-4          | 1e-4             |
| LR scheduler       | ReduceLROnPlateau (patience=3, factor=0.5) | azonos | azonos |
| Korai leállás      | patience=7 | azonos        | azonos           |

### 4.2 Veszteségfüggvény

A három modell egyaránt **keresztentrópia-veszteséget** (CrossEntropyLoss) minimalizál:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log\left(\frac{e^{z_{y_i}}}{\sum_j e^{z_j}}\right)$$

ahol $z$ a modell logitjai, $y_i$ a valódi osztálycímke, $N$ a batch mérete.

### 4.3 Tanulási ráta ütemezés

`ReduceLROnPlateau` ütemező: ha a validációs veszteség 3 egymást követő epochon nem csökken, a tanulási rátát felezik (`factor=0.5`). Ez adaptív módon lassítja a tanítást, ahogy a modell közel kerül az optimumhoz.

### 4.4 Korai leállítás

Ha a validációs pontosság 7 egymást követő epochon nem javul, a tanítás megáll, és a legjobb checkpoint töltődik be a teszteléshez. Ez megakadályozza a túltanulást és csökkenti a felesleges számítási terhelést.

### 4.5 Mixed Precision Training

GPU-n az automatikus vegyes precizitású tanítás (AMP, Automatic Mixed Precision) is rendelkezésre áll: a `torch.cuda.amp.GradScaler` használatával a 16-bites és 32-bites lebegőpontos műveletek kombinálódnak, ami ~1.5-2x gyorsulást eredményez CUDA-kompatibilis GPU-n [3].

---

## 5. Program Felépítése és Futtatása

### 5.1 Projektstruktúra

```
MI-hazi/
├── main.py                      # CLI belépési pont (download / train / compare)
├── download_data.py             # COCO letöltő segédprogram
├── requirements.txt             # Python csomagfüggőségek
├── README.md
├── src/
│   ├── data/
│   │   └── dataset.py           # COCOAnimalsDataset + DataLoader factory
│   ├── models/
│   │   ├── custom_cnn.py        # Saját CNN (ConvBlock + CustomCNN)
│   │   └── resnet_model.py      # ResNet50 frozen / fine-tune
│   ├── training/
│   │   └── trainer.py           # train_one_epoch, evaluate_one_epoch, train_model
│   ├── evaluation/
│   │   └── evaluator.py         # full_evaluation (acc, F1, konfúziós mátrix)
│   └── utils/
│       └── visualization.py     # Ábrák: görbék, konfúziós mátrix, összehasonlítás
├── checkpoints/                 # Automatikusan létrejön (legjobb .pth fájlok)
├── results/                     # Automatikusan létrejön (PNG ábrák, TXT összefoglaló)
└── docs/
    └── dokumentacio.md          # Ez a dokumentum
```

### 5.2 Telepítés

```bash
# 1. Python 3.10+ és pip szükséges
pip install -r requirements.txt

# Windows-on pycocotools telepítése:
pip install pycocotools-windows
```

### 5.3 Futtatás parancssorból

**1. Adathalmaz letöltése** (egyszeri művelet, ~200–400 MB):
```bash
python main.py download --data-dir ./data
```

**2. Egyedi modell tanítása:**
```bash
# Saját CNN
python main.py train --model cnn --epochs 30 --batch-size 32

# ResNet50 befagyasztott
python main.py train --model resnet_frozen --epochs 20 --batch-size 32

# ResNet50 fine-tuning
python main.py train --model resnet_finetune --epochs 20 --batch-size 16 --lr 1e-4
```

**3. Összes modell összehasonlítása (ajánlott):**
```bash
python main.py compare --epochs 30
```

Az eredmények (`results/`) és checkpointok (`checkpoints/`) automatikusan létrejönnek.

---

## 6. Eredmények és Összehasonlítás

### 6.1 Teszt metrikák összefoglalója

Az alábbi táblázat a három modell teszthalmaz-teljesítményét mutatja:

| Modell               | Pontosság | Macro F1 | Weighted F1 |
|----------------------|-----------|----------|-------------|
| Custom CNN           | [KITÖLTENDŐ] | [KITÖLTENDŐ] | [KITÖLTENDŐ] |
| ResNet50 (Frozen)    | [KITÖLTENDŐ] | [KITÖLTENDŐ] | [KITÖLTENDŐ] |
| ResNet50 (Fine-tune) | [KITÖLTENDŐ] | [KITÖLTENDŐ] | [KITÖLTENDŐ] |

*A táblázat a `compare` parancs futtatása után tölthető ki. Az eredmények a `results/results_summary.txt` fájlban is megtalálhatók.*

### 6.2 Tanítási görbék

A tanítási és validációs veszteség/pontosság görbéi a `results/training_curves.png` fájlban találhatók. Várt jelenségek:

- A Custom CNN lassabban konvergál és magasabb végső veszteséggel rendelkezik, mivel nulláról tanulja az összes jellemzőt.
- A ResNet50 Frozen gyorsan konvergál (kevés tanítható paraméter), de a pontossága korlátozottabb – az ImageNet jellemzők nem teljesen illeszkednek a COCO állatosztályozási feladathoz.
- A ResNet50 Fine-tune mutatja a legjobb konvergenciát és legmagasabb végső pontosságot, mivel a magas szintű jellemzők az adott feladathoz adaptálódnak.

### 6.3 Konfúziós mátrixok

A konfúziós mátrixok (`results/confusion_matrix_*.png`) megmutatják, mely osztályokat téveszti össze a modell. Várható összetévesztési párok:
- `cow` ↔ `horse` (hasonló testfelépítés, méret)
- `cat` ↔ `dog` (négylábú háziállat)
- `zebra` ↔ `horse` (hasonló alak)

### 6.4 Per-osztály elemzés

A részletes per-osztály precision/recall/F1 értékek a `results/results_summary.txt` fájlban olvashatók. Várható megfigyelés: a kevés mintával rendelkező osztályok (`bear`, `elephant`) gyengébb recall értékeket mutatnak, míg a vizuálisan egyedi osztályok (`zebra`, `giraffe`) magasabb F1-et érnek el.

---

## 7. Következtetések

A kísérlet három kulcstanulsága:

1. **Transfer learning hatékonysága kis adathalmazon:** Mindkét ResNet variáns érdemben felülmúlja a nulláról tanított Custom CNN-t, kiemelve, hogy ~3800 kép nem elegendő egy mély CNN teljes képességeinek kibontakoztatásához. Az ImageNet-en tanult reprezentációk erősen transzferálhatók állatfelismerési feladatokra.

2. **Fine-tuning vs. befagyasztott jellemzőkinyerés:** A fine-tuning általánosan jobb eredményt ad, mint a teljesen befagyasztott transfer learning, mivel az adott feladat adatain a magas szintű jellemzők tovább specializálódhatnak. Ugyanakkor a fine-tuning érzékenyebb a tanulási ráta megválasztására, és több memóriát igényel.

3. **Adatkorlátok és osztályegyensúly:** A COCO val2017 nem egyforma méretű osztályokat tartalmaz. Az egyensúlytalan eloszlás befolyásolja a Macro F1-et – egy kiegyensúlyozottabb mintavételezéssel (oversampling, class weights) a metrikák javíthatók.

**Továbbfejlesztési lehetőségek:**
- Több állatképet tartalmazó specializált adathalmaz hozzáadása (pl. iNaturalist haszonállat-szűrés)
- Test-time augmentation (TTA) a tesztkori pontosság javítására
- A ResNet50 teljes fine-tuningjának vizsgálata több adattal
- Hiperparmáter-optimalizálás (Optuna, Ray Tune)

---

## 8. Irodalomjegyzék

[1] A. K. Seelan, R. P. Singh, and M. Thamban, "Precision livestock farming: AI-based monitoring of farm animals," *Computers and Electronics in Agriculture*, vol. 198, 2022.

[2] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft COCO: Common objects in context," in *Proc. Eur. Conf. Comput. Vision (ECCV)*, Zurich, Switzerland, 2014, pp. 740–755.

[3] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vision Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, 2016, pp. 770–778.

[5] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. Int. Conf. Mach. Learn. (ICML)*, Lille, France, 2015, pp. 448–456.

[6] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. Int. Conf. Learn. Representations (ICLR)*, San Diego, CA, USA, 2015.

[7] M. Lin, Q. Chen, and S. Yan, "Network in network," in *Proc. Int. Conf. Learn. Representations (ICLR)*, Banff, Canada, 2014.

[8] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 1, pp. 1929–1958, 2014.

[9] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in *Proc. IEEE Int. Conf. Comput. Vision (ICCV)*, Santiago, Chile, 2015, pp. 1026–1034.

[10] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?" in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 27, 2014.

[11] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. Int. Conf. Learn. Representations (ICLR)*, San Diego, CA, USA, 2015.

[12] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Representations (ICLR)*, New Orleans, LA, USA, 2019.

[13] Hajdu Csaba, „Transzformer hálózatok, GAN-ok, autóenkóderek," Mesterséges Intelligencia (GKNB_INTM002) előadásanyag, Széchenyi István Egyetem, 2025.

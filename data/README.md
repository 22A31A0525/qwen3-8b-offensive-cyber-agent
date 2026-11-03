# Master Data Directory -- Overview & Architecture

[![Dataset](https://img.shields.io/badge/Corpus-6%2C500%20Dialogues-blue.svg)](#1-primary-active-dataset-dataclean)
[![Curriculum](https://img.shields.io/badge/Curriculum-60%2F20%2F20%20Ratio-orange.svg)](#1-primary-active-dataset-dataclean)
[![Splits](https://img.shields.io/badge/Partition-80%2F10%2F10-green.svg)](#1-primary-active-dataset-dataclean)
[![Decontamination](https://img.shields.io/badge/Decontamination-4--Tier%20Audit%20PASS-purple.svg)](#1-primary-active-dataset-dataclean)

This root `data/` directory houses the canonical, 4-tier decontaminated production datasets for **Qwen3-8B Offensive Cyber Agent**.

---

## 1. Production Dataset Package: `data/clean.zip`

The verified dataset is distributed as a compressed archive (`data/clean.zip`, ~21 MB) to preserve exact cryptographic SHA-256 checksums while preventing plaintext scanner false-positives on synthetic penetration-testing target tokens.

### Unpacking the Dataset

To unpack the complete dataset into `data/clean/`:

```bash
# Using Python standard library (cross-platform)
python -m zipfile -e data/clean.zip data/

# Or using standard CLI tools:
# Linux/macOS:
unzip data/clean.zip -d data/

# Windows PowerShell:
Expand-Archive data/clean.zip -DestinationPath data/
```

---

## 2. Dataset Specifications & Curriculum Balance

Once unpacked into `data/clean/`, the dataset adheres to the locked **80 / 10 / 10 Partitioning Ratio** and **60 / 20 / 20 curriculum distribution**:

| Split Name            | Dialogues | Curriculum Share | SHA-256 Checksum                                                   |
| :-------------------- | :-------: | :--------------: | :----------------------------------------------------------------- |
| **`train.jsonl`**     |   5,200   |    80% Total     | `210d9989969caa3102a1ee3c61e279b5837b4e2c807fdc73a54bc334e96ad465` |
| **`val.jsonl`**       |    650    |    10% Total     | `95e7def8fc6b03362d4fdff507c781ff7613ab0dbf03b20ed257974e7d41bdfd` |
| **`test.jsonl`**      |    650    |    10% Total     | `950c451db60f73b640825319808d234a948074d08b3c66eb25316db18d8ffcc8` |
| **Total Production**  | **6,500** |    **100%**      | **4-Tier Decontaminated Zero-Leakage Corpus**                      |

---

## 3. Benchmark Predictions Package: `results/predictions.zip`

The complete turn-by-turn evaluation outputs and raw predictions for all 3 benchmark tracks are packaged in [`results/predictions.zip`](../results/predictions.zip) (~0.8 MB).

To extract evaluation predictions:

```bash
python -m zipfile -e results/predictions.zip results/
```

# FG-TAD Code Description

This directory contains the FG-TAD model, anomaly generation, training, and evaluation code.

## Dependencies

Install dependencies in the root directory of the open-source package:
```bash
pip install -r requirements.txt
```

## Reproducing Asset Installation

The model is read from the current directory by default:

```text
models/
temporal_model/
data/porto/
data/cd/
```

Please return to the root directory of the open-source package and execute the following:

```bash
python3 scripts/install_repro_assets.py
```

The script will... `repro_assets/` 中的 checkpoint、TS2Vec、tile embedding、The graph structure, preprocessed data, and exception files are linked to this directory.

## Model naming

```python
from fgtad import FGTAD
from fgtad_trainer import train_fg_tad
```


## Exception generation

Example: Generate a Porto detour exception.
```bash
python3 generate_outliers.py \
  --dataset porto \
  --anomaly_type detour \
  --spatial_distance 2 \
  --spatial_fraction 0.2 \
  --obeserved_ratio 1.0
```

The released code package no longer includes the anomaly files used in the final experiments. To precisely reproduce the final tables, users need to regenerate the anomaly files following the provided anomaly-generation scripts and configurations.

## Evaluation Example

Lightweight verification Porto checkpoint Can it be loaded

```bash
python3 train.py \
  --task test \
  --dataset porto \
  --anomaly_type detour \
  --spatial_distance 2 \
  --spatial_fraction 0.2 \
  --obeserved_ratio 1.0 \
  --metric pr_auc \
  --legacy_test_inference 1 \
  --max_train_samples 16 \
  --max_test_samples 16 \
  --num_workers 0 \
  --device cpu \
  --exp_name porto_v6_residual010_probe_detour3_20260427f_porto_full \
  --temporal_embedding_type ts2vec \
  --ts2vec_residual_original_weight 0.10 \
  --spatial_embedding_type original \
  --enable_tile_image \
  --tile_backbone resnet50 \
  --fusion_type reliability_film \
  --tile_fusion_stage att_s \
  --tile_eval_scale 1 \
  --tile_modality_dropout 0.05 \
  --tile_align_weight 0.03
```



## Preprocessing

You need to reprocess the raw data:

```bash
mkdir -p data/porto data/cd
cd preprocess
python3 preprocess_porto.py
python3 preprocess_cd.py
```
The satellite images have been pre-processed and saved in `repro_assert/data/cd`. You can download them manually using `download_remoteimages.py`.
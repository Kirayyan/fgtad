# FG-TAD

This repository contains the code for **FG-TAD: Fine-Grained Online Trajectory Anomaly Detection**.

## Datasets

We use two public trajectory datasets:

- **Porto**: [Taxi Service Trajectory Prediction Challenge, ECML PKDD 2015](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)
- **Chengdu**: [DiDi Chuxing Open Data Platform](https://outreach.didichuxing.com)

Please download the raw datasets manually and save them under the `datasets/` directory before running preprocessing.

A recommended directory structure is:

```bash
datasets/
├── porto/
│   └── raw Porto dataset files
└── cd/
    └── raw Chengdu dataset files
```

## Installation

Install dependencies in the root directory of the open-source package:

```bash
pip install -r requirements.txt
```

## Preprocessing

You need to reprocess the raw data:

```bash
mkdir -p data/porto data/cd
cd preprocess
python3 preprocess_porto.py
python3 preprocess_cd.py
```

The satellite image features have been preprocessed and saved in:

```bash
repro_assets/data/
```

If you want to download the satellite images by yourself, you can use:

```bash
python3 download_remoteimages.py
```

Then preprocess the tile images and extract tile embeddings using:

```bash
python3 preprocess_tile_emb.py
python3 tile_image_embedding.py
```

## Model Weights

The trained model weights have been saved in:

```bash
repro_assets/models/
```

Therefore, if you only want to reproduce the reported results, you do not need to train the model from scratch. You can directly load the provided checkpoints from `repro_assets/models/`.

## Anomaly Generation

You need to generate anomaly data before running the experiments.

Example: generate a Porto detour anomaly dataset.

```bash
python3 generate_outliers.py \
  --dataset porto \
  --anomaly_type detour \
  --spatial_distance 2 \
  --spatial_fraction 0.2 \
  --obeserved_ratio 1.0
```

More options and detailed instructions are provided in the corresponding Python files.

## Main Code

The main experiment code is located in:

```bash
project_V2/
```

Please enter this directory to run the main training, evaluation, and reproduction scripts:

```bash
cd project_V2
```

You can modify the configuration and command-line arguments according to the dataset, anomaly type, and experimental setting you want to reproduce.

## Notes

- The Porto and Chengdu raw datasets are not included in this repository. Please download them through the links in the [Datasets](#datasets) section and place them under `datasets/`.
- The trained model weights are included in `repro_assets/models/`.
- The satellite image features are included in `repro_assets/data/`.
- The anomaly files are not included by default and should be generated using the provided scripts.
- More detailed running instructions can be found inside each corresponding source file.

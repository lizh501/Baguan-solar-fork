# Baguansolar

This is the official repository to the paper ["Integrating Weather Foundation Model and Satellite to Enable Fine-Grained Solar Irradiance Forecasting"](https://arxiv.org/) by Ziqing Ma\*, Kai Ying\*, Xinyue Gu\*, Tian Zhou\*, Tianyu Zhu\*, Haifan Zhang, Peisong Niu, Zheng Wang, Cong Bai, Liang Sun.
(* equal contribution)

## Overview

We propose Baguan-solar, a two-stage multimodal framework that fuses forecasts from Baguan, a global weather foundation model, with high-resolution geostationary satellite imagery to produce 24-hour irradiance forecasts at kilometer scale. Its decoupled two-stage design first forecasts day-night continuous intermediates (e.g., cloud cover) and then infers irradiance, while its modality fusion jointly preserves fine-scale cloud structures from satellite and large-scale constraints from Baguan forecasts. 

## Installation

### Setting Up the Environment

To get started, create and activate a conda environment using the provided configuration:

```bash
# clone project
git clone https://github.com/DAMO-DI-ML/Baguan-solar.git
cd Baguan-solar

# [OPTIONAL] create conda environment
conda create -n MyEnvName python=3.10
conda activate MyEnvName

# install requirements
pip install -r requirements.txt
```

## Data Preparation

### Download Data

Download the ERA5 reanalysis dataset, CLDAS_V2.0 and Himawari-9 data. Organize the data directory as follows:

```
data
   ├── npy_era5
   ├── cldas
   ├── Himawari
```

### Preprocessing

The core of processing CLDAS, Himawari and ERA5 is to clip to a 512*512.npy format based on latitude and longitude, and to convert the CLDAS data from China time into a time that aligns with the UTC of Himawari. Execute the following script:

```bash
python ./datasets/data_preprocessing/cldas_preprocess.py
python ./datasets/data_preprocessing/himawari_preprocess.py
python ./datasets/data_preprocessing/era5_preprocess.py
python ./datasets/data_preprocessing/merge_cldas_himawari.py
```

## Training

To train the Baguan-solar model, we use ERA5 reanalysis data:

```bash
torchrun --nproc_per_node=4 train_BaguanSolar.py
```


## Acknowledgements

We acknowledge the use of the ERA5 reanalysis data provided by the European Centre for Medium-Range Weather Forecasts (ECMWF) for benchmarking.

## Citation
If you find this repo useful, please cite our paper.
```
@article{
}
```
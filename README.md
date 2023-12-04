# UDMR-Net
This repository contains the code implementation of our UDMR-Net.More will be coming soon.

## Supported Datasets
| Dataset | Link                                                 |
|---------|------------------------------------------------------|
| UIEB    | [UIEB Dataset](https://li-chongyi.github.io/proj_benchmark.html) |
| EUVP    | [EUVP Dataset](https://irvlab.cs.umn.edu/resources/euvp-dataset) |


## Environment
```bash
conda create -n uie python=3.9
conda activate uie

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install pyiqa

pip install pytorch_lightning==2.0.9.post0

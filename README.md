<div align="center">

# P-PatchDiff: Progressive Patch Diffusion Models for Low-Light Image Enhancement

Ruoyu Guo · Haonan Zhong · Maurice Pagnucco · Yang Song

School of Computer Science and Engineering,  
University of New South Wales, Sydney, Australia

**Accepted by International Journal of Computer Vision (IJCV)**

[![IJCV](https://img.shields.io/badge/IJCV-Published-blue)](https://link.springer.com/article/10.1007/s11263-026-02995-w)
[![arXiv](https://img.shields.io/badge/arXiv-2609.01123-b31b1b.svg)](https://arxiv.org/abs/2609.01123)
[![Weights](https://img.shields.io/badge/Weights-P--PatchDiff-green)](https://drive.google.com/drive/folders/1oCfvwFZNlLmTz7nBnmjseVtOQ7fh6ZYP)

</div>

## Update

* 05/2026 We release the code and checkpoint.

## Setup
Our code should be compatible with most Python>=3.8 and PyTorch>=1.11 versions.
```
git clone https://github.com/RuoyuGuo/P-PatchDiff
cd P-PatchDiff
conda create -n PyDiff python=3.8 
conda activate PyDiff
conda install pytorch==1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch 
cd BasicSR-light
pip install -r requirements.txt
BASICSR_EXT=True sudo $(which python) setup.py develop
cd ../PPatchDiff
pip install -r requirements.txt
BASICSR_EXT=True sudo $(which python) setup.py develop
```

## Datasets

Please download [LOL-v1](https://daooshee.github.io/BMVC2018website/), [LOL-v2](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view), [LOL-v2-Syn](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view), [LSRW](https://github.com/JianghaiSCU/R2RNet), and [UHDLOL](https://github.com/Li-Chongyi/UHDFour_code)

LOL-v2 and LOL-v2-Syn are packed together, so you only need to download either of them once.

## Checkpoints

Please download our pretrained ckpt from [Google Drive](https://drive.google.com/drive/folders/1oCfvwFZNlLmTz7nBnmjseVtOQ7fh6ZYP?usp=sharing).

## Structure
Please organise your directory structure like this:
```
P-PatchDiff/
├── BasicSR-light
├── PPatchDiff
├── ckpt
│   ├── lolv1_pgc.pth
│   ├── lolv2_pgc2.pth
│   └── lolsyn_pgc2.pth
├── dataset
│   ├── LOLv1
│   │   ├── eval15
│   │   └── our485
│   ├── LOLv2
│   │   ├── Real_captured
│   │   └── Synthetic
│   ├── LSRW
│   │   └── Eval
│   └── UHDLOL
│       ├── testing_set
│       └── train_set
└── README.md
```

## Training
In ```/PPatchDiff/options/```, we provide a set of ```yaml``` files to manage our training configs. 

Please double-check ```gt_root``` and ```input_root``` in each config and ensure they can successfully reach the data.

* Training on the LOL-v1 training set
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=22666 pydiff/train.py -opt options/train_lolv1_pgc.yaml --launcher pytorch
```

## Testing

* Evaluating on the LOL-v1 testing set
```
CUDA_VISIBLE_DEVICES=0 python pydiff/train.py -opt options/infer_lolv1_pgc.yaml
```

## Progressive strategy
You can use the ```progressive_list``` and ```stride_list``` parameters in the config files to adjust patch size and stride size at each step.

## Acknowledge
[PyDiff](https://github.com/limuloo/pydiff), [WeatherDiff](https://github.com/IGITUGraz/WeatherDiffusion)
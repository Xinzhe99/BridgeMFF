```
██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗███╗   ███╗███████╗███████╗
██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝████╗ ████║██╔════╝██╔════╝
██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗  ██╔████╔██║█████╗  █████╗  
██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝  ██║╚██╔╝██║██╔══╝  ██╔══╝  
██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗██║ ╚═╝ ██║██║     ██║     
╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝     
```

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official PyTorch implementation of ["BridgeMFF: Bridging the semantic and texture gap via dual adversarial learning for multi-focus image fusion"](paper_link)

</div>

## 📢 News

> [!NOTE]
> 📝 **2024.03**: Our paper is currently under review. Paper link will be available upon acceptance.

## 👥 Authors

**Xinzhe Xie** 👨‍🎓, **Buyu Guo**<sup>✉</sup> 👨‍🏫, **Peiliang Li** 👨‍🏫, **Shuangyan He** 👩‍🏫, **Sangjun Zhou** 👩‍🏫

### 🏛️ Institutions

- Ocean College, Zhejiang University, Zhoushan, P. R. China
- Hainan Institute, Zhejiang University, Sanya, P. R. China
- Donghai Laboratory, Zhoushan, P. R. China

<sup>✉</sup> Corresponding author: guobuyuwork@163.com

## 📑 Table of Contents

- [Overview](#-overview)
- [Highlights](#-highlights)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Citation](#-citation)

## 📖 Overview

In recent years, the two-stage multi-focus image fusion (MFF) method, which utilizes neural networks to first generate decision maps and then calculate the fused image, has witnessed significant advancements. However, after supervised training, many networks become overly reliant on semantic information, making it challenging to discern whether homogeneous regions and flat regions are in focus or not, as these regions lack distinct blur cues. To alleviate this issue, this paper proposes a multi-focus image fusion network named BridgeMFF by applying a visual state space model and developing a general fine-tuning technique named BridgeTune, which bridges the semantic and texture gap via dual adversarial learning. By fine-tuning the entire network in an adversarial manner, decision maps are generated to synthesize clear and blurred images with probability density distributions closely approximating real ones, thereby implicitly learning local spatial patterns and statistical properties of pixel values. Extensive experiments demonstrate that the proposed BridgeMFF achieves superior fusion quality, especially in challenging homogeneous regions. Moreover, BridgeMFF has the smallest model size (0.05M) and fastest processing speed (0.09s per image pair), enabling real-time fusion applications. The code will be attached here after the article is accepted.

<div align="center">
<img src="https://github.com/Xinzhe99/BridgeMFF/assets/113503163/17d21d4f-720a-4472-92ac-0ba9e90eb935" width="800px"/>
<p>Overview of BridgeMFF Framework</p>
</div>

## ✨ Highlights

- 🔄 A fine-tuning approach BridgeTune for multi-focus image fusion
- 🎯 A visual state space model-based fusion network BridgeMFF
- ⚡ SOTA performance with highest efficiency (0.05M parameters, 0.09s per image pair)
- 📊 Superior fusion quality especially in challenging homogeneous regions

## 🚀 Performance

<div align="center">
<img src="https://github.com/Xinzhe99/BridgeMFF/assets/113503163/5751cc4c-e3d7-47b5-b401-a0dd557e1372" width="800px"/>
<p>Quantitative comparison on Lytro dataset</p>
</div>

## Installation
### Environment Setup
We recommend using conda to manage the dependencies. Please refer to [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet) for detailed environment setup.

### Dataset Preparation
1. Download [DUTS](http://saliencydetection.net/duts/) dataset and organize it as follows:
```
DUTS/
├── DUTS-TE/
│   ├── DUTS-TE-Image/
│   └── DUTS-TE-Mask/
└── DUTS-TR/
    ├── DUTS-TR-Image/
    └── DUTS-TR-Mask/
```

2. Organize the test datasets as follows:
```
three_datasets_MFF/
├── Lytro/
│   ├── A/
│   └── B/
├── MFFW/
│   ├── A/
│   └── B/
└── MFI-WHU/
    ├── A/
    └── B/
```

3. Generate training data:
```bash
cd ./tools
# Generate training set
python make_datasets_DUTS.py --mode='TR' --data_root='/path/to/DUTS' --out_dir_name='DUTS_MFF'
# Generate validation set 
python make_datasets_DUTS.py --mode='TE' --data_root='/path/to/DUTS' --out_dir_name='DUTS_MFF'
cd ..
```

## Training
### Pre-training
1. Prepare visualization datasets:
```
three_datasets_MFF/
├── Lytro/
│   ├── A/
│   └── B/
├── MFFW/
│   ├── A/
│   └── B/
└── MFI-WHU/
    ├── A/
    └── B/
```

2. Start pre-training:
```bash
python train_1.py --dataset_path='./tools/DUTS_MFF' --Visualization_datasets='./three_datasets_MFF'
```

### Fine-tuning
**Note**: Ensure the last output layer is normalized to [0,1] before fine-tuning. GPU is required.

```bash
python train_2.py --dataset_path='./tools/DUTS_MFF' --pretrained_model='/path/to/pretrained.pth' --Visualization_datasets='./three_datasets_MFF'
```

## Inference
Test on different datasets:
```bash
# Test on Lytro
python predict.py --model_path='./generator.pth' --test_dataset_path='./three_datasets_MFF/Lytro'
# Test on MFFW
python predict.py --model_path='./generator.pth' --test_dataset_path='./three_datasets_MFF/MFFW'
# Test on MFI-WHU
python predict.py --model_path='./generator.pth' --test_dataset_path='./three_datasets_MFF/MFI-WHU'
```

## 📝 Citation

If you find this work useful for your research, please consider citing our papers:
```bibtex

@article{xie2024swinmff,
  title={SwinMFF: toward high-fidelity end-to-end multi-focus image fusion via swin transformer-based network},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and He, Shuangyan and Zhou, Sangjun},
  journal={The Visual Computer},
  pages={1--24},
  year={2024},
  publisher={Springer}
}

@inproceedings{xie2024underwater,
  title={Underwater Three-Dimensional Microscope for Marine Benthic Organism Monitoring},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and Jiang, Qingyan},
  booktitle={OCEANS 2024-Singapore},
  pages={1--4},
  year={2024},
  organization={IEEE}
}
```

## 🙏 Acknowledgements
- [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet)

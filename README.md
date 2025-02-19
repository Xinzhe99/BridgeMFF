# VSSM-MIF 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official PyTorch implementation of ["Multi-focus Image Fusion with Visual State Space Model and Dual Adversarial Learning"](paper_link)

## Authors
- Xinzhe Xie (xiexinzhe@zju.edu.cn)
- Buyu Guo (guobuyuwork@163.com)
- Peiliang Li (lipeiliang@zju.edu.cn) 
- Shuangyan He (hesy103@163.com)
- Sangjun Zhou (sjune163@163.com)

## Framework
![image](https://github.com/Xinzhe99/BridgeMFF/assets/113503163/17d21d4f-720a-4472-92ac-0ba9e90eb935)

## Performance
Quantitative comparison on Lytro dataset:
![image](https://github.com/Xinzhe99/BridgeMFF/assets/113503163/5751cc4c-e3d7-47b5-b401-a0dd557e1372)

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

2. Generate training data:
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

## Citation
If you find this work useful for your research, please consider citing our paper:
```bibtex
@article{xie2024vssm,
  title={Multi-focus Image Fusion with Visual State Space Model and Dual Adversarial Learning},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and He, Shuangyan and Zhou, Sangjun},
  journal={},
  year={2024}
}
```

## License
This project is released under the [MIT License](LICENSE).

## Acknowledgements
- [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet)

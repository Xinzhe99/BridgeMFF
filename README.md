# BridgeMFF
Code for “BridgeMFF: Bridging the semantic and texture gap via dual adversarial learning for multi-focus image fusion”，
## Creating our training dataset
1. Download [DUTS](https://www.openai.com) dataset and put it in a folder ./xxxx/DUTS
DUTS
├─DUTS-TE
│  ├─DUTS-TE-Image
│  └─DUTS-TE-Mask
└─DUTS-TR
    ├─DUTS-TR-Image
    └─DUTS-TR-Mask
```python
cd ./tools
python make_datasets_DUTS.py --mode='TR' --data_root=r'./xxxx/DUTS' --out_dir_name=DUTS_MFF #Training set
python make_datasets_DUTS.py --mode='TE' --data_root=r'./xxxx/DUTS' --out_dir_name=DUTS_MFF #Validation set
```

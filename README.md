# BridgeMFF
Code for “BridgeMFF: Bridging the semantic and texture gap via dual adversarial learning for multi-focus image fusion”
## Preparing the virtual environment using conda
Please refer [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet)
## Creating our training dataset
1. Download [DUTS]([https://www.openai.com](http://saliencydetection.net/duts/)) dataset and put it in a folder ./xxxx/DUTS
DUTS
├─DUTS-TE
│  ├─DUTS-TE-Image
│  └─DUTS-TE-Mask
└─DUTS-TR
    ├─DUTS-TR-Image
    └─DUTS-TR-Mask
2. Unzip this project and run:
```python
cd ./tools
python make_datasets_DUTS.py --mode='TR' --data_root='./xxxx/DUTS' --out_dir_name='DUTS_MFF' #Training set
python make_datasets_DUTS.py --mode='TE' --data_root='./xxxx/DUTS' --out_dir_name='DUTS_MFF' #Validation set
cd ..
```
## Fine-tuning a model
Before fine-tuning, please make sure the last output layer should be normalized to 0 to 1! Pleause use gpu device.
```python
python train_2.py --dataset_path='/tools/DUTS_MFF' --pretrained_model='./xxxx.pth'
```
## Predict using our model
├─Lytro
    ├─A
    └─B
├─MFFW
    ├─A
    └─B
├─MFI-WHU
    ├─A
    └─B
```python
python predict.py --model_path='./generator.pth' --test_dataset_path=r'/xxx/Lytro'#Lytro
python predict.py --model_path='./generator.pth' --test_dataset_path=r'/xxx/MFFW'#MFFW
python predict.py --model_path='./generator.pth' --test_dataset_path=r'/xxx/MFI-WHU'#MFI-WHU
```

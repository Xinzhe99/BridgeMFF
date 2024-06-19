# BridgeMFF
Code for “BridgeMFF: Bridging the semantic and texture gap via dual adversarial learning for multi-focus image fusion”
## Framework
![image](https://github.com/Xinzhe99/BridgeMFF/assets/113503163/17d21d4f-720a-4472-92ac-0ba9e90eb935)

## Performance on Lytro dataset
![image](https://github.com/Xinzhe99/BridgeMFF/assets/113503163/5751cc4c-e3d7-47b5-b401-a0dd557e1372)

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
## Pre-training a model
1. Prepare three datasets for visualization

three_datasets_MFF
├─Lytro

│  ├─A

│  └─B

├─MFFW

│  ├─A

│  └─B

├─MFI-WHU

│  ├─A

│  ├─B

```python
python train_1.py --dataset_path='/tools/DUTS_MFF' --Visualization_datasets='/three_datasets_MFF'
```

## Fine-tuning a model
1. Before fine-tuning, please make sure the last output layer should be normalized to 0 to 1! Pleause use gpu device.

2. run
   
```python
python train_2.py --dataset_path='/tools/DUTS_MFF' --pretrained_model='./xxxx.pth'#your model path
```
## Predict using our model

run

```python
python predict.py --model_path='./generator.pth' --test_dataset_path=r'/three_datasets_MFF/Lytro'#Lytro
python predict.py --model_path='./generator.pth' --test_dataset_path=r'/three_datasets_MFF/MFFW'#MFFW
python predict.py --model_path='./generator.pth' --test_dataset_path=r'/three_datasets_MFF/MFI-WHU'#MFI-WHU
```

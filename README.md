# BridgeMFF
Code for “BridgeMFF: Bridging the semantic and texture gap via dual adversarial learning for multi-focus image fusion”，
## Creating our training dataset
```python
cd tools
python make_datasets_DUTS.py --mode='TR'#Training set
python make_datasets_DUTS.py --mode='TE'#Validation set

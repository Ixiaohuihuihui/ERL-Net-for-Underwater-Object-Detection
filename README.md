# ERL-Net
## Note 
This project code is for Underwater Object Detection. 

## Install
It is based on [MMdetection](https://github.com/open-mmlab/mmdetection), please refer to [install.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) to install MMdetection.

## Dataset
We conduct experiments on the two challenging underwater datasets [UTDAC2020](https://www.flyai.com/d/underwaterdetection) and [Brackish dataset](https://www.kaggle.com/aalborguniversity/brackish-dataset). UTDAC2020 is the newest underwater dataset which is from Underwater Target Detection Algorithm Competition 2020. In addition, there are many wrong annotations in the original dataset, thus we manually corrected
the wrong data annotations on UTDAC2020. The refined UTDAC2020 dataset is open-sourced in https://drive.google.com/file/d/1avyB-ht3VxNERHpAwNTuBRFOxiXDMczI/view?usp=sharing.

The structure of this dataset is:
```
├── data
│   ├── UTDAC2020
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── annotations
```

## Train 
### Train on UTDAC2020 dataset
(1) Cascade R-CNN
```python
python tools/train.py configs/erl/erl_cascade_utdac.py
```
(2) Faster R-CNN
```python
python tools/train.py configs/erl/erl_faster_rcnn_utdac.py
```
(3) Retina
```python
python tools/train.py configs/erl/erl_retina_utdac
```

### Train on Brackish dataset
(1)
```python
python tools/train.py configs/erl/erl_cascade_brackish.py
```



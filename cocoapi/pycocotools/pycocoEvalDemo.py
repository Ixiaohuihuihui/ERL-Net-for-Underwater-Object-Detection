# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 15:45
# @Author  : Linhui Dai
# @FileName: pycocoEvalDemo.py
# @Software: PyCharm
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
annType = ['segm','bbox','keypoints']
annType = annType[1]
dataDir='/data3/dailh/train/coco_new/'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
cocoGt=COCO(annFile)
resFile = '/home/dailh/mmdetection/results/sabl_cas.bbox.json'
cocoDt = cocoGt.loadRes(resFile)
imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
if True:
  cocoEval.params.outDir = '/home/dailh/mmdetection/results/analyze_figures/sabl_cas/'
  cocoEval.analyze(save_to_dir=cocoEval.params.outDir)
  print('Done')
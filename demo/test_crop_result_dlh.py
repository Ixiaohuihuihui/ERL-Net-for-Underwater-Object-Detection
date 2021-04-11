#这个脚本是为了把crop之后网络的得到的结果转换到原图上，可以提交

import pandas as pd
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
import cv2

result = pd.read_csv("/home/dailh/crop_test.csv", index_col='image_id')

num = 0
with open("/data3/dailh/train/coco_new/crop_offset_8.txt", "r") as f:
    for line in f.readlines():
        pic_name = line.split(' ')[0]
        xml_name = pic_name.split('.')[0]+'.xml'
        detla_x = int(line.split(' ')[1])
        detla_y = int(line.split(' ')[2])

        r = result.loc[xml_name]
        r.xmin += detla_x
        r.xmax += detla_x

        r.ymin += detla_y
        r.ymax += detla_y

        print("这是第", num, "个, 这是类：", r.name)
        num += 1

    result.to_csv("/home/dailh/testA_submission.csv")




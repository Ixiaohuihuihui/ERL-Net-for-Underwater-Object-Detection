from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result_pyplot
import numpy as np
import cv2
import os
import torch
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config_file = '/data3/dailh/mmdetection/work_dirs/EGP_cascade/egp_cascade.py'
checkpoint_file = "/data3/dailh/mmdetection/work_dirs/EGP_cascade/epoch_15.pth"
#img_root = '/data3/dailh/train/coco_new/test-A-image'
img_root = '/data3/dailh/train/coco_new/val2017/'
def mmdet_csv_transform(result):
    dets = result
    flag = 0
    targets = None
    # dets are a list
    for det_obj_i in range(len(dets)):
        det_obj = dets[det_obj_i]
        if det_obj.shape[0] != 0:
            target = torch.zeros(det_obj.shape[0],7)
            target[:,:5] = torch.from_numpy(det_obj)
            target[:,6] = det_obj_i
            if flag == 0:
                targets = target
            else:
                targets = torch.cat((targets,target),0)
            flag = 1
    return targets


class detection_csv_generator(object):
    def __init__(self):
        self.total_result = []  # save all the results
        self.obj_name_list = ['echinus',  'starfish', 'holothurian','scallop']

    def save_csv(self, save_csv_path):
        # save_csv_path: the path you want to save your csv, like 'my_new_file.csv'
        df = pd.DataFrame(self.total_result, columns=['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
        df.to_csv(save_csv_path, index=None)
        return

    def add_result(self, file_path, detections):
        # detections: tensor  object detection output of one image,
        # The form of detections is (xmin,ymin,xmax,ymax,obj_conf,class_conf,obj_index), coordinates should be 1:1 ratio

        # file_path is the corresponding img path
        # return: pd (obj_name,filename,confidence,xmin,ymin,xmax,ymax)
        img_info = []
        for i in range(detections.size(0)):
            info = []
            if int(detections[i, -1].cpu()) > 3:
                continue
            obj_name = self.obj_name_list[int(detections[i, -1].cpu())]
            info.append(obj_name)

            file_name = file_path.split('/')[-1]
            file_name = file_name.split('.')[0] + '.xml'
            info.append(file_name)

            # class_conf
            conf = [float(detections[i, 4].cpu())]
            info = info + conf

            bbox = detections[i, :4].cpu().numpy().astype('int32').tolist()
            info = info + bbox

            img_info.append(info)
        self.total_result += img_info
        return
if __name__=='__main__':
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    total_results = detection_csv_generator()
    lists = os.listdir(img_root)
    for img_name in os.listdir(img_root):
        print('now detect',img_name)
        img_path = os.path.join(img_root,img_name)
        img = cv2.imread(img_path)
        # x, y = img.shape[0:2]
        # img = cv2.resize(img, (int(y / 1), int(x / 1))) #
        result = inference_detector(model, img)
        detections = mmdet_csv_transform(result[0])
        try:
            total_results.add_result(img_path,detections)
            #show_result(img, result, model.CLASSES)
        except:
            print('detection is None in',img_name)
    total_results.save_csv('../egp_cas_val.csv')

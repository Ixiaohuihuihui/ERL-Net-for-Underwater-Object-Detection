# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:05:22 2020

@author: mouse
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os
import cv2 as cv

obj_name_list = ['echinus', 'holothurian', 'starfish', 'scallop']
dicts = {'echinus': 0, 'holothurian': 1, 'starfish': 2, 'scallop': 3}

val_img_root = "/data3/dailh/train/coco_new/test-A-image/"
# val_img_root = "/home/dailh/data/coco_ori/train2017"

xml_root = "/data3/dailh/train/coco_new/box"
save_img_root = '/data3/dailh/train/coco_new/analysis_output_test/'

classes_num = np.zeros((1, 4))
cluster_num = np.zeros((1, 5))
corelation_matrix = np.zeros((4, 4))


red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
cyan = (255, 255, 0)
yellow = (0, 255, 255)
magenta = (255, 0, 255)
white = (255, 255, 255)
black = (0, 0, 0)

def xml2array(xml_path):
    targets = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for element in root.findall('object'):
        target = []

        cls = element.find('name').text
        if cls == 'waterweeds':
            continue
        target.append(int(dicts[cls]))

        bndbox = element.find('bndbox')
        xmin = bndbox.find('xmin').text
        target.append(int(xmin))

        ymin = bndbox.find('ymin').text
        target.append(int(ymin))

        xmax = bndbox.find('xmax').text
        target.append(int(xmax))

        ymax = bndbox.find('ymax').text
        target.append(int(ymax))

        targets.append(target)
    return np.array(targets)

def depart_data(df):
    names = df['name']
    label_list = []
    for name in names:
        label_list.append(dicts[name])
    boxes = df.loc[:, ("xmin", "ymin", "xmax", 'ymax')]
    boxes_list = boxes.values.tolist()
    scores = df.loc[:, ("confidence")]
    score_list = scores.tolist()
    return label_list, boxes_list, score_list


def combine_data(name, boxes, scores, labels):
    num_obj = boxes.shape[0]
    img_info = []
    for i in range(num_obj):
        info = []
        info.append(obj_name_list[int(labels[i])])
        info.append(name)
        info.append(scores[i])
        info += boxes[i].astype('int32').tolist()
        img_info.append(info)
    return img_info

def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof','overlap']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = True
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = False
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        if mode == 'overlap':
            return overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def detect_class_num(targets, name):
    flag = np.zeros((1, 4))
    for i in range(targets.shape[0]):
        flag[0, targets[i, 0]] = 1
        classes_num[0, targets[i, 0]] += 1
    cluster_num[0, int(np.sum(flag))] += 1
    for j in range(4):
        for k in range(4):
            if flag[0, j] == 1 and flag[0, k] == 1:
                corelation_matrix[j, k] += 1
    if int(np.sum(flag)) == 4:
        print(name)
    return

def generate_groundtruth():
    for img_name in os.listdir(val_img_root):
        print(img_name)
        name = img_name.split('.')[0]
        xml_path = os.path.join(xml_root,name+'.xml')
        targets = xml2array(xml_path)

        img_path = os.path.join(val_img_root,img_name)
        img = cv.imread(img_path)

        for i in range(targets.shape[0]):
            img = cv.rectangle(img,(int(targets[i,1]),int(targets[i,2])), (int(targets[i,3]),int(targets[i,4])), green,1)
            font = cv.FONT_HERSHEY_COMPLEX
            conf = str(1)
            text = obj_name_list[int(targets[i,0])] + ':' + conf
            cv.putText(img,text,(int(targets[i,1]),int(targets[i,2])-3),font,0.5,green,1)
        save_path = os.path.join(save_img_root,img_name)
        cv.imwrite(save_path,img)
    return

def generate_det_result():
    result = pd.read_csv("/home/dailh/test_gcnet_htc.csv")
    for img_name in os.listdir(save_img_root):
        print(img_name)
        name = img_name.split('.')[0]
        xml_name = name+'.xml'

        img_path = os.path.join(save_img_root,img_name)
        img = cv.imread(img_path)

        targets = result.loc[result["image_id"] == xml_name]
        names = targets['name'].values
        boxes = targets.loc[:, ("xmin", "ymin", "xmax", 'ymax')].values
        scores = targets.loc[:, ("confidence")].values
        for i in range(targets.shape[0]):
            if scores[i] >= 0.1:
                img = cv.rectangle(img,(int(boxes[i,0]),int(boxes[i,1])), (int(boxes[i,2]),int(boxes[i,3])), red,1)
                font = cv.FONT_HERSHEY_COMPLEX
                conf = str(scores[i])
                text = names[i] + ':' + conf
                cv.putText(img,text,(int(boxes[i,0]),int(boxes[i,1])-3),font,0.5,red,1)
        save_path = os.path.join(save_img_root,img_name)
        cv.imwrite(save_path, img)
    return

def fp_analysis(iou_thr=0.5,high_conf_thr = 0.5):
    result = pd.read_csv("/home/dailh/val2017_ors.csv")
    fp = 0
    tn = 0
    fp_score_sum = 0
    high_score_fp = 0
    dets_num = 0
    gts_num = 0
    small_tn = 0
    medium_tn = 0
    large_tn = 0

    tn_classes_num = np.zeros((4,1))
    fp_classes_num = np.zeros((4,1))

    for img_name in os.listdir(val_img_root):
        # print(img_name)
        name = img_name.split('.')[0]
        xml_name = name + '.xml'
        xml_path = os.path.join(xml_root,xml_name)
        gts = xml2array(xml_path)
        gts_name = gts[:,0]
        gts_bbox = gts[:,1:]

        dets = result.loc[result["image_id"] == xml_name]
        # dets.sort_values(by='confidence',ascending = False)
        names = dets['name'].values
        dets_boxes = dets.loc[:, ("xmin", "ymin", "xmax", 'ymax')].values
        dets_scores = dets.loc[:, ("confidence")].values
        inds = np.argsort(-dets_scores)
        dets_scores = dets_scores[inds]
        dets_boxes = dets_boxes[inds]
        names = names[inds]
        dets_num += dets_boxes.shape[0]
        gts_num += gts.shape[0]
        for i in range(dets.shape[0]):
            ious = bbox_overlaps(dets_boxes[i,:].reshape((1,-1)),gts_bbox)
            idx = np.argmax(ious)
            if ious[idx,0] < iou_thr:
                fp += 1
                # print('fp:', img_name)
                try:
                    a = names[i]
                    fp_classes_num[dicts[a]] += 1
                except:
                    print('x')

            if ious[idx,0] < iou_thr and i >= gts.shape[0]:
                high_conf_thr += 1
        for j in range(gts.shape[0]):
            ious = bbox_overlaps(gts_bbox[j,:].reshape((1,-1)),dets_boxes)
            idx = np.argmax(ious)
            if ious[idx,0] < iou_thr:
                # print('tn:',img_name)
                img_path = os.path.join(val_img_root, img_name)
                img = cv.imread(img_path)
                ratio = (800.0 / img.shape[0]) ** 2
                area = (gts_bbox[j,2]-gts_bbox[j,0])*(gts_bbox[j,3]-gts_bbox[j,1])*ratio
                if area < 32*32:
                    small_tn +=1
                    print('small_tn:', img_name)
                elif area >= 32*32 and area < 96*96:
                    medium_tn += 1
                else:
                    large_tn += 1
                tn += 1
                tn_classes_num[int(gts_name[j])] += 1
    print('dets_num:',dets_num)
    print('gts_num:',gts_num)
    print('fp:',fp)
    print('tn:',tn)
    print('small_tn:', small_tn)
    print('medium_tn:', medium_tn)
    print('large_tn:', large_tn)
    print('high_score_fp:', high_score_fp)
    print('fp_classes_num:',fp_classes_num)
    print('tn_classes_num:', tn_classes_num)
    return

def class_distribution_analysis():
    # training_area = 1333.0 * 800.0
    small = 0
    medium = 0
    large = 0
    for img_name in os.listdir(val_img_root):

        img_path = os.path.join(val_img_root, img_name)
        img = cv.imread(img_path)
        # ori_area = img.shape[0]*img.shape[1]
        ratio = (800.0/img.shape[0])**2
        # print(img_name)
        name = img_name.split('.')[0]
        xml_name = name + '.xml'
        xml_path = os.path.join(xml_root,xml_name)
        gts = xml2array(xml_path)

        gts_bbox = gts[:,1:]
        area = (gts_bbox[:,2] - gts_bbox[:,0])*(gts_bbox[:,3] - gts_bbox[:,1]).astype('float32')
        gts_name = gts[:, 0]
        area = area * ratio
        small += np.sum(area < 32*32)
        medium  += np.sum((32*32 <= area) * (area< 96 * 96))
        large += np.sum(area > 96*96)
        detect_class_num(gts, name)
    print(classes_num)
    print('small:',small)
    print('medium:',medium)
    print('large:',large)
    return

if __name__ == '__main__':
    # obj = []
    # obj.append(pd.read_csv('cascade_rcnn_dcn_testaug.csv'))

    # weights = np.ones((1,len(obj))).tolist()
    # targets = []
    #generate_groundtruth()
    generate_det_result()
    #class_distribution_analysis()
    #fp_analysis()




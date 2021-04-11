# -*- coding: utf-8 -*-
# @Time    : 2020/9/1 9:44
# @Author  : Linhui Dai
# @FileName: sinet_r50.py
# @Software: PyCharm
_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='SINet_ResNet'
    ),

)
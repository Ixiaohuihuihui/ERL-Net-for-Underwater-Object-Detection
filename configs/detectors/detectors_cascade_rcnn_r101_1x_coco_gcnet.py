# -*- coding: utf-8 -*-
# @Time    : 2020/8/31 17:41
# @Author  : Linhui Dai
# @FileName: detectors_cascade_rcnn_r101_1x_coco_gcnet.py
# @Software: PyCharm

_base_ = '../gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py'
model = dict(
    backbone=dict(
        type='DetectoRS_ResNeXt',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='torchvision://resnet50',
            style='pytorch')))

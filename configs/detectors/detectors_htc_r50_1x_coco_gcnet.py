# -*- coding: utf-8 -*-
# @Time    : 2020/9/1 15:09
# @Author  : Linhui Dai
# @FileName: detectors_htc_r50_1x_coco_gcnet.py
# @Software: PyCharm

_base_ = '../htc/htc_without_semantic_r50_fpn_1x_coco.py'
# _base_ = '../htc/htc_x101_32x4d_fpn_16x1_20e_coco.py'

model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True,
        plugins=[
            dict(
                cfg=dict(type='ContextBlock', ratio=0.0625),
                stages=(False, True, True, True),
                position='after_conv3')
        ],
        # dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
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
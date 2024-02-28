# Modified from https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend/point_head/point_head.py  # noqa

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init
from mmcv.ops import point_sample, rel_roi_point_to_rel_img_point
from mmdet.models.builder import HEADS, build_loss, build_roi_extractor
from mmcv.cnn import ConvModule, build_upsample_layer
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss


class SFMStage(nn.Module):

    def __init__(self,
                 semantic_in_channel=256,
                 semantic_out_channel=256,
                 fc_in_channels=256,
                 fc_channels=256,
                 fc_out_channels=256,
                 num_fcs=3,
                 num_classes=80,
                 semantic_out_stride=4,
                 mask_use_sigmoid=False,
                 coarse_pred_each_layer=True,
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(SFMStage, self).__init__()
        self.num_fcs = num_fcs
        self.semantic_out_stride = semantic_out_stride
        self.mask_use_sigmoid = mask_use_sigmoid
        self.num_classes = num_classes

        self.fcs = nn.ModuleList()
        for _ in range(num_fcs):
            fc = ConvModule(
                fc_in_channels,
                fc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.fcs.append(fc)

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.fc_logits = nn.Conv1d(
            fc_in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # for extracting instance-wise semantic feats
        self.semantic_transform_in = nn.Conv2d(semantic_in_channel, semantic_out_channel, 1)
        # self.semantic_transform_out = nn.Conv2d(semantic_out_channel, semantic_out_channel, 1)

        self.instance_logits = nn.Conv2d(fc_in_channels, num_classes, 1)
        self.detail_logits = nn.Conv2d(fc_in_channels, num_classes, 1)   

        # fuse_in_channel = instance_in_channel + semantic_out_channel + 2
        self.fuse_transform_out = nn.Conv2d(fc_in_channels, fc_out_channels, 1)
        self.upsample = build_upsample_layer(upsample_cfg.copy())
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.semantic_transform_in, self.detail_logits, self.instance_logits]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, semantic_pred, rois, roi_labels):
        # instance-wise semantic feats
        semantic_feat = self.relu(self.semantic_transform_in(semantic_feat))

        instance_preds = self.instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        detail_preds = self.detail_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        # instance_masks = instance_preds.sigmoid() if self.mask_use_sigmoid else instance_preds
        detail_masks = detail_preds.sigmoid() if self.mask_use_sigmoid else instance_preds

        point_indices, rel_roi_points = self.get_roi_rel_points_train(
                                            detail_masks, cfg=self.train_cfg)
        fine_grained_point_feats = self._get_fine_grained_point_feats(semantic_feat, rois,
                             rel_roi_points)
        coarse_feats = instance_feats.view(instance_feats.shape[0], -1)
        coarse_point_feats = torch.gather(coarse_feats, 2, point_indices)
        # coarse_point_feats = point_sample(coarse_feats, rel_roi_points)
        
        mask_point_feats = torch.cat([fine_grained_point_feats, coarse_point_feats], dim=1)
        for fc in self.fcs:
            mask_point_feats = fc(mask_point_feats)
            if self.coarse_pred_each_layer:
                mask_point_feats = torch.cat((mask_point_feats, coarse_point_feats), dim=1)
        refined_instance_feats = coarse_feats.scatter_(2, point_indices, mask_point_feats)
        refined_instance_feats = self.relu(self.sample(refined_instance_feats))
        refined_instance_feats = self.relu(self.upsample(refined_instance_feats))
        return instance_preds, detail_preds, refined_instance_feats


    def get_roi_rel_points_train(self, detail_pred, cfg):
        """Get ``num_points`` most uncertain points with random points during
        train.

        Sample points in [0, 1] x [0, 1] coordinate space based on their
        uncertainty. The uncertainties are calculated for each point using
        '_get_uncertainty()' function that takes point's logit prediction as
        input.

        Args:
            mask_pred (Tensor): A tensor of shape (num_rois, num_classes,
                mask_height, mask_width) for class-specific or class-agnostic
                prediction.
            labels (list): The ground truth class for each instance.
            cfg (dict): Training config of point head.

        Returns:
            point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
                that contains the coordinates sampled points.
        """
        num_points = cfg.num_points
        num_rois, _, mask_height, mask_width = detail_pred.shape
        h_step = 1.0 / mask_height
        w_step = 1.0 / mask_width

        point_detail_map = detail_pred.view(num_rois, mask_height * mask_width)
        num_points = min(mask_height*mask_width, num_points)
        point_indices = point_detail_map.topk(num_points, dim=1)[1]

        point_coords = point_detail_map.new_zeros(num_rois, num_points, 2)
        point_coords[:, :, 0] = w_step / 2.0 + (point_indices %
                                                mask_width).float() * w_step
        point_coords[:, :, 1] = h_step / 2.0 + (point_indices //
                                                mask_width).float() * h_step
        return point_indices, point_coords


    def _get_fine_grained_point_feats(self, feats, rois, rel_roi_points):
        """Sample fine grained feats from each level feature map and
        concatenate them together."""
        num_imgs = feats.shape[0]
        fine_grained_feats = []
        # for idx in range(self.mask_roi_extractor.num_inputs):
            # feats = x[idx]
            # spatial_scale = 1. / float(
            #     self.mask_roi_extractor.featmap_strides[idx])
        spatial_scale = 1.0 / float(self.semantic_out_stride)
        point_feats = []
        for batch_ind in range(num_imgs):
            # unravel batch dim
            feat = feats[batch_ind].unsqueeze(0)
            inds = (rois[:, 0].long() == batch_ind)
            if inds.any():
                rel_img_points = rel_roi_point_to_rel_img_point(
                    rois[inds], rel_roi_points[inds], feat.shape[2:],
                    spatial_scale).unsqueeze(0)
                point_feat = point_sample(feat, rel_img_points)
                point_feat = point_feat.squeeze(0).transpose(0, 1)
                point_feats.append(point_feat)
        # fine_grained_feats.append(torch.cat(point_feats, dim=0))
        return torch.cat(point_feats, dim=1)


@HEADS.register_module()
class RefineMaskHead(nn.Module):

    def __init__(self,
                 num_convs_instance=2,
                 num_convs_semantic=4,
                 num_fcs=3,
                 conv_in_channels_instance=256,
                 conv_in_channels_semantic=256,
                 conv_kernel_size_instance=3,
                 conv_kernel_size_semantic=3,
                 conv_out_channels_instance=256,
                 conv_out_channels_semantic=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 semantic_out_stride=4,
                 mask_use_sigmoid=False,
                 stage_num_classes=[80, 80, 80, 80],
                 stage_sup_size=[14, 28, 56, 112],
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 loss_cfg=dict(
                    type='RefineCrossEntropyLoss',
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    semantic_loss_weight=1.0,
                    boundary_width=2,
                    start_stage=1)):
        super(RefineMaskHead, self).__init__()

        self.num_convs_instance = num_convs_instance
        self.conv_kernel_size_instance = conv_kernel_size_instance
        self.conv_in_channels_instance = conv_in_channels_instance
        self.conv_out_channels_instance = conv_out_channels_instance

        self.num_convs_semantic = num_convs_semantic
        self.conv_kernel_size_semantic = conv_kernel_size_semantic
        self.conv_in_channels_semantic = conv_in_channels_semantic
        self.conv_out_channels_semantic = conv_out_channels_semantic

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.semantic_out_stride = semantic_out_stride
        self.stage_sup_size = stage_sup_size
        self.stage_num_classes = stage_num_classes

        self._build_conv_layer('instance')
        self._build_conv_layer('semantic')
        self.loss_func = build_loss(loss_cfg)

        assert len(self.stage_sup_size) > 1
        self.stages = nn.ModuleList()
        out_channel = conv_out_channels_instance
        for idx, out_size in enumerate(self.stage_sup_size[:-1]):
            in_channel = out_channel
            out_channel = in_channel // 2

            new_stage = SFMStage(
                semantic_in_channel=conv_out_channels_semantic,
                semantic_out_channel=in_channel,
                fc_in_channels=in_channel,
                fc_channels=in_channel,
                fc_out_channels=out_channel,
                num_fcs=num_fcs,
                num_classes=self.stage_num_classes[idx],
                semantic_out_stride=semantic_out_stride,
                mask_use_sigmoid=mask_use_sigmoid,
                upsample_cfg=upsample_cfg)

            self.stages.append(new_stage)

        self.final_instance_logits = nn.Conv2d(out_channel, self.stage_num_classes[-1], 1)
        self.final_detail_logits = nn.Conv2d(out_channel, self.stage_num_classes[-1], 1)
        self.semantic_logits = nn.Conv2d(conv_out_channels_semantic, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def _build_conv_layer(self, name):
        out_channels = getattr(self, f'conv_out_channels_{name}')
        conv_kernel_size = getattr(self, f'conv_kernel_size_{name}')

        convs = []
        for i in range(getattr(self, f'num_convs_{name}')):
            in_channels = getattr(self, f'conv_in_channels_{name}') if i == 0 else out_channels
            conv = ConvModule(in_channels, out_channels, conv_kernel_size, dilation=1, padding=1)
            convs.append(conv)

        self.add_module(f'{name}_convs', nn.ModuleList(convs))

    def init_weights(self):
        for m in [self.final_instance_logits, self.final_detail_logits, self.semantic_logits]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, rois, roi_labels):
        for conv in self.instance_convs:
            instance_feats = conv(instance_feats)

        for conv in self.semantic_convs:
            semantic_feat = conv(semantic_feat)

        semantic_pred = self.semantic_logits(semantic_feat)

        stage_instance_preds = []
        stage_detail_preds = []
        for stage in self.stages:
            instance_preds, detail_preds, instance_feats = stage(instance_feats, semantic_feat, semantic_pred, rois, roi_labels)
            stage_instance_preds.append(instance_preds)
            stage_detail_preds.append(detail_preds)
        # for LVIS, use class-agnostic classifier for the last stage
        if self.stage_num_classes[-1] == 1:
            roi_labels = roi_labels.clamp(max=0)

        instance_preds = self.final_instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        stage_instance_preds.append(instance_preds)
        detail_preds = self.final_detail_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        stage_detail_preds.append(detail_preds)
        return stage_instance_preds, stage_detail_preds, semantic_pred

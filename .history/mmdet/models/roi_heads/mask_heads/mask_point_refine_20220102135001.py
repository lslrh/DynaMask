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
                 instance_in_channel=256,
                 instance_out_channel=256,
                 fc_in_channels=256,
                 fc_channels=256,
                 num_fcs=3,
                 out_size=14,
                 num_classes=80,
                 semantic_out_stride=4,
                 mask_use_sigmoid=False,
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
                act_cfg=act_cfg )

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.fc_logits = nn.Conv1d(
            fc_in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # for extracting instance-wise semantic feats
        self.semantic_transform_in = nn.Conv2d(semantic_in_channel, semantic_out_channel, 1)
        # self.semantic_transform_out = nn.Conv2d(semantic_out_channel, semantic_out_channel, 1)

        self.instance_logits = nn.Conv2d(instance_in_channel, num_classes, 1)
        self.detail_logits = nn.Conv2d(instance_in_channel, num_classes, 1)   

        fuse_in_channel = instance_in_channel + semantic_out_channel + 2

        self.upsample = build_upsample_layer(upsample_cfg.copy())
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.semantic_transform_in, self.detail_logits, self.instance_logits, self.fuse_transform_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, semantic_pred, rois, roi_labels):
        concat_tensors = [instance_feats]

        # instance-wise semantic feats
        semantic_feat = self.relu(self.semantic_transform_in(semantic_feat))

        instance_preds = self.instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        detail_preds = self.detail_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        instance_masks = instance_preds.sigmoid() if self.mask_use_sigmoid else instance_preds
        detail_masks = detail_preds.sigmoid() if self.mask_use_sigmoid else instance_preds
        
        point_indices, rel_roi_points = self.get_roi_rel_points_train(
                                            detail_masks, cfg=self.train_cfg)
        fine_grained_point_feats = self._get_fine_grained_point_feats(semantic_feat, rois,
                             rel_roi_points)
        coarse_feats = instance_feats.view(instance_feats.shape[0], -1)
        coarse_point_feats = torch.gather(coarse_feats, 2, point_indices)
        # coarse_point_feats = point_sample(coarse_feats, rel_roi_points)
        coarse_feats[:,:,point_indices[2]]
        
        mask_point_feats = torch.cat([fine_grained_point_feats, coarse_point_feats], dim=1)
        for fc in self.fcs:
            mask_point_feats = fc(mask_point_feats)
            if self.coarse_pred_each_layer:
                mask_point_feats = torch.cat((mask_point_feats, coarse_point_feats), dim=1)
        refined_instance_feats = coarse_feats.scatter_(2, point_indices, mask_point_feats)

        

        concat_tensors.append(ins_semantic_feats)

        # instance masks
        instance_preds = self.instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        detail_preds = self.detail_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        _instance_preds = instance_preds.sigmoid() if self.mask_use_sigmoid else instance_preds
        instance_masks = F.interpolate(_instance_preds, instance_feats.shape[-2], mode='bilinear', align_corners=True)
        concat_tensors.append(instance_masks)

        # instance masks
        instance_preds = self.instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        detail_preds = self.detail_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        _instance_preds = instance_preds.sigmoid() if self.mask_use_sigmoid else instance_preds
        instance_masks = F.interpolate(_instance_preds, instance_feats.shape[-2], mode='bilinear', align_corners=True)
        _detail_preds = detail_preds.sigmoid() if self.mask_use_sigmoid else instance_preds
        detail_masks = F.interpolate(_detail_preds, instance_feats.shape[-2], mode='bilinear', align_corners=True)
        concat_tensors.append(instance_masks)
        concat_tensors.append(detail_masks)

        # instance-wise semantic masks
        _semantic_pred = semantic_pred.sigmoid() if self.mask_use_sigmoid else semantic_pred
        ins_semantic_masks = roi_align(
            _semantic_pred, rois, instance_feats.shape[-2:], 1.0 / self.semantic_out_stride, 0, 'avg', True)
        ins_semantic_masks = F.interpolate(
            ins_semantic_masks, instance_feats.shape[-2:], mode='bilinear', align_corners=True)
        concat_tensors.append(ins_semantic_masks)

        # fuse instance feats & instance masks & semantic feats & semantic masks
        fused_feats = torch.cat(concat_tensors, dim=1)

        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))
        
        fused_feats = self.relu(self.fuse_transform_out(fused_feats))

        fused_feats = self.relu(self.upsample(fused_feats))

        # concat instance and semantic masks with fused feats again
        instance_masks = F.interpolate(_instance_preds, fused_feats.shape[-2], mode='bilinear', align_corners=True)
        detail_masks = F.interpolate(_detail_preds, fused_feats.shape[-2], mode='bilinear', align_corners=True)
        ins_semantic_masks = F.interpolate(ins_semantic_masks, fused_feats.shape[-2], mode='bilinear', align_corners=True)
        fused_feats = torch.cat([fused_feats, instance_masks, detail_masks, ins_semantic_masks], dim=1)

        return instance_preds, detail_preds, fused_feats


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
                 conv_in_channels_instance=256,
                 conv_in_channels_semantic=256,
                 conv_kernel_size_instance=3,
                 conv_kernel_size_semantic=3,
                 conv_out_channels_instance=256,
                 conv_out_channels_semantic=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 dilations=[1, 3, 5],
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
                instance_in_channel=in_channel,
                instance_out_channel=out_channel,
                dilations=dilations,
                out_size=out_size,
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


@HEADS.register_module()
class MaskPointHead(nn.Module):
    """A mask point head use in PointRend.

    ``MaskPointHead`` use shared multi-layer perceptron (equivalent to
    nn.Conv1d) to predict the logit of input points. The fine-grained feature
    and coarse feature will be concatenate together for predication.

    Args:
        num_fcs (int): Number of fc layers in the head. Default: 3.
        in_channels (int): Number of input channels. Default: 256.
        fc_channels (int): Number of fc channels. Default: 256.
        num_classes (int): Number of classes for logits. Default: 80.
        class_agnostic (bool): Whether use class agnostic classification.
            If so, the output channels of logits will be 1. Default: False.
        coarse_pred_each_layer (bool): Whether concatenate coarse feature with
            the output of each fc layer. Default: True.
        conv_cfg (dict | None): Dictionary to construct and config conv layer.
            Default: dict(type='Conv1d'))
        norm_cfg (dict | None): Dictionary to construct and config norm layer.
            Default: None.
        loss_point (dict): Dictionary to construct and config loss layer of
            point head. Default: dict(type='CrossEntropyLoss', use_mask=True,
            loss_weight=1.0).
    """

    def __init__(self,
                 num_classes,
                 num_fcs=3,
                 in_channels=256,
                 fc_channels=256,
                 class_agnostic=False,
                 coarse_pred_each_layer=True,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 loss_point=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super().__init__()
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channles = fc_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.coarse_pred_each_layer = coarse_pred_each_layer
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_point = build_loss(loss_point)

        fc_in_channels = in_channels + num_classes
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
            fc_in_channels = fc_channels
            fc_in_channels += num_classes if self.coarse_pred_each_layer else 0

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.fc_logits = nn.Conv1d(
            fc_in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def init_weights(self):
        """Initialize last classification layer of MaskPointHead, conv layers
        are already initialized by ConvModule."""
        normal_init(self.fc_logits, std=0.001)

    def forward(self, fine_grained_feats, coarse_feats):
        """Classify each point base on fine grained and coarse feats.

        Args:
            fine_grained_feats (Tensor): Fine grained feature sampled from FPN,
                shape (num_rois, in_channels, num_points).
            coarse_feats (Tensor): Coarse feature sampled from CoarseMaskHead,
                shape (num_rois, num_classes, num_points).

        Returns:
            Tensor: Point classification results,
                shape (num_rois, num_class, num_points).
        """

        x = torch.cat([fine_grained_feats, coarse_feats], dim=1)
        for fc in self.fcs:
            x = fc(x)
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_feats), dim=1)
        return self.fc_logits(x)

    def get_targets(self, rois, rel_roi_points, sampling_results, gt_masks,
                    cfg):
        """Get training targets of MaskPointHead for all images.

        Args:
            rois (Tensor): Region of Interest, shape (num_rois, 5).
            rel_roi_points: Points coordinates relative to RoI, shape
                (num_rois, num_points, 2).
            sampling_results (:obj:`SamplingResult`): Sampling result after
                sampling and assignment.
            gt_masks (Tensor) : Ground truth segmentation masks of
                corresponding boxes, shape (num_rois, height, width).
            cfg (dict): Training cfg.

        Returns:
            Tensor: Point target, shape (num_rois, num_points).
        """

        num_imgs = len(sampling_results)
        rois_list = []
        rel_roi_points_list = []
        for batch_ind in range(num_imgs):
            inds = (rois[:, 0] == batch_ind)
            rois_list.append(rois[inds])
            rel_roi_points_list.append(rel_roi_points[inds])
        pos_assigned_gt_inds_list = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        cfg_list = [cfg for _ in range(num_imgs)]

        point_targets = map(self._get_target_single, rois_list,
                            rel_roi_points_list, pos_assigned_gt_inds_list,
                            gt_masks, cfg_list)
        point_targets = list(point_targets)

        if len(point_targets) > 0:
            point_targets = torch.cat(point_targets)

        return point_targets

    def _get_target_single(self, rois, rel_roi_points, pos_assigned_gt_inds,
                           gt_masks, cfg):
        """Get training target of MaskPointHead for each image."""
        num_pos = rois.size(0)
        num_points = cfg.num_points
        if num_pos > 0:
            gt_masks_th = (
                gt_masks.to_tensor(rois.dtype, rois.device).index_select(
                    0, pos_assigned_gt_inds))
            gt_masks_th = gt_masks_th.unsqueeze(1)
            rel_img_points = rel_roi_point_to_rel_img_point(
                rois, rel_roi_points, gt_masks_th.shape[2:])
            point_targets = point_sample(gt_masks_th,
                                         rel_img_points).squeeze(1)
        else:
            point_targets = rois.new_zeros((0, num_points))
        return point_targets

    def loss(self, point_pred, point_targets, labels):
        """Calculate loss for MaskPointHead.

        Args:
            point_pred (Tensor): Point predication result, shape
                (num_rois, num_classes, num_points).
            point_targets (Tensor): Point targets, shape (num_roi, num_points).
            labels (Tensor): Class label of corresponding boxes,
                shape (num_rois, )

        Returns:
            dict[str, Tensor]: a dictionary of point loss components
        """

        loss = dict()
        if self.class_agnostic:
            loss_point = self.loss_point(point_pred, point_targets,
                                         torch.zeros_like(labels))
        else:
            loss_point = self.loss_point(point_pred, point_targets, labels)
        loss['loss_point'] = loss_point
        return loss

    def _get_uncertainty(self, mask_pred, labels):
        """Estimate uncertainty based on pred logits.

        We estimate uncertainty as L1 distance between 0.0 and the logits
        prediction in 'mask_pred' for the foreground class in `classes`.

        Args:
            mask_pred (Tensor): mask predication logits, shape (num_rois,
                num_classes, mask_height, mask_width).

            labels (list[Tensor]): Either predicted or ground truth label for
                each predicted mask, of length num_rois.

        Returns:
            scores (Tensor): Uncertainty scores with the most uncertain
                locations having the highest uncertainty score,
                shape (num_rois, 1, mask_height, mask_width)
        """
        if mask_pred.shape[1] == 1:
            gt_class_logits = mask_pred.clone()
        else:
            inds = torch.arange(mask_pred.shape[0], device=mask_pred.device)
            gt_class_logits = mask_pred[inds, labels].unsqueeze(1)
        return -torch.abs(gt_class_logits)

    def get_roi_rel_points_train(self, detail_pred, labels, cfg):
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

    def get_roi_rel_points_test(self, mask_pred, pred_label, cfg):
        """Get ``num_points`` most uncertain points during test.

        Args:
            mask_pred (Tensor): A tensor of shape (num_rois, num_classes,
                mask_height, mask_width) for class-specific or class-agnostic
                prediction.
            pred_label (list): The predication class for each instance.
            cfg (dict): Testing config of point head.

        Returns:
            point_indices (Tensor): A tensor of shape (num_rois, num_points)
                that contains indices from [0, mask_height x mask_width) of the
                most uncertain points.
            point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
                that contains [0, 1] x [0, 1] normalized coordinates of the
                most uncertain points from the [mask_height, mask_width] grid .
        """
        num_points = cfg.subdivision_num_points
        uncertainty_map = self._get_uncertainty(mask_pred, pred_label)
        batch_size, num_rois, mask_height, mask_width = uncertainty_map.shape
        h_step = 1.0 / mask_height
        w_step = 1.0 / mask_width

        uncertainty_map = uncertainty_map.view(num_rois,
                                               mask_height * mask_width)
        num_points = min(mask_height * mask_width, num_points)
        point_indices = uncertainty_map.topk(num_points, dim=1)[1]

        point_coords = uncertainty_map.new_zeros(num_rois, num_points, 2)
        point_coords[:, :, 0] = w_step / 2.0 + (point_indices %
                                                mask_width).float() * w_step
        point_coords[:, :, 1] = h_step / 2.0 + (point_indices //
                                                mask_width).float() * h_step
        return point_indices, point_coords

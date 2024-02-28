import torch
import torch.nn.functional as F

from mmdet.core import bbox2roi
from mmdet.models.losses.cross_entropy_loss import generate_block_target
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.builder import HEADS
import numpy as np
import pdb

@HEADS.register_module()
class RefineRoIHead(StandardRoIHead):
    def init_weights(self, pretrained):
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None):
        # assign gts and sample proposals
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas)

        mask_results = self._mask_forward_train(
            x, sampling_results, bbox_results['bbox_feats'], gt_bboxes, gt_masks, gt_labels, img_metas)

        losses = {}
        losses.update(bbox_results['loss_bbox'])
        losses.update(mask_results['loss_mask'])
        losses.update(mask_results['loss_flops'])
        return losses

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_bboxes, gt_masks, gt_labels, img_metas):
        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_rois = bbox2roi(pos_bboxes)

        stage_mask_targets = \
            self.mask_head.get_targets(pos_bboxes, pos_assigned_gt_inds, gt_masks)

        mask_results = self._mask_forward(x[0], pos_rois, torch.cat(pos_labels))

        ins_semantic_feats = self.semantic_roi_extractor([x[0].detach(),], pos_rois)
        mask_labels = self.get_mask_label(ins_semantic_feats)

        # resize the semantic target
        # semantic_target = F.interpolate(
        #     semantic_target.unsqueeze(1),
        #     mask_results['semantic_pred'].shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        # semantic_target = (semantic_target >= 0.5).float()

        loss_mask = self.mask_head.loss(
            mask_results['stage_instance_preds'],
            mask_results['stage_detail_preds'],
            mask_labels,
            stage_mask_targets)
        loss_flops = {}
        loss_flops['loss_flops'] = self.train_cfg.flops_loss_weight * torch.clamp\
            ((torch.sum(mask_labels*torch.Tensor(self.train_cfg.flops).cuda())/len(mask_labels)-self.train_cfg.target_flops)/(self.train_cfg.flops[-1]-self.train_cfg.flops[0]), min=0)
        mask_results.update(loss_mask=loss_mask, loss_flops=loss_flops)
        # mask_results.update(loss_mask=loss_mask, loss_semantic=loss_semantic)
        return mask_results

    def _mask_forward(self, x, rois, roi_labels):
        """Mask head forward function used in both training and testing."""
        pdb.set_trace()
        ins_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        stage_instance_preds, stage_detail_preds = self.mask_head(ins_feats, x[0], rois, roi_labels)
        return dict(stage_instance_preds=stage_instance_preds, \
            stage_detail_preds=stage_detail_preds)


    def get_mask_label(self, ins_semantic_feats):
        mask_logits = self.mask_predictor(ins_semantic_feats)
        mask_labels = self.gumbel_softmax(mask_logits, temperature=0.5, hard=True)
        return mask_labels

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature=1):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature=1, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if not hard:
            return y
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard


    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for mask head without augmentation."""

        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            mask_rois = bbox2roi([_bboxes])

            interval = 100  # to avoid memory overflow
            segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
            for i in range(0, det_labels.shape[0], interval):
                mask_results = self._mask_forward(x, mask_rois[i: i + interval], det_labels[i: i + interval])

                # refine instance masks from stage 1
                stage_instance_preds = mask_results['stage_instance_preds'][1:]
                for idx in range(len(stage_instance_preds) - 1):
                    instance_pred = stage_instance_preds[idx].squeeze(1).sigmoid() >= 0.5
                    non_boundary_mask = (generate_block_target(instance_pred, boundary_width=1) != 1).unsqueeze(1)
                    non_boundary_mask = F.interpolate(
                        non_boundary_mask.float(),
                        stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True) >= 0.5
                    pre_pred = F.interpolate(
                        stage_instance_preds[idx],
                        stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
                    stage_instance_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
                instance_pred = stage_instance_preds[-1]

                chunk_segm_result = self.mask_head.get_seg_masks(
                    instance_pred, _bboxes[i: i + interval], det_labels[i: i + interval],
                    self.test_cfg, ori_shape, scale_factor, rescale)

                for c, segm in zip(det_labels[i: i + interval], chunk_segm_result):
                    segm_result[c].append(segm)

        return segm_result

    # def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
    #     """Simple test for mask head without augmentation."""
    #     ori_shape = img_metas[0]['ori_shape']
    #     scale_factor = img_metas[0]['scale_factor']
    #     class_distribution = torch.zeros(4).cuda()
    #     if det_bboxes.shape[0] == 0:
    #         segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
    #     else:
    #         # if det_bboxes is rescaled to the original image size, we need to
    #         # rescale it back to the testing scale to obtain RoIs.
    #         if rescale and not isinstance(scale_factor, float):
    #             scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
    #         _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
    #         mask_rois = bbox2roi([_bboxes])
    #         interval = 100  # to avoid memory overflow
    #         segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
    #         for i in range(0, det_labels.shape[0], interval):
    #             mask_results = self._mask_forward(x, mask_rois[i: i + interval], det_labels[i: i + interval])
    #             # refine instance masks from stage 1
    #             stage_instance_preds = mask_results['stage_instance_preds'][0:]
    #             # for idx in range(len(stage_instance_preds) - 1):
    #             #     instance_pred = stage_instance_preds[idx].squeeze(1).sigmoid() >= 0.5
    #             #     non_boundary_mask = (generate_block_target(instance_pred, boundary_width=1) != 1).unsqueeze(1)
    #             #     non_boundary_mask = F.interpolate(
    #             #         non_boundary_mask.float(),
    #             #         stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True) >= 0.5
    #             #     pre_pred = F.interpolate(
    #             #         stage_instance_preds[idx],
    #             #         stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
    #             #     stage_instance_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
    #             chunk_segm_result = []
    #             for idx in range(len(stage_instance_preds)):
    #                 instance_pred = stage_instance_preds[idx]
    #                 chunk_segm_result_tmp = self.mask_head.get_seg_masks(
    #                     instance_pred, _bboxes[i: i + interval], det_labels[i: i + interval],
    #                     self.test_cfg, ori_shape, scale_factor, rescale)
    #                 chunk_segm_result.append(chunk_segm_result_tmp)
    #             # mask_labels
    #             ins_semantic_feats = self.semantic_roi_extractor([x[0].detach(),], mask_rois[i: i + interval])
    #             mask_labels = self.get_mask_label(ins_semantic_feats) 
    #             class_distribution = torch.sum(mask_labels, dim=0) #/ torch.sum(mask_labels)
    #             # print(class_distribution)
    #             mask_labels = torch.argmax(mask_labels, dim=1).cpu().numpy()
    #             for j in range(det_labels.shape[0]):
    #                 segm_result[det_labels[j]].append(chunk_segm_result[mask_labels[j]][j])
    #     return segm_result, class_distribution
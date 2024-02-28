import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from PIL import Image
import pdb

def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    """Calculate the CrossEntropy loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    # Caution: this function should only be used in RPN
    # in other files such as in ghm_loss, the _expand_binary_labels
    # is used for multi-class classification.
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None):
    """Calculate the CrossEntropy loss for masks.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
    Returns:
        torch.Tensor: The calculated loss
    """
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    # num_rois = pred.size()[0]
    # inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    # pred_slice = pred[inds, label].squeeze(1)
    x = torch.sigmoid(pred)
    eps = 1e-10
    result = -torch.mean((target*torch.log(x+eps)+(1-target)*torch.log(1-x+eps))*class_weight)
    return result

def mask_cross_entropy_(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss

    Example:
        >>> N, C = 3, 11
        >>> H, W = 2, 2
        >>> pred = torch.randn(N, C, H, W) * 1000
        >>> target = torch.rand(N, H, W)
        >>> label = torch.randint(0, C, size=(N,))
        >>> reduction = 'mean'
        >>> avg_factor = None
        >>> class_weights = None
        >>> loss = mask_cross_entropy(pred, target, label, reduction,
        >>>                           avg_factor, class_weights)
        >>> assert loss.shape == (1,)
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)

    pred_slice = pred[inds, label].squeeze(1)

    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


def generate_block_target(mask_target, boundary_width=3):
    mask_target = mask_target.float()

    # boundary region
    kernel_size = 2 * boundary_width + 1
    laplacian_kernel = - torch.ones(1, 1, kernel_size, kernel_size).to(
        dtype=torch.float32, device=mask_target.device).requires_grad_(False)
    laplacian_kernel[0, 0, boundary_width, boundary_width] = kernel_size ** 2 - 1

    pad_target = F.pad(mask_target.unsqueeze(1), (boundary_width, boundary_width, boundary_width, boundary_width), "constant", 0)

    # pos_boundary
    pos_boundary_targets = F.conv2d(pad_target, laplacian_kernel, padding=0)
    pos_boundary_targets = pos_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    pos_boundary_targets[pos_boundary_targets > 0.1] = 1
    pos_boundary_targets[pos_boundary_targets <= 0.1] = 0
    pos_boundary_targets = pos_boundary_targets.squeeze(1)

    # neg_boundary
    neg_boundary_targets = F.conv2d(1 - pad_target, laplacian_kernel, padding=0)
    neg_boundary_targets = neg_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    neg_boundary_targets[neg_boundary_targets > 0.1] = 1
    neg_boundary_targets[neg_boundary_targets <= 0.1] = 0
    neg_boundary_targets = neg_boundary_targets.squeeze(1)

    # generate block target
    block_target = torch.zeros_like(mask_target).long().requires_grad_(False)
    boundary_inds = (pos_boundary_targets + neg_boundary_targets) > 0
    foreground_inds = (mask_target - pos_boundary_targets) > 0
    block_target[boundary_inds] = 1
    block_target[foreground_inds] = 2
    return block_target

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    # loss = loss*mask_labels
    return loss.mean()

class DetailTarget(nn.Module):
    def __init__(self):
        super(DetailTarget, self).__init__()
        
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
        
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[7./10], [3./10]],
            dtype=torch.float32).reshape(1, 2, 1, 1).type(torch.cuda.FloatTensor))
    def forward(self, gtmasks):
        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0
        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        
        # boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
        # boundary_targets_x4 = boundary_targets_x4.clamp(min=0)
        # boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
        # boundary_targets_x8 = boundary_targets_x8.clamp(min=0)
    
        # boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        # boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
        
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        
        
        # boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        # boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
       
        
        # boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        # boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
        
        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up), dim=1)
        
        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)
        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0
        
        
        # if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
        #     boundary_logits = F.interpolate(
        #         boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
        
        # bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        # dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        # return bce_loss,  dice_loss
        
        return boudary_targets_pyramid



@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.
        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy_
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


@LOSSES.register_module()
class RefineCrossEntropyLoss(nn.Module):

    def __init__(self,
                 stage_instance_loss_weight=[1.0, 1.0, 1.0, 1.0],
                 stage_detail_loss_weight=[1.0, 1.0, 1.0, 1.0],
                 detail_loss_weight=1.0,
                 cb_loss_weight=1.0,
                 boundary_width=2,
                 start_stage=1):
        super(RefineCrossEntropyLoss, self).__init__()

        self.stage_instance_loss_weight = stage_instance_loss_weight
        self.stage_detail_loss_weight = stage_detail_loss_weight
        self.detail_loss_weight = detail_loss_weight
        self.cb_loss_weight = cb_loss_weight
        self.boundary_width = boundary_width
        self.start_stage = start_stage
        self.detail_target = DetailTarget()

    def forward(self, stage_instance_preds, stage_detail_preds, stage_instance_targets):
        loss_mask_set = []
        loss_detail_set = []
        for idx in range(len(stage_instance_preds)):
            instance_pred, instance_target = stage_instance_preds[idx].squeeze(1), stage_instance_targets[idx]
            detail_pred, detail_target = stage_detail_preds[idx].squeeze(1), self.detail_target(instance_target).squeeze(1)          
            # A = torch.sigmoid(detail_pred).detach().cpu().numpy()
            # out = Image.fromarray(A[0]*255)
            # out = out.convert('L')
            # out.save(str(idx)+'.jpg')
            
            # B = detail_target.detach().cpu().numpy()
            # out = Image.fromarray(B[0]*255)
            # out = out.convert('L')
            # out.save('4.jpg')

            if idx <= self.start_stage:
                loss_mask = binary_cross_entropy(instance_pred, instance_target)

                loss_detail_ = dice_loss_func(torch.sigmoid(detail_pred), detail_target)
                # loss_detail = mask_cross_entropy(detail_pred, detail_target, class_weight=mask_labels[:,idx].view(-1,1,1))\
                #                 *len(instance_pred)/(torch.sum(mask_labels[:,idx].detach()).item()+1e-5)

                loss_mask_set.append(loss_mask + self.detail_loss_weight * loss_detail_)
                # loss_detail_set.append(loss_detail)
        
        # A = torch.sigmoid(instance_pred).detach().cpu().numpy()
        # out = Image.fromarray(A[0]*255)
        # out = out.convert('L')
        # out.save('instance.jpg')
        
        # B = torch.sigmoid(detail_pred).detach().cpu().numpy()
        # out = Image.fromarray(B[0]*255)
        # out = out.convert('L')
        # out.save('detail.jpg')        
        # pdb.set_trace()

        # class_distribution = torch.sum(mask_labels, dim=0) / torch.sum(mask_labels)
        # print(class_distribution)
        # print(loss_mask_set)
        # loss_cb = torch.sum(class_distribution*torch.log(class_distribution+1e-10))
        assert len(self.stage_instance_loss_weight) == len(loss_mask_set)
        loss_mask = sum([weight * loss for weight, loss in zip(self.stage_instance_loss_weight, loss_mask_set)]) 
        # loss_detail = sum([weight * loss for weight, loss in zip(self.stage_detail_loss_weight, loss_detail_set)]) + self.cb_loss_weight * loss_cb
        loss_instance = loss_mask #+ loss_detail
        return loss_instance

# @LOSSES.register_module()
# class PointRefineCrossEntropyLoss(nn.Module):

#     def __init__(self,
#                  stage_instance_loss_weight=[1.0, 1.0, 1.0, 1.0],
#                  stage_detail_loss_weight=[1.0, 1.0, 1.0, 1.0],
#                  semantic_loss_weight=1.0,
#                  detail_loss_weight=1.0,
#                  boundary_width=2,
#                  start_stage=1):
#         super(PointRefineCrossEntropyLoss, self).__init__()

#         self.stage_instance_loss_weight = stage_instance_loss_weight
#         self.semantic_loss_weight = semantic_loss_weight
#         self.detail_loss_weight = detail_loss_weight
#         self.boundary_width = boundary_width
#         self.start_stage = start_stage
#         self.detail_target = DetailTarget()

#     def forward(self, stage_instance_preds, stage_detail_preds, stage_instance_targets):
#         loss_mask_set = []
#         for idx in range(len(stage_instance_preds)):
#             instance_pred, instance_target = stage_instance_preds[idx].squeeze(1), stage_instance_targets[idx]
#             detail_pred, detail_target = stage_detail_preds[idx].squeeze(1), self.detail_target(instance_target).squeeze(1)          

#             # A = torch.sigmoid(detail_pred).detach().cpu().numpy()
#             # out = Image.fromarray(A[0]*255)
#             # out = out.convert('L')
#             # out.save(str(idx)+'.jpg')
            
#             # B = detail_target.detach().cpu().numpy()
#             # out = Image.fromarray(B[0]*255)
#             # out = out.convert('L')
#             # out.save('4.jpg')

#             if idx <= self.start_stage:
#                 loss_mask = binary_cross_entropy(instance_pred, instance_target)
#                 loss_detail = dice_loss_func(torch.sigmoid(detail_pred), detail_target)
#                 loss_mask_set.append(loss_mask + self.detail_loss_weight * loss_detail)
        
#         # A = torch.sigmoid(instance_pred).detach().cpu().numpy()
#         # out = Image.fromarray(A[0]*255)
#         # out = out.convert('L')
#         # out.save('instance.jpg')
        
#         # B = torch.sigmoid(detail_pred).detach().cpu().numpy()
#         # out = Image.fromarray(B[0]*255)
#         # out = out.convert('L')
#         # out.save('detail.jpg')        
#         # pdb.set_trace()

#         # print(class_distribution)
#         # print(loss_mask_set)

#         assert len(self.stage_instance_loss_weight) == len(loss_mask_set)
#         loss_instance = sum([weight * loss for weight, loss in zip(self.stage_instance_loss_weight, loss_mask_set)]) 

#         return loss_instance
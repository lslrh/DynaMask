from abc import ABCMeta, abstractmethod

import torch.nn as nn
import torch
from mmdet.models.builder import build_roi_extractor
import torch.nn.functional as F
from ..builder import build_shared_head


class MaskPre(nn.Module):
    def __init__(self):
        super(MaskPre, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 4)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = x.view(x.size(0), 3136)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BaseRoIHead(nn.Module, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(BaseRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()
        self.semantic_roi_extractor = build_roi_extractor(dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=56, sampling_ratio=0),
            out_channels= 256,
            featmap_strides=[4, ]))
        self.mask_predictor = MaskPre()

    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self):
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_weights(self, pretrained):
        """Initialize the weights in head.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pass

    @abstractmethod
    def init_bbox_head(self):
        """Initialize ``bbox_head``"""
        pass

    @abstractmethod
    def init_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def forward_train(self,
                      x,
                      img_meta,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """Forward function during training."""
        pass

    async def async_simple_test(self, x, img_meta, **kwargs):
        """Asynchronized test function."""
        raise NotImplementedError

    def simple_test(self,
                    x,
                    proposal_list,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        pass

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        pass
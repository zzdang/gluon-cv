"""Region Proposal Target Generator."""
from __future__ import absolute_import

import numpy as np
import mxnet as mx
from mxnet import gluon
from ...nn.bbox import BBoxSplit
from ...nn.coder import SigmoidClassEncoder, NormalizedBoxCenterEncoder
from ...utils.nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from ...utils.nn.sampler import QuotaSampler


class RPNTargetGenerator(gluon.Block):
    def __init__(self, num_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5, stds=(1., 1., 1., 1.)):
        super(RPNTargetGenerator, self).__init__()
        self._num_sample = num_sample
        self._pos_iou_thresh = pos_iou_thresh
        self._neg_iou_thresh = neg_iou_thresh
        self._pos_ratio = pos_ratio
        self._bbox_split = BBoxSplit(axis=-1)
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(pos_iou_thresh)])
        # self._matcher = MaximumMatcher(pos_iou_thresh)
        self._sampler = QuotaSampler(num_sample, pos_iou_thresh, neg_iou_thresh, 0., pos_ratio)
        self._cls_encoder = SigmoidClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)

    def forward(self, bbox, anchor, width, height):
        """
        Only support batch_size=1 now.
        Be careful there's numpy operations inside
        """
        F = mx.nd
        # anchor with shape (N, 4)
        a_xmin, a_ymin, a_xmax, a_ymax = self._bbox_split(anchor)
        # valid anchor mask with shape (N, 1)
        # anchor_mask = ((a_xmin >= 0) * (a_ymin >= 0) *
            # (a_xmax <= width) * (a_ymax <= height)) > 0
        # anchor_mask = mx.nd.array(np.where(anchor_mask.asnumpy() > 0)[0], ctx=anchor.context)
        invalid_mask = ((a_xmin >= 0) * (a_ymin >= 0) *
            (a_xmax <= width) * (a_ymax <= height)) <= 0
        invalid_mask = mx.nd.array(np.where(invalid_mask.asnumpy() > 0)[0], ctx=anchor.context)
        # valid_anchor = anchor[anchor_mask, :]
        # broadcast to (N, 4) mask
        # anchor_mask = F.repeat(anchor_mask, axis=-1, repeats=4)
        # mask out invalid anchors, (N, 4)
        # valid_anchor = F.where(anchor_mask, anchor, F.zeros_like(anchor))
        # calculate ious between (N, 4) anchors and (M, 4) bbox ground-truths
        # ious is (N, M)
        ious = F.contrib.box_iou(anchor, bbox, format='corner').transpose((1, 0, 2))
        ious[:, invalid_mask, :] = -1
        matches = self._matcher(ious)
        samples = self._sampler(matches, ious)
        # samples = F.Custom(matches, ious, op_type='quota_sampler',
        #                      num_sample=self._num_sample,
        #                      pos_thresh=self._pos_iou_thresh,
        #                      neg_thresh_high=self._neg_iou_thresh,
        #                      neg_thresh_low=0.,
        #                      pos_ratio=self._pos_ratio)
        # training targets for RPN
        cls_target, cls_mask = self._cls_encoder(samples)
        box_target, box_mask = self._box_encoder(
            samples, matches, anchor.expand_dims(axis=0), bbox)
        return cls_target, box_target, box_mask

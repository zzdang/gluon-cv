"""Faster RCNN Model."""
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from .rcnn_target import RCNNTargetSampler, RCNNTargetGenerator
from ..rcnn import RCNN_2
from ..rpn import RPN
from ...nn.coder import NormalizedBoxCenterDecoder, MultiPerClassDecoder
from easydict import EasyDict as edict

__all__ = ['CascadeRCNN', 'get_cascade_rcnn',
           'cascade_rcnn_resnet50_v1b_voc',
           'cascade_rcnn_resnet50_v1b_coco',
           'cascade_rcnn_resnet50_v2a_voc',
           'cascade_rcnn_resnet50_v2a_coco',
           'cascade_rcnn_resnet50_v2_voc',
           'cascade_rcnn_vgg16_voc']


class CascadeRCNN(RCNN_2):
    r"""Faster RCNN network.

    Parameters
    ----------
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    train_patterns : str
        Matching pattern for trainable parameters.
    scales : iterable of float
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    ratios : iterable of float
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    roi_mode : str
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2
        (height, width) of the ROI region.
    stride : int, default is 16
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    rpn_channel : int, default is 1024
        Channel number used in RPN convolutional layers.
    nms_thresh : float, default is 0.3.
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    num_sample : int, default is 128
        Number of samples for RCNN targets.
    pos_iou_thresh : float, default is 0.5
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    neg_iou_thresh_high : float, default is 0.5
        Proposal whose IOU smaller than ``neg_iou_thresh_high``
        and larger than ``neg_iou_thresh_low`` is regarded as negative samples.
        Proposals with IOU in between ``pos_iou_thresh`` and ``neg_iou_thresh`` are
        ignored.
    neg_iou_thresh_low : float, default is 0.0
        See ``neg_iou_thresh_high``.
    pos_ratio : float, default is 0.25
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.

    """
    def __init__(self, features, top_features, scales, ratios, classes, roi_mode, roi_size,
                 stride=16, rpn_channel=1024, nms_thresh=0.3, nms_topk=400,
                 num_sample=128, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5,
                 neg_iou_thresh_low=0.0, pos_ratio=0.25, **kwargs):
        super(CascadeRCNN, self).__init__(
            features, top_features, classes, roi_mode, roi_size, **kwargs)
        self.stride = stride
        self._max_batch = 1  # currently only support batch size = 1
        self._max_roi = 100000  # maximum allowed ROIs
        stds_2nd = (.05, .05, .1, .1)
        stds_3rd = (.033, .033, .067, .067)
        means_2nd= (0., 0., 0., 0.)
        self._target_generator     = set([RCNNTargetGenerator(self.num_class)])
        self._target_generator_2nd = set([RCNNTargetGenerator(self.num_class, means_2nd, stds_2nd)])
        self._target_generator_3rd = set([RCNNTargetGenerator(self.num_class, means_2nd, stds_3rd)])
        with self.name_scope():
            self.rpn = RPN(rpn_channel, stride, scales=scales, ratios=ratios)
            self.sampler = RCNNTargetSampler(num_sample, pos_iou_thresh, neg_iou_thresh_high,
                                             neg_iou_thresh_low, pos_ratio,1)
            self.sampler_2nd = RCNNTargetSampler(-1, 0.6, 0.6,
                                             neg_iou_thresh_low, pos_ratio,0.95)
            self.sampler_3rd = RCNNTargetSampler(-1, 0.7, 0.7,
                                             neg_iou_thresh_low, pos_ratio,0.95)
            self.box_decoder_2nd = NormalizedBoxCenterDecoder(stds=(.05, .05, .1, .1))
            self.box_decoder_3rd = NormalizedBoxCenterDecoder(stds=(.033, .033, .067, .067))

    
    @property
    def target_generator(self):
        """Returns stored target generator

        Returns
        -------
        mxnet.gluon.HybridBlock
            The RCNN target generator

        """
        return list(self._target_generator)[0]
    @property
    def target_generator_2nd(self):
        return list(self._target_generator_2nd)[0]
    @property
    def target_generator_3rd(self):
        return list(self._target_generator_3rd)[0]

    def extract_ROI(self, F, feature, bbox):

        roi = self.add_batchid(F, bbox)

        # ROI features
        if self._roi_mode == 'pool':
            pooled_feat = F.ROIPooling(feature, roi, self._roi_size, 1. / self.stride)
        elif self._roi_mode == 'align':
            pooled_feat = F.contrib.ROIAlign(feature, roi, self._roi_size, 1. / self.stride)
        else:
            raise ValueError("Invalid roi mode: {}".format(self._roi_mode))
        return pooled_feat

    def add_batchid(self, F, bbox):
        with autograd.pause():
            roi_batchid = F.arange(0, self._max_batch, repeat=self._max_roi).reshape((-1, self._max_roi))
            roi_batchid = F.slice_like(roi_batchid, bbox * 0, axes=(0, 1))
            roi = F.concat(*[roi_batchid.reshape((-1, 1)), bbox.reshape((-1, 4))], dim=-1)
            return roi

    def decode_bbox(self, source_bbox, encoded_bbox, stds):
        with autograd.pause():
            box_decoder = NormalizedBoxCenterDecoder(stds=stds)
            roi = box_decoder(encoded_bbox, self.box_to_center(source_bbox))
            roi = roi.reshape((1,-1, 4))
            return roi



    def cascade_rcnn_org(self, F, feature, roi):
        """Forward Faster-RCNN network.

        The behavior during traing and inference is different.

        Parameters
        ----------
        feature: feature map
        roi: ROI region to be pooled (decoded bbox)


        Returns
        -------
        box_pred:  bbox prediction(encoded bbox) 
        cls_pred:  cls prediction

        """
        pooled_feat = self.extract_ROI(F=F, feature=feature, bbox=roi)
        top_feat = self.top_features(pooled_feat)
        cls_pred = self.class_predictor(top_feat)
        box_pred = self.box_predictor(top_feat).reshape((-1, 1, 4)).transpose((1, 0, 2))
        return cls_pred, box_pred


    def cascade_rcnn(self, F, feature, roi, sampler, gt_box):
        """Forward Faster-RCNN network.

        The behavior during traing and inference is different.

        Parameters
        ----------
        feature: feature map
        roi: ROI region to be pooled (decoded bbox)


        Returns
        -------
        box_pred:  bbox prediction(encoded bbox) 
        cls_pred:  cls prediction

        """

        if autograd.is_training():
            roi, samples, matches = sampler(roi, gt_box)
            sample_data = edict()
            sample_data.roi = roi
            sample_data.samples = samples
            sample_data.matches = matches

        pooled_feat = self.extract_ROI(F=F, feature=feature, bbox=roi)
        top_feat = self.top_features(pooled_feat)
        cls_pred = self.class_predictor(top_feat)
        box_pred = self.box_predictor(top_feat).reshape((-1, 1, 4)).transpose((1, 0, 2))


        if autograd.is_training():
            return cls_pred, box_pred, sample_data
        else:
            return cls_pred, box_pred, None



    def cascade_rcnn(self, F, feature, roi, sampler, gt_box):
        """Forward Faster-RCNN network.

        The behavior during traing and inference is different.

        Parameters
        ----------
        feature: feature map
        roi: ROI region to be pooled (decoded bbox)


        Returns
        -------
        box_pred:  bbox prediction(encoded bbox) 
        cls_pred:  cls prediction

        """

        if autograd.is_training():
            roi, samples, matches = sampler(roi, gt_box)
            sample_data = edict()
            sample_data.roi = roi
            sample_data.samples = samples
            sample_data.matches = matches

        pooled_feat = self.extract_ROI(F=F, feature=feature, bbox=roi)
        top_feat = self.top_features(pooled_feat)
        cls_pred = self.class_predictor(top_feat)
        box_pred = self.box_predictor(top_feat).reshape((-1, 1, 4)).transpose((1, 0, 2))


        if autograd.is_training():
            return cls_pred, box_pred, sample_data
        else:
            return cls_pred, box_pred, None


    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, gt_box=None):
        """Forward Faster-RCNN network.

        The behavior during traing and inference is different.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            The network input tensor.
        gt_box : type, only required during training
            The ground-truth bbox tensor with shape (1, N, 4).

        Returns
        -------
        (ids, scores, bboxes)
            During inference, returns final class id, confidence scores, bounding
            boxes.

        """
        feat = self.features(x)
        # RPN proposals
        if autograd.is_training():
            _, rpn_box, raw_rpn_score, raw_rpn_box, anchors = self.rpn(feat, F.zeros_like(x))
            # sample 128 roi
            assert gt_box is not None
            
        else:
            _, rpn_box = self.rpn(feat, F.zeros_like(x))

        
        if autograd.is_training():
            #rpn_box, samples, matches = self.sampler(rpn_box, gt_box)
            cls_pred, box_pred, sample_data_1st = self.cascade_rcnn(F=F, feature=feat, roi=rpn_box, sampler=self.sampler, gt_box=gt_box)            
            
            roi_2nd = self.decode_bbox(source_bbox=sample_data_1st.roi, encoded_bbox=box_pred, stds=(.05, .05, .1, .1))
            cls_pred_2nd, box_pred_2nd, sample_data_2nd = self.cascade_rcnn(F=F, feature=feat, roi=roi_2nd, sampler=self.sampler_2nd, gt_box=gt_box)

            roi_3rd = self.decode_bbox(source_bbox=sample_data_2nd.roi, encoded_bbox=box_pred_2nd, stds=(.033, .033, .067, .067))
            cls_pred_3rd, box_pred_3rd, sample_data_3rd = self.cascade_rcnn(F=F, feature=feat, roi=roi_3rd, sampler=self.sampler_3rd, gt_box=gt_box)

            box_pred = box_pred.transpose((1, 0, 2))
            box_pred_2nd = box_pred_2nd.transpose((1, 0, 2))
            box_pred_3rd = box_pred_3rd.transpose((1, 0, 2))

            # no need to convert bounding boxes in training, just return
            # translate bboxes

            rpn_result  = raw_rpn_score, raw_rpn_box, anchors
            cascade_rcnn_result = [  [cls_pred_3rd, box_pred_3rd, sample_data_3rd.roi, sample_data_3rd.samples, sample_data_3rd.matches ], 
                                     [cls_pred_2nd, box_pred_2nd, sample_data_2nd.roi, sample_data_2nd.samples, sample_data_2nd.matches],
                                     [cls_pred, box_pred, sample_data_1st.roi, sample_data_1st.samples, sample_data_1st.matches  ] ]
 
            return  rpn_result, cascade_rcnn_result

        else:
            cls_pred, box_pred, *_ = self.cascade_rcnn(F=F, feature=feat, roi=rpn_box, sampler=self.sampler_3rd, gt_box=gt_box)
            roi_2nd = self.decode_bbox(source_bbox=rpn_box, encoded_bbox=box_pred, stds=(.05, .05, .1, .1))
            cls_pred_2nd, box_pred_2nd, *_ = self.cascade_rcnn(F=F, feature=feat, roi=roi_2nd, sampler=self.sampler_3rd, gt_box=gt_box)
            roi_3rd = self.decode_bbox(source_bbox=roi_2nd, encoded_bbox=box_pred_2nd, stds=(.033, .033, .067, .067))
            cls_pred_3rd, box_pred_3rd, *_ = self.cascade_rcnn(F=F, feature=feat, roi=roi_3rd, sampler=self.sampler_3rd, gt_box=gt_box)
 
            # bboxes = self.box_decoder(box_pred, self.box_to_center(rpn_box)).split(
            #     axis=0, num_outputs=self.num_class, squeeze_axis=True)
            # cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
            
            bboxes = self.box_decoder_3rd(box_pred_3rd, self.box_to_center(roi_3rd)).split(
                axis=0, num_outputs=1, squeeze_axis=True)
            cls_prob_3rd = F.softmax(cls_pred_3rd, axis=-1)
            cls_prob_2nd = F.softmax(cls_pred_2nd, axis=-1)
            cls_prob_1st = F.softmax(cls_pred, axis=-1)
            cls_prob_3rd_avg = F.ElementWiseSum(cls_prob_3rd,cls_prob_2nd,cls_prob_1st)
            cls_ids, scores = self.cls_decoder(cls_prob_3rd_avg )
            results = []
            for i in range(self.num_class):
                cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
                score = scores.slice_axis(axis=-1, begin=i, end=i+1)
                # per class results
                bg_fg_bool=0 # min(i,0)
                per_result = F.concat(*[cls_id, score, bboxes[bg_fg_bool]], dim=-1)

                results.append(per_result)
            result = F.concat(*results, dim=0).expand_dims(0)
            if self.nms_thresh > 0 and self.nms_thresh < 1:
                result = F.contrib.box_nms(
                    result, overlap_thresh=self.nms_thresh, topk=self.nms_topk,
                    id_index=0, score_index=1, coord_start=2)
                if self.nms_topk > 0:
                    result = result.slice_axis(axis=1, begin=0, end=100).squeeze(axis=0)
            ids = F.slice_axis(result, axis=-1, begin=0, end=1)
            scores = F.slice_axis(result, axis=-1, begin=1, end=2)
            bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)

            
            return ids, scores, bboxes




def get_cascade_rcnn(name, features, top_features, scales, ratios, classes,
                    roi_mode, roi_size, dataset, stride=16,
                    rpn_channel=1024, pretrained=False, ctx=mx.cpu(),
                    root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Utility function to return faster rcnn networks.

    Parameters
    ----------
    name : str
        Model name.
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    scales : iterable of float
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    ratios : iterable of float
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    roi_mode : str
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2
        (height, width) of the ROI region.
    dataset : str
        The name of dataset.
    stride : int, default is 16
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    rpn_channel : int, default is 1024
        Channel number used in RPN convolutional layers.
    pretrained : bool, optional, default is False
        Load pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The Faster-RCNN network.

    """


    net = CascadeRCNN(features, top_features, scales, ratios, classes, roi_mode, roi_size,
                     stride=stride, rpn_channel=rpn_channel, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('cascade_rcnn', name, dataset))
 

        net.load_params(get_model_file(full_name, root=root), ctx=ctx)
    return net

def cascade_rcnn_resnet50_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_cascade_rcnn_resnet50_v1b_voc(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet50_v1b
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_cascade_rcnn('resnet50_v1b', features, top_features, scales=(2, 4, 8, 16, 32),
                           ratios=(0.5, 1, 2), classes=classes, dataset='voc',
                           roi_mode='align', roi_size=(14, 14), stride=16,
                           rpn_channel=1024, train_patterns=train_patterns,
                           pretrained=pretrained, **kwargs)

def cascade_rcnn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_cascade_rcnn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet50_v1b
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_cascade_rcnn('resnet50_v1b', features, top_features, scales=(2, 4, 8, 16, 32),
                           ratios=(0.5, 1, 2), classes=classes, dataset='coco',
                           roi_mode='align', roi_size=(14, 14), stride=16,
                           rpn_channel=1024, train_patterns=train_patterns,
                           pretrained=pretrained, **kwargs)

def cascade_rcnn_resnet50_v2a_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_cascade_rcnn_resnet50_v2a_voc(pretrained=True)
    >>> print(model)
    """
    from .resnet50_v2a import resnet50_v2a
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v2a(pretrained=pretrained_base)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['rescale'] + ['layer' + str(i) for i in range(4)]:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*stage(2|3|4)_conv'])
    print("~~~~~")
    print(features.collect_params())
    print(top_features.collect_params())
    return get_cascade_rcnn('resnet50_v2a', features, top_features, scales=(2, 4, 8, 16, 32),
                           ratios=(0.5, 1, 2), classes=classes, dataset='voc',
                           roi_mode='align', roi_size=(14, 14), stride=16,
                           rpn_channel=1024, train_patterns=train_patterns,
                           pretrained=pretrained, **kwargs)

def cascade_rcnn_resnet50_v2a_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_cascade_rcnn_resnet50_v2a_coco(pretrained=True)
    >>> print(model)
    """
    from .resnet50_v2a import resnet50_v2a
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v2a(pretrained=pretrained_base)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['rescale'] + ['layer' + str(i) for i in range(4)]:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*stage(2|3|4)_conv'])
    return get_cascade_rcnn('resnet50_v2a', features, top_features, scales=(2, 4, 8, 16, 32),
                           ratios=(0.5, 1, 2), classes=classes, dataset='coco',
                           roi_mode='align', roi_size=(14, 14), stride=16,
                           rpn_channel=1024, train_patterns=train_patterns,
                           pretrained=pretrained, **kwargs)

def cascade_rcnn_resnet50_v2_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_cascade_rcnn_resnet50_v2_voc(pretrained=True)
    >>> print(model)
    """
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = mx.gluon.model_zoo.vision.get_model('resnet50_v2', pretrained=pretrained_base)
    features = base_network.features[:8]
    top_features = base_network.features[8:11]

    train_patterns = '|'.join(['.*dense', '.*rpn', '.*stage(2|3|4)_conv'])
    return get_cascade_rcnn('resnet50_v2', features, top_features, scales=(2, 4, 8, 16, 32),
                           ratios=(0.5, 1, 2), classes=classes, dataset='voc',
                           roi_mode='align', roi_size=(14, 14), stride=16,
                           rpn_channel=1024, train_patterns=train_patterns,
                           pretrained=pretrained, **kwargs)




def cascade_rcnn_vgg16_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_cascade_rcnn_resnet50_v2_voc(pretrained=True)
    >>> print(model)
    """

    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = mx.gluon.model_zoo.vision.get_model('vgg16', pretrained=pretrained_base)
    features = base_network.features[:30]
    top_features =base_network.features[31:]
    # print("~~~~~")
    # print(features.collect_params())
    # print(top_features.collect_params())
    train_patterns = '|'.join(['.*dense', '.*rpn','.*vgg0_conv(4|5|6|7|8|9|10|11|12)'])
    return get_cascade_rcnn('vgg16', features, top_features, scales=( 16, 32,64),
                           ratios=(0.5, 1, 2), classes=classes, dataset='voc',
                           roi_mode='align', roi_size=(7, 7), stride=16,
                           rpn_channel=1024, train_patterns=train_patterns,
                           pretrained=pretrained, **kwargs)

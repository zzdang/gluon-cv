"""Transforms for RCNN series."""
from __future__ import absolute_import
import numpy as np
import mxnet as mx
from .. import bbox as tbbox
from .. import image as timage
from .. import experimental

__all__ = ['load_test', 'FasterRCNNDefaultTrainTransform', 'FasterRCNNDefaultValTransform']

def load_test(filenames, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or list of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(filenames, str):
        filenames = [filenames]
    tensors = []
    origs = []
    for f in filenames:
        img = mx.image.imread(f)
        img = mx.image.resize_short(img, short)
        if isinstance(max_size, int) and max(img.shape) > max_size:
            img = timage.resize_long(img, max_size)
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


class FasterRCNNDefaultTrainTransform(object):
    """Default Faster-RCNN training transform.

    Parameters
    ----------
    short : int, default is 600
        Resize image shorter side to ``short``.
    max_size : int, default is 1000
        Make sure image longer side is smaller than ``max_size``.
    anchors : mxnet.nd.NDArray, optional
        Anchors generated from RPN networks, the shape must be ``(1, N, 4)``.
        Since anchors are shared in the entire batch so it is ``1`` for the first dimension.
        ``N`` is the number of anchors for each image.

        .. hint::

            If anchors is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """
    def __init__(self, short=600, max_size=1000, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), box_norm=(1., 1., 1., 1.),
                 num_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5, stride=16, **kwargs):
        self._short = short
        self._max_size = max_size
        self._anchors = anchors
        self._mean = mean
        self._std = std
        self._stride = int(stride)
        if anchors is None:
            return

        self._anchors = anchors
        from ....model_zoo.rpn.rpn_target import RPNTargetGenerator
        self._target_generator = RPNTargetGenerator(
            num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh, pos_ratio=pos_ratio,
            stds=box_norm, **kwargs)

    def __call__(self, src, label):
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = timage.resize_short_within(src, short, max_size)
        bbox = tbbox.resize(label, (w, h), (img.shape[0], img.shape[1]))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype('float32')

        # generate RPN target so cpu workers can help reduce the workload
        feat_h, feat_w = (img.shape[1] // self._stride, img.shape[2] // self._stride)
        anchor = self._anchors[:, :, :feat_h, :feat_w, :].reshape((-1, 4))
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        cls_target, cls_mask, box_target, box_mask = self._target_generator(
            gt_bboxes, anchor, img.shape[2], img.shape[1])
        return img, cls_target[0], box_target[0], box_mask[0]


class FasterRCNNDefaultValTransform(object):
    """Default Faster-RCNN validation transform.

    Parameters
    ----------
    short : int, default is 600
        Resize image shorter side to ``short``.
    max_size : int, default is 1000
        Make sure image longer side is smaller than ``max_size``.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    def __init__(self, short=600, max_size=1000,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std

    def __call__(self, src, label):
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = timage.resize_short_within(src, short, max_size)
        bbox = tbbox.resize(label, (w, h), (img.shape[0], img.shape[1]))

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype('float32')

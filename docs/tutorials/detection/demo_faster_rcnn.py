"""2. Predict with pre-trained Faster RCNN models
==============================================

This article shows how to play with pre-trained Faster RCNN model.

First let's import some necessary libraries:
"""

from matplotlib import pyplot as plt
<<<<<<< HEAD
from mxnet import image
=======
>>>>>>> c29ba472a93c3d197cd1c5eabd6f3113d3330d18
import gluoncv
from gluoncv import model_zoo, data, utils

######################################################################
# Load a pretrained model
# -------------------------
#
<<<<<<< HEAD
# Let's get an Faster RCNN model trained on MS-COCO
=======
# Let's get an Faster RCNN model trained on Pascal VOC
>>>>>>> c29ba472a93c3d197cd1c5eabd6f3113d3330d18
# dataset with ResNet-50 backbone. By specifying
# ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.

<<<<<<< HEAD
net = gluoncv.model_zoo.get_model('faster_rcnn_resnet50_v2a_voc', pretrained=True)
=======
net = model_zoo.get_model('faster_rcnn_resnet50_v2a_voc', pretrained=True)
>>>>>>> c29ba472a93c3d197cd1c5eabd6f3113d3330d18

######################################################################
# Pre-process an image
# --------------------
#
# Next we download an image, and pre-process with preset data transforms. We
<<<<<<< HEAD
# resize the short edge of the image to 800 px and subtract the ImageNet mean.
#
=======
# resize the short edge of the image to 600 px and apply image transforms
# with mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)
>>>>>>> c29ba472a93c3d197cd1c5eabd6f3113d3330d18

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/biking.jpg?raw=true',
                          path='biking.jpg')
<<<<<<< HEAD
x, orig_img = gluoncv.data.transforms.presets.rcnn.load_test(im_fname)
=======
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)
>>>>>>> c29ba472a93c3d197cd1c5eabd6f3113d3330d18

######################################################################
# Inference and display
# ---------------------
#
# The Faster RCNN model
#
# We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
# results. We slice the results for the first image and feed them into `plot_bbox`:

box_ids, scores, bboxes = net(x)
ax = utils.viz.plot_bbox(orig_img, bboxes, scores, box_ids, class_names=net.classes)

plt.show()

import numpy as np
from PIL import Image
import sys
import cv2

caffe_root = '/home/sensetime/caffe-master/'
sys.path.append(caffe_root + 'python')
import caffe

pretrained_model = '/home/sensetime/caffe-master/models/fcn32s-heavy-pascal.caffemodel'
deploy_path = '/home/sensetime/caffe-master/qilu_test/fcn.berkeleyvision.org-master/voc-fcn32s/deploy.prototxt'
# deploy_path = '/home/sensetime/caffe-master/qilu_test/fcn.berkeleyvision.org-master/voc-fcn8s/deploy.prototxt'

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('/media/sensetime/64C43C54C43C2AA6/VOC2012/JPEGImages/2007_000187.jpg')
im = im.resize((224, 224))
# im.show()

in_ = np.array(im, dtype=np.float32)
in_ = in_[:, :, ::-1]
in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
in_ = in_.transpose((2, 0, 1))
#
#
# # load net
net = caffe.Net(deploy_path, pretrained_model, caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

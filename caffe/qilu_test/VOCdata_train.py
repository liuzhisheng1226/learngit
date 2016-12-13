import sys
import pprint
import numpy as np


caffe_root = '/home/sensetime/caffe-master/'
sys.path.append(caffe_root + 'python')
sys.path.append('/home/sensetime/caffe-master/qilu_test/fcn.berkeleyvision.org-master')
import caffe
from caffe import layers as L, params as P

#import voc_layers


caffe.set_device(0)
caffe.set_mode_gpu()

pretrained_model = '/home/sensetime/caffe-master/models/fcn32s-heavy-pascal.caffemodel'

solver = caffe.SGDSolver('/home/sensetime/caffe-master/qilu_test/fcn.berkeleyvision.org-master/voc-fcn32s/solver.prototxt')
solver.net.copy_from(pretrained_model)

niter = 200
test_interval = 25
# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))
output = np.zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data

    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

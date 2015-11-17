from __future__ import division
import numpy as np
from scipy.ndimage import zoom
from scipy.misc import imresize

class Hypercolumns(object):
    """ImageNet Hypercolumn features

    This class implements Hypercolumn features proposed by Hariharan etal.
    Features are constructed by passing the input image through the
    network, then concatenating the output of each of the convolutional layers.
    """

    def __init__(self,
            model='data/model.prototxt',
            network='data/network.caffemodel',
            mean=np.load('data/network_mean.npy'),
            layers=('conv1','conv2','conv3','conv4','conv5'), **kwargs):
        """Initialize the network for generating features

        Keyword Args:
            model: The caffe model description (.prototxt)
            network: The pretrained network (.caffemodel)
            layers: The layers to extract as features
            **kwargs: Arguments to forward to caffe.Classifier
        """
        import caffe
        self.layers = layers
        mean = mean.mean(1).mean(1)
        self.network = caffe.Classifier(model, network, mean=mean, channel_swap=(2,1,0), **kwargs)
        for layer in layers:
            if layer not in self.network.blobs:
                raise ValueError("'{}' not a valid network layer in '{}'".format(layer,model))

    @property
    def shape(self):
        """The native shape of the classifier"""
        return tuple(self.network.image_dims)

    def preprocess(self, image):
        """Preprocess the image for the given network architecture"""
        return self.network.transformer.preprocess('data',image)

    def compute(self, image):
        """Compute the hypercolumns for a given input image"""

        # preprocess the input
        data  = self.preprocess(image)
        shape = np.array(self.shape)
        # compute the network response
        prob = self.network.forward(data=data[None,...])
        # extract the responses
        layers = (self.network.blobs[layer].data[0] for layer in self.layers)
        features = np.row_stack((
            zoom(layer, np.append(1, shape / layer.shape[1:]), order=1) for layer in layers))

        # return the transpose of the features
        return features.transpose(1,2,0)

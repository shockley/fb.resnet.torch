# imports
import json
import time
import pickle
import scipy.misc
from scipy.sparse import *
#import skimage.io

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
#from threading import Thread
#from PIL import Image



from optparse import OptionParser



class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, opts):
        self.pascal_root = opts.pascal_root
        # get list of image indexes.
        list_file = opts.split + '.txt'
        self.indexlist = [line.rstrip('\n') for line in open(
            osp.join(self.pascal_root, 'ImageSets/Main', list_file))]
        self._cur = 0  # current image

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist))

    def load_allimages(self):
        """
        Load all images
        """
        index2labels = {}
        while self._cur != len(self.indexlist):
            index, multilabel = self.load_next_image()
            index2labels[index] = multilabel
        return index2labels


    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        # Load and prepare ground truth
        multilabel = np.zeros(20).astype(np.int32)
        anns = load_pascal_annotation(index, self.pascal_root)
        for label in anns['gt_classes']:
            # in the multilabel problem we don't care how MANY instances
            # there are of each class. Only if they are present.
            # The "-1" is b/c we are not interested in the background
            # class.
            multilabel[label - 1] = 1

        self._cur += 1
        return index, multilabel



def load_pascal_annotation(index, pascal_root):
    """
    This code is borrowed from Ross Girshick's FAST-RCNN code
    (https://github.com/rbgirshick/fast-rcnn).
    It parses the PASCAL .xml metadata files.
    See publication for further details: (http://arxiv.org/abs/1504.08083).

    Thanks Ross!

    """
    classes = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, xrange(21)))

    filename = osp.join(pascal_root, 'Annotations', index + '.xml')
    # print 'Loading: {}'.format(filename)

    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 21), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin')) - 1
        y1 = float(get_data_from_tag(obj, 'ymin')) - 1
        x2 = float(get_data_from_tag(obj, 'xmax')) - 1
        y2 = float(get_data_from_tag(obj, 'ymax')) - 1
        cls = class_to_ind[
            str(get_data_from_tag(obj, "name")).lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'index': index}


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--root", "-r", dest='pascal_root',
        default="/opt/caffe/data/pascal/VOC2012/", type="string", help="pascal_root folder")
    parser.add_option("--split", "-s", dest='split',
        default="trainval", type="string", help="split name, can be train, val, trainval")
    (opts, args) = parser.parse_args()
    loader = BatchLoader(opts)
    labelfile = "PascalLabel_" + opts.split + '.pickle'
    # labeldict = loader.load_allimages()
    # with open(labelfile, 'w') as f:
    #      pickle.dump(labeldict,f)
    with open(labelfile, 'r') as f:
         labeldict = pickle.load(f)
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.probeLoader import probeLoader
from sklearn.metrics import average_precision_score, accuracy_score, precision_recall_curve
import cv2
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_ground_truths
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from scipy.misc import imshow, imread
from datasets.factory import get_imdb

#from model.faster_rcnn.inceptionV3_classifier import inceptionV3
from model.faster_rcnn.my_inceptionV3_classifier import inceptionV3


from pathlib import Path
import pandas as pd
from cub_dataloader import CUB
import torch.nn.functional as F

import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ[" CUDA_LAUNCH_BLOCKING"]= "1"

import pdb


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='cub', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="/export/work/m.bharti/output",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true', default=False)
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=6, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=22411, type=int) #changed from 10021 of voc to psdb persnsearch: 22411 prw:11407
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')

  parser.add_argument('--eval_only', dest='eval_only',
                      help='evaluate only',
                      action='store_true')

  parser.add_argument('--exp_name', dest='exp_name',
                      help='Experiment name for the folder',
                      default='detection_iden_changed_aspect_ratio_flipped_on_class_agnostic')


  parser.add_argument('--include_id_loss', dest='include_id_loss',
                      help='Experiment name for the folder',
                      default=True, type=bool)

  parser.add_argument('--det_thresh', dest='det_thresh',
                      help='Detection threshold for person search evaluation ',
                      default=0.5, type=float)

  parser.add_argument('--gallery_size', dest='gallery_size',
                      help='Gallery Size for each probe',
                      default=100)

  parser.add_argument('--use_gt', dest='use_gt',
                      help='USe ground-truth as detections',
                      action='store_true'
                      )

  parser.add_argument('--dump_test_images', dest='dump_test_images',
                      help='Dump test images', action='store_true')

  parser.add_argument('--with_improv', dest='with_improvements',
                      help='Resnet with improvements', action='store_true')

  # added by Alessandro
  parser.add_argument('--use_cross_entropy', dest='use_cross_entropy',
                      help='whether use cross_entropy instead of OIM loss',
                      action='store_true', default=False)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY



if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)

  if args.dataset == "cub":
      # 5990 cub train, 11780 total
      total_ids = 200
      args.imdb_name = "cub_train"
      args.imdbval_name = "cub_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]',
                   'MAX_NUM_GT_BOXES', '80']
                   
  else: 
    print("This script is made for CUB only")
    sys.exit(0)

      # args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
  print("Using configuration file {}".format(args.cfg_file))

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.imdb_name + "/" + args.exp_name

  load_name= '/export/work/m.bharti/output/inception/cub_train/default/faster_rcnn_1_30_93.pth'
  # initilize the network here.
  inception = inceptionV3(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, include_id_loss=args.include_id_loss, use_cross_entropy=args.use_cross_entropy, total_ids=total_ids)

  #inception = Inception3()
  #fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  inception.load_state_dict(checkpoint['model'])
  inception = inception.cuda()

  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.use_gt:
      print("Using groud-truths as fixed proposals")
      proposals = torch.FloatTensor(1)
      if args.cuda:
          proposals = proposals.cuda()
      # make variable
      proposals = Variable(proposals, volatile=True)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    inception.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = args.exp_name #experiment name shud be same as in train
  num_images = len(imdb.image_index)
  all_boxes = [0 for _ in xrange(num_images)]
  all_features = [0 for _ in xrange(num_images)]

  output_dir = args.load_dir + "/" + args.net + "/" + args.imdbval_name + "/" + args.exp_name

  output_dir = osp.join(output_dir, "epoch_{}_use_gt_{}".format(args.checkepoch, args.use_gt))
  print(output_dir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)


  print("Using output directory {}".format(output_dir))

#
  PATH = Path('data/cub')
  labels = pd.read_csv(PATH/"image_class_labels.txt", header=None, sep=" ")
  labels.columns = ["id", "label"]

  train_test = pd.read_csv(PATH/"train_test_split.txt", header=None, sep=" ")
  train_test.columns = ["id", "is_train"]

  images = pd.read_csv(PATH/"images.txt", header=None, sep=" ")
  images.columns = ["id", "name"]

  classes = pd.read_csv(PATH/"classes.txt", header=None, sep=" ")
  classes.columns = ["id", "class"]


  valid_dataset = CUB(PATH, labels, train_test, images, train= False, transform= False)

  dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=100,
                                            shuffle=False, num_workers=0)


# #
 # output_dir = osp.join(output_dir, "whole_prw_ours")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)



  preds = {}
  trues = {}
  inception.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  criterion = nn.CrossEntropyLoss()

  total = 0
  correct = 0
  for i, (x, y) in enumerate(dataloader):
      batch = y.shape[0]
      
      x = x.cuda().float()
      y = y.cuda().long()

      _, y_pred = inception(x,y)

      # im = imread(valid_dataset.image_path_at(i))
      # im2show = np.copy(im)
      # imshow("ww", im2show)

      _, pred = torch.max(y_pred, 1)
      #print(pred)
      correct += (pred.data == y).sum()
      total += batch
      print('imgs:', total)
  print("accuracy", correct/total)  



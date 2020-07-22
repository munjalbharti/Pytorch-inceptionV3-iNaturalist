# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import torch

from my_inceptionV3_classifier import inceptionV3

from pathlib import Path
import pandas as pd
from cub_dataloader import CUB

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"




try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')



  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')

  args = parser.parse_args()
  return args




if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  total_ids = 200
  args.imdb_name = "cub_train"
  args.imdbval_name = "cub_test"

  load_name= '/export/work/m.bharti/output/inception/cub_train/default/faster_rcnn_1_30_93.pth'

  # initilize the network here.
  inception = inceptionV3(200)

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  inception.load_state_dict(checkpoint['model'])
  inception = inception.cuda()

  print('load model successfully!')


  if args.cuda:
    inception.cuda()

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

  dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=100, shuffle=False, num_workers=0)


  preds = {}
  trues = {}
  inception.eval()

  total = 0
  correct = 0
  for i, (x, y) in enumerate(dataloader):
      batch = y.shape[0]
      
      x = x.cuda().float()
      y = y.cuda().long()

      _, y_pred = inception(x,y)

      _, pred = torch.max(y_pred, 1)
      correct += (pred.data == y).sum()

      total += batch
      print('imgs:', total)

  print("accuracy", correct/total)



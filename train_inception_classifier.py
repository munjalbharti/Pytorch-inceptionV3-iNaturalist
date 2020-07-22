from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import torch
from torch.autograd import Variable

from torch.utils.data.sampler import Sampler

#from model.utils.config import cfg, cfg_from_file, cfg_from_list


import os

from my_inceptionV3_classifier import inceptionV3
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import pandas as pd
from cub_dataloader import CUB

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_checkpoint(state, filename):
    torch.save(state, filename)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train Inception network on CUB dataset using iNaturalist pretrained model')

    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='cub', type=str)  # changed from pascal_voc

    parser.add_argument('--net', dest='net',
                        help='vgg16, res101,inception',
                        default='inception', type=str)  # vgg16

    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)

    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=30, type=int)


    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="/export/work/m.bharti/output",
                        type=str)

    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)

    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')

    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=64, type=int)

    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.01, type=float)

    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=10, type=int)

    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)

    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)

    parser.add_argument('--exp_name', dest='exp_name',
                        help='Experiment name for the folder',
                        default='default1')



    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    total_ids = 200
    args.imdb_name = "cub_train"
    args.imdbval_name = "cub_test"

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    output_dir = args.save_dir + "/" + args.net + "/" + args.imdb_name + "/" + args.exp_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    PATH = Path('data/cub')
    labels = pd.read_csv(PATH / "image_class_labels.txt", header=None, sep=" ")
    labels.columns = ["id", "label"]

    train_test = pd.read_csv(PATH / "train_test_split.txt", header=None, sep=" ")
    train_test.columns = ["id", "is_train"]

    images = pd.read_csv(PATH / "images.txt", header=None, sep=" ")
    images.columns = ["id", "name"]

    classes = pd.read_csv(PATH / "classes.txt", header=None, sep=" ")
    classes.columns = ["id", "class"]

    train_dataset = CUB(PATH, labels, train_test, images, train=True, transform=True)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)


    print("###########Input network {}###########".format(args.net))

    inception = inceptionV3(200)
    lr = args.lr

    params = []

    weight_decay = 0.00004

    for key, value in inception.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr , 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]


    optimizer = torch.optim.SGD(params, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    iter_no = -1

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'cub_inception_{}.pth'.format(args.checkepoch))
        print("loading checkpoint %s" % (load_name))

        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']

        inception.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("loaded checkpoint %s" % (load_name))


    if args.cuda:
        inception.cuda()


    iters_per_epoch = len(dataloader)
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        inception.train()
        loss_temp = 0
        start = time.time()

        scheduler.step()

        for i, (x, y) in enumerate(dataloader):
            iter_no = iter_no + 1

            x = x.cuda().float()
            y = y.cuda().long()
            inception.zero_grad()

            loss, _ = inception(x, y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            print("[epoch %2d][iter %4d/%4d][iter_no %4d] loss: %.4f, lr: %.4f" \
                  % (epoch, i, iters_per_epoch, iter_no, loss, optimizer.param_groups[0]['lr']))


        save_name = os.path.join(output_dir, 'cub_inception_{}.pth'.format(epoch))
        save_checkpoint({
                'epoch': epoch + 1,
                'model': inception.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_name)

        print('save model: {}'.format(save_name))

        end = time.time()
        print(end - start)


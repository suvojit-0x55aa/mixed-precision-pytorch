from __future__ import print_function

import argparse

import torch

from data import testloader, trainloader
from models import model_factory
from train import Trainer

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning Rate')
parser.add_argument('--steps', '-n', default=200, type=int, help='No of Steps')
parser.add_argument('--gpu', '-p', action='store_true', help='Train on GPU')
parser.add_argument(
    '--fp16', action='store_true', help='Train with FP16 weights')
parser.add_argument(
    '--loss_scaling', '-s', action='store_true', help='Scale FP16 losses')
parser.add_argument(
    '--model', '-m', default='resnet50', type=str, help='Name of Network')
args = parser.parse_args()

train_on_gpu = False
if args.gpu and torch.cuda.is_available():
    train_on_gpu = True
    # CuDNN must be enabled for FP16 training.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

model_name = args.model
model = model_factory(model_name)

trainer = Trainer(model_name, model, train_on_gpu, args.fp16,
                  args.loss_scaling)

trainer.train_and_evaluate(trainloader, testloader, args.steps, args.lr)

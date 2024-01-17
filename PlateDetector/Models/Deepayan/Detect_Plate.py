import pandas as pd

import sys
import os
import cv2
import pdb
import json
import math
import pickle
import logging
import warnings
from tqdm import *
import numpy as np
import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torch.utils.data import random_split
from argparse import ArgumentParser

from PlateDetector.Models.Deepayan.Model.Models import CustomCTCLoss
from PlateDetector.Models.Deepayan.utils.utils import *
from PlateDetector.Models.Deepayan.Model.Models import CRNN
from PlateDetector.Models.Deepayan.Data.PickleDataset import PickleDataset
from PlateDetector.Models.Deepayan.Data.DataSynthsis import SynthDataset, SynthCollator

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_accuracy(args):
    loader = torch.utils.data.DataLoader(args.data,
                                         batch_size=args.batch_size,
                                         collate_fn=args.collate_fn)
    model = args.model
    model.eval()
    converter = OCRLabelConverter(args.alphabet)
    evaluator = Eval()
    labels, predictions = [], []
    for iteration, batch in enumerate(tqdm(loader)):
        input_, targets = batch['img'].to(device), batch['label']
        labels.extend(targets)
        targets, lengths = converter.encode(targets)
        logits = model(input_).transpose(1, 0)
        logits = torch.nn.functional.log_softmax(logits, 2)
        logits = logits.contiguous().cpu()
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        probs, pos = logits.max(2)
        pos = pos.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(pos.data, pred_sizes.data, raw=False)
        predictions.extend(sim_preds)
    ca = np.mean((list(map(evaluator.char_accuracy, list(zip(predictions, labels))))))
    wa = np.nanmean((list(map(evaluator.word_accuracy_line, list(zip(predictions, labels))))))
    return ca, wa


def main(**kwargs):
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--imgdir", type=str, default='source_bold')
    parser.add_argument("--log_dir", type=str, default='logs')
    parser.add_argument("--save_dir", type=str, default='Train/saves')

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--imgH", type=int, default=48)
    parser.add_argument("--nHidden", type=int, default=256)
    parser.add_argument("--nChannels", type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--language", type=str, default='English')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--schedule", action='store_true')

    args = parser.parse_args()
    args.data = SynthDataset(args)
    args.collate_fn = SynthCollator()
    args.alphabet = """Only thewigsofrcvdampbkuq.$A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%"""
    args.nClasses = len(args.alphabet)
    model = CRNN(args)
    if(torch.cuda.is_available()):
        model = model.cuda()
    resume_file = os.path.join(args.save_dir, args.name, 'best.ckpt')
    if os.path.isfile(resume_file):
        print('Loading model %s' % resume_file)
        checkpoint = torch.load(resume_file, map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])
        args.model = model
        ca, wa = get_accuracy(args)
        print("Character Accuracy: %.2f\nWord Accuracy: %.2f" % (ca, wa))
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))
        print('Exiting')


if __name__ == '__main__':
    main()

import logging
import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import *
from torch.utils.data import random_split

from PlateDetector.Models.Deepayan.Data.PickleDataset import PickleDataset
from PlateDetector.Models.Deepayan.Model.Models import CRNN, CustomCTCLoss
from PlateDetector.Models.Deepayan.utils import utils
from collections import OrderedDict
from PlateDetector.Models.Deepayan.Data.DataSynthsis import SynthDataset, SynthCollator

class OCRTrainer(object):
    def __init__(self, opt):
        super(OCRTrainer, self).__init__()
        self.data_train = opt.data_train
        self.data_val = opt.data_val
        self.model = opt.model
        self.criterion = opt.criterion
        self.optimizer = opt.optimizer
        self.schedule = opt.schedule
        self.converter = utils.OCRLabelConverter(opt.alphabet)
        self.evaluator = utils.Eval()
        print('Scheduling is {}'.format(self.schedule))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=opt.epochs)
        self.batch_size = opt.batch_size
        self.count = opt.epoch
        self.epochs = opt.epochs
        self.cuda = opt.cuda
        self.collate_fn = opt.collate_fn
        self.init_meters()

    def init_meters(self):
        self.avgTrainLoss = utils.AverageMeter("Train loss")
        self.avgTrainCharAccuracy = utils.AverageMeter("Train Character Accuracy")
        self.avgTrainWordAccuracy = utils.AverageMeter("Train Word Accuracy")
        self.avgValLoss = utils.AverageMeter("Validation loss")
        self.avgValCharAccuracy = utils.AverageMeter("Validation Character Accuracy")
        self.avgValWordAccuracy = utils.AverageMeter("Validation Word Accuracy")

    def forward(self, x):
        if(torch.cuda.is_available()):
            x = x.to(torch.device("cuda"))
        logits = self.model(x)
        return logits.transpose(1, 0)

    def loss_fn(self, logits, targets, pred_sizes, target_sizes):
        loss = self.criterion(logits, targets, pred_sizes, target_sizes)
        return loss


    def step(self):
        self.max_grad_norm = 0.05
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def schedule_lr(self):
        if self.schedule:
            self.scheduler.step()

    def _run_batch(self, batch, report_accuracy=False, validation=False):
        input_, targets = batch['img'], batch['label']
        targets, lengths = self.converter.encode(targets)
        logits = self.forward(input_)
        logits = logits.contiguous().cpu()
        logits = torch.nn.functional.log_softmax(logits, 2)
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        targets= targets.view(-1).contiguous()
        loss = self.loss_fn(logits, targets, pred_sizes, lengths)
        if report_accuracy:
            probs, preds = logits.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, pred_sizes.data, raw=False)
            ca = np.mean((list(map(self.evaluator.char_accuracy, list(zip(sim_preds, batch['label']))))))
            wa = np.mean((list(map(self.evaluator.word_accuracy, list(zip(sim_preds, batch['label']))))))
        return loss, ca, wa

    def run_epoch(self, validation=False):
        if not validation:
            loader = self.train_dataloader()
            pbar = tqdm(loader, desc='Epoch: [%d]/[%d] Training'%(self.count, self.epochs), leave=True)
            self.model.train()
        else:
            loader = self.val_dataloader()
            pbar = tqdm(loader, desc='Validating', leave=True)
            self.model.eval()
        outputs = []
        for batch_nb, batch in enumerate(pbar):
            if not validation:
                output = self.training_step(batch)
            else:
                output = self.validation_step(batch)
            pbar.set_postfix(output)
            outputs.append(output)
        self.schedule_lr()
        if not validation:
            result = self.train_end(outputs)
        else:
            result = self.validation_end(outputs)
        return result

    def training_step(self, batch):
        loss, ca, wa = self._run_batch(batch, report_accuracy=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.step()
        output = OrderedDict({
            'loss': abs(loss.item()),
            'train_ca': ca.item(),
            'train_wa': wa.item()
            })
        return output

    def validation_step(self, batch):
        loss, ca, wa = self._run_batch(batch, report_accuracy=True, validation=True)
        output = OrderedDict({
            'val_loss': abs(loss.item()),
            'val_ca': ca.item(),
            'val_wa': wa.item()
            })
        return output

    def train_dataloader(self):
        # logging.info('training data loader called')
        loader = torch.utils.data.DataLoader(self.data_train,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                shuffle=True)
        return loader

    def val_dataloader(self):
        # logging.info('val data loader called')
        loader = torch.utils.data.DataLoader(self.data_val,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn)
        return loader

    def train_end(self, outputs):
        for output in outputs:
            self.avgTrainLoss.add(output['loss'])
            self.avgTrainCharAccuracy.add(output['train_ca'])
            self.avgTrainWordAccuracy.add(output['train_wa'])

        train_loss_mean = abs(self.avgTrainLoss.compute())
        train_ca_mean = self.avgTrainCharAccuracy.compute()
        train_wa_mean = self.avgTrainWordAccuracy.compute()

        result = {'train_loss': train_loss_mean, 'train_ca': train_ca_mean,
        'train_wa': train_wa_mean}
        # result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': train_loss_mean}
        return result

    def validation_end(self, outputs):
        for output in outputs:
            self.avgValLoss.add(output['val_loss'])
            self.avgValCharAccuracy.add(output['val_ca'])
            self.avgValWordAccuracy.add(output['val_wa'])

        val_loss_mean = abs(self.avgValLoss.compute())
        val_ca_mean = self.avgValCharAccuracy.compute()
        val_wa_mean = self.avgValWordAccuracy.compute()

        result = {'val_loss': val_loss_mean, 'val_ca': val_ca_mean,
        'val_wa': val_wa_mean}
        return result


class Learner(object):
    def __init__(self, model, optimizer, savepath=None, resume=False):
        self.model = model
        self.optimizer = optimizer
        self.savepath = os.path.join(savepath, 'best.ckpt')
        self.cuda = torch.cuda.is_available()
        self.cuda_count = torch.cuda.device_count()
        if self.cuda:
            self.model = self.model.cuda()
        self.epoch = 0
        if self.cuda_count > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.best_score = None
        if resume and os.path.exists(self.savepath):
            self.checkpoint = torch.load(self.savepath)
            self.epoch = self.checkpoint['epoch']
            self.best_score = self.checkpoint['best']
            self.load()
        else:
            print('checkpoint does not exist')

    def fit(self, opt):
        opt.cuda = self.cuda
        opt.model = self.model
        opt.optimizer = self.optimizer
        logging.basicConfig(filename="%s/%s.csv" % (opt.log_dir, opt.name), level=logging.INFO)
        self.saver = utils.EarlyStopping(self.savepath, patience=15, verbose=True, best_score=self.best_score)
        opt.epoch = self.epoch
        trainer = OCRTrainer(opt)

        for epoch in range(opt.epoch, opt.epochs):
            train_result = trainer.run_epoch()
            val_result = trainer.run_epoch(validation=True)
            trainer.count = epoch
            info = '%d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f' % (epoch, train_result['train_loss'],
                                                               val_result['val_loss'], train_result['train_ca'],
                                                               val_result['val_ca'],
                                                               train_result['train_wa'], val_result['val_wa'])
            logging.info(info)
            self.val_loss = val_result['val_loss']
            print(self.val_loss)
            if self.savepath:
                self.save(epoch)
            if self.saver.early_stop:
                print("Early stopping")
                break

    def load(self):
        print('Loading checkpoint at {} trained for {} epochs'.format(self.savepath, self.checkpoint['epoch']))
        self.model.load_state_dict(self.checkpoint['state_dict'])
        if 'opt_state_dict' in self.checkpoint.keys():
            print('Loading optimizer')
            self.optimizer.load_state_dict(self.checkpoint['opt_state_dict'])

    def save(self, epoch):
        self.saver(self.val_loss, epoch, self.model, self.optimizer)

if (__name__=="__main__"):
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--imgdir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default='logs')
    parser.add_argument("--save_dir", type=str, default='saves')

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--imgH", type=int, default=48)
    parser.add_argument("--nHidden", type=int, default=1024)
    parser.add_argument("--nChannels", type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--language", type=str, default='English')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--schedule", action='store_true')

    args = parser.parse_args()

    data = SynthDataset(args)
    args.collate_fn = SynthCollator()
    train_split = int(0.8 * len(data))
    val_split = len(data) - train_split
    args.data_train, args.data_val = random_split(data, (train_split, val_split))
    print('Traininig Data Size:{}\nVal Data Size:{}'.format(
        len(args.data_train), len(args.data_val)))

    alphabet = """Only thewigsofrcvdampbkuq.$A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%"""
    args.nClasses = len(args.alphabet)
    model = CRNN(args)
    args.criterion = CustomCTCLoss()
    savepath = os.path.join(args.save_dir, args.name)
    utils.gmkdir(savepath)
    utils.gmkdir(args.log_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    learner = Learner(model, optimizer, savepath=savepath, resume=args.resume)
    learner.fit(args)
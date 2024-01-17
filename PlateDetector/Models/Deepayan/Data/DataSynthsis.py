from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from tqdm import *
import torch
from PlateDetector.Models.Deepayan.utils import utils
import pandas as pd
import cv2

class SynthDataset(Dataset):
    def __init__(self, opt):
        super(SynthDataset,self).__init__()
        self.path = opt.imgdir
        self.data = pd.read_excel(self.path)

        self.nSamples = self.data.shape[0]

        transform_list = [transforms.Grayscale(1),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_list)
        self.collate_fn = SynthCollator()

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        row = self.data.iloc[index]
        imagepath = row["plate_path"]
        # imagefile = os.path.basename(imagepath)


        img = Image.open(imagepath)
        if self.transform is not None:
            img = self.transform(img)
        item = {'img': img, 'idx':index}
        item['label'] = str(row["license_number"])

        if(index % 10 == 0):
            img1 = cv2.imread(imagepath)
            img1 = cv2.putText(img1,item['label'],(5,5),1,0.7,(0,0,255))
            cv2.imwrite("img.png",img1)

        return item

class SynthCollator(object):

    def __call__(self, batch):
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1],
                           max(width)], dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)
        item = {'img': imgs, 'idx': indexes}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        return item

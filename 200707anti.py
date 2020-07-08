# -*- coding: utf-8 -*-


import os
import csv
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import models
import torch.nn as nn
from fastai.vision import Path
import torch
from torch.autograd import Variable

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NUMBER_LEN = len(NUMBER)
MAX_CAPTCHA = 5

def encode(a):
    onehot = [0]*NUMBER_LEN
    idx = NUMBER.index(a)
    onehot[idx] += 1
    return onehot

class Mydataset(Dataset):
    def __init__(self, path, is_train=True, transform=None):
        self.path = path
        if is_train: self.img = os.listdir(self.path)[:1000]
        else: self.img = os.listdir(self.path)[1001:]
        
        except: pass
        self.transform = transform
        
    def __getitem__(self, idx):
        img_path = self.img[idx]
        img = Image.open(self.path/img_path)
        img = img.convert('L')
        label = Path(self.path/img_path).name[:-4]
        label_oh = []
        for i in label:
            label_oh += encode(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, np.array(label_oh), label
    
    def __len__(self):
        return len(self.img)

from google.colab import drive #подключаю google drive
drive.mount('/content/gdrive/')

import shutil
shutil.rmtree('pix/')

!unzip -q /content/gdrive/My\ Drive/captcha/data.zip -d pix # распаковываю архив с картинками в папку pix

labels = pd.read_csv('/content/gdrive/My Drive/captcha/data.csv', sep = ',', encoding = 'utf-8', header = None)
labels[0] = labels[0].astype( str )
labels = dict(zip(labels[0], labels[1]))

import glob

from google.colab import drive
from google.colab import files
data = 'pix/'

ldseg = os.listdir(data)
dst = ""
src= ""

for filename in ldseg:
    key = filename[:-4]
   
    dst = data + labels.get(key) + ".jpg"
    src = data + filename

    os.rename(src, dst)

print (ldseg)

!ls pix

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])

train_ds = Mydataset(Path('pix'), transform = transform)

test_ds = Mydataset(Path('pix'), False, transform)

train_dl = DataLoader (train_ds, batch_size = 64, num_workers = 0)

test_dl = DataLoader (train_ds, batch_size = 1, num_workers = 0)

model = models.resnet18( pretrained = False )

model.conv1 = nn.Conv2d(1, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias=False)

model.fc = nn.Linear(in_features = 512, out_features = NUMBER_LEN * MAX_CAPTCHA, bias=True)

model.cuda()

loss_func = nn.MultiLabelSoftMarginLoss()
optm = torch.optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(20):
    for step, i in enumerate(train_dl):
        img, label_oh, keys = i
        img = Variable(img).cuda()
        label_oh = Variable(label_oh.float()).cuda()
        pred = model(img)
        loss = loss_func(pred, label_oh)
        optm.zero_grad()
        loss.backward()
        optm.step()
        print('epoch:', epoch + 1, 'step:', step + 1, 'loss:', loss.item())

model.eval()

for step, (img, label_oh, label) in enumerate(test_dl):
    img = Variable(img).cuda()
    pred = model(img)

    c0 = NUMBER[np.argmax(pred.squeeze().cpu().tolist()[0: NUMBER_LEN])]
    c1 = NUMBER[np.argmax(pred.squeeze().cpu().tolist()[NUMBER_LEN: NUMBER_LEN*2])]
    c2 = NUMBER[np.argmax(pred.squeeze().cpu().tolist()[NUMBER_LEN*2: NUMBER_LEN*3])]
    c3 = NUMBER[np.argmax(pred.squeeze().cpu().tolist()[NUMBER_LEN*3: NUMBER_LEN*4])]
    c4 = NUMBER[np.argmax(pred.squeeze().cpu().tolist()[NUMBER_LEN*4: NUMBER_LEN*5])]
    c = '%s%s%s%s%s' % (c0, c1, c2, c3, c4)

    print('label:', label[0], 'pred:', c)

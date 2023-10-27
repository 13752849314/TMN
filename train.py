import argparse
import logging
import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from options import parse, dict2str
from utils.common import setup_logger
from Network import getNetwork
from loss import STLoss
from dataset import getdataSet

# load config file
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options yaml file.')
args = parser.parse_args()
opt = parse(args.opt)

# set log
name = 'train_' + opt['name']
setup_logger(opt['name'], opt['logs_path'], name, level=logging.INFO,
             screen=True, tofile=True)
logger = logging.getLogger(opt['name'])
logger.info(dict2str(opt))  # print config file

# get network
device = torch.device('cuda') if opt['use_cuda'] and torch.cuda.is_available() else torch.device('cpu')
model = getNetwork(opt)
if opt['gpus'] > 1:
    model = nn.DataParallel(model).to(device)
else:
    model = model.to(device)
logger.info(model)
# loss function
if opt['lossF'] == 'L1':
    criteon = nn.L1Loss().to(device)
elif opt['lossF'] == 'SLoss':
    criteon = STLoss(weight=opt['loss_weight']).to(device)
else:
    raise NotImplementedError(f"Loss function {opt['lossF']} don't implement!")
# optimizer
optimizer = optim.Adam(model.parameters(), lr=opt['lr'])
# load training data
train_data = getdataSet(opt['data_name'], opt, model='train')
train = DataLoader(train_data, batch_size=opt['batch_size'])

# start training
epochs = opt['epochs']
sf = opt['sf']

for epoch in range(epochs):
    model.train()
    total_loss = []

    for batchidx, item in enumerate(train):
        x, label = item['LR'].to(device), item['HR'].to(device)
        out = model(x)
        loss = criteon(out, label)
        total_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(f"epoch={epoch + 1}/{epochs}  batch={batchidx + 1}/{len(train)}  loss={loss.item()}")

    logger.info(f"epoch={epoch + 1}  avg_loss={sum(total_loss) / len(total_loss)}")
    if (epoch + 1) % sf == 0:
        torch.save(model.state_dict(), os.path.join(opt['model_path'], f"{epoch + 1}_{epochs}.pth"))

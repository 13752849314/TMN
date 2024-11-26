import argparse
import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from options import parse, dict2str
from utils.common import setup_logger, makedir, get_timestamp, tensor2img, save_img
from Network import getNetwork
from dataset import getdataSet
from utils.data import forward_chop
from loss import evaluation

# load config file
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options yaml file.')
args = parser.parse_args()
opt = parse(args.opt)

# set log
name = 'test_' + opt['name']
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

# load weight
weight = torch.load(os.path.join(opt['model_path'], opt['check_point']), map_location=device)
model.load_state_dict(weight)
model.eval()

with torch.no_grad():
    for k, v in opt['test_data'].items():
        val_path = os.path.join(opt['result_path'],
                                f"{opt['check_point'].split('.')[0]}_test_{v['data_name']}_{get_timestamp()}_{opt['min_size']}")
        makedir([val_path])
        opt['data_path'] = v['data_path']
        test_data = getdataSet(v['data_name'], opt, 'test')
        test = DataLoader(test_data, batch_size=1)
        total_p = 0.
        total_s = 0.
        logger.info(f"Test {v['data_name']}")
        for item in test:
            x = item['LR'].to(device)
            y = item['HR'].to(device)
            out = forward_chop(model, x, scale=opt['scale'], min_size=opt['min_size'])
            sr_img = tensor2img(out[0])
            save_img(sr_img, os.path.join(val_path, f"{item['name'][0]}.png"))
            p, s = evaluation(out[0].clamp_(0, 1), y[0])
            total_p += p
            total_s += s
            logger.info(f"image {item['name'][0]}\n"
                        f"psnr={p}\n"
                        f"ssim={s}")
        logger.info(f"avg_psnr={total_p / len(test)}")
        logger.info(f"avg_ssim={total_s / len(test)}")

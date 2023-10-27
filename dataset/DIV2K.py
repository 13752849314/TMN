# create by 敖鸥 at 2023/5/4
import os
import csv
import random

from torch.utils.data import Dataset

from utils.data import ToTensor, openImg
from utils.data import div2k


class DIV2K(Dataset):
    def __init__(self, opt, model='train', scale='X2', tf=None, sub='', lr_type='bicubic', fz=8, fm=9):
        super(DIV2K, self).__init__()
        assert model in ['train', 'val', 'test'], f'Model {model} is not support!'
        assert scale in ['X2', 'X3', 'X4'], f'Scale {scale} is not support!'
        self.opt = opt
        self.model = model
        self.scale = scale
        self.sub = sub
        self.lr_type = lr_type
        self.path = opt['data_path']
        self.upscale = int(self.scale[-1])
        self.patch_size = opt['patch_size']
        if tf is None:
            self.tf = ToTensor()
        else:
            self.tf = tf
        if model == 'test':
            self.sub = ''
            self.x_path = os.path.join(self.path, f'DIV2K_{self.model}_LR_{self.lr_type}', self.scale)
        else:
            self.x_path = os.path.join(self.path, f'DIV2K_train_LR_{self.lr_type}', self.scale + self.sub)
            self.y_path = os.path.join(self.path, f'DIV2K_train_HR{self.sub}')
        self.images = self._load_csv()
        if self.model == 'train':
            self.images = self.images[:len(self.images) * fz // fm]
        elif model == 'val':
            self.images = self.images[len(self.images) * fz // fm:]
        else:
            pass

    def __getitem__(self, index):
        name = self.images[index]
        x = os.path.join(self.x_path, name)
        if self.model == 'test':
            tag = None
        else:
            if self.sub == '':
                name_tag = name.replace(self.scale.lower(), '')
                tag = os.path.join(self.y_path, name_tag)
            else:
                tag = os.path.join(self.y_path, name)
            if self.model == 'train':
                tag, x = div2k(tag, x, patch_size=self.patch_size // self.upscale, scale=self.upscale)
            else:
                tag, x = openImg(tag), openImg(x)
            tag = self.tf(tag)
        x = self.tf(x)
        return {'LR': x, 'HR': tag, 'name': name.split('.')[0]}

    def __len__(self):
        return len(self.images)

    def _load_csv(self):
        model = self.model if self.model != 'val' else 'train'
        name = model + '_' + self.lr_type + '_' + self.scale + self.sub + '.csv'
        print(name)
        if not os.path.exists(os.path.join(self.path, name)):
            images = []
            for i in os.listdir(self.x_path):
                images.append(i)
            random.shuffle(images)
            with open(os.path.join(self.path, name), mode='w', newline='') as f:
                writer = csv.writer(f)
                for i in images:
                    writer.writerow([i])
        images = []
        with open(os.path.join(self.path, name)) as f:
            reader = csv.reader(f)
            for raw in reader:
                images.append(raw[0])
        return images


if __name__ == '__main__':
    arg = {'data_path': r'../data/DIV2K'}
    data = DIV2K(arg, model='val', sub='_sub')
    print(data.__len__())

    item = next(iter(data))
    print(item)

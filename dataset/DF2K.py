import os
import random
import csv

from torch.utils.data import Dataset

from utils.data import div2k
from utils.data import ToTensor


class DF2K(Dataset):

    def __init__(self, opt, scale=4, model='train', tf=None):
        super(DF2K, self).__init__()
        assert model in ['train'], f'Model {model} is not support!'
        assert scale in [2, 3, 4], f'Scale {scale} is not support!'
        self.opt = opt
        self.model = model
        self.scale = scale
        self.path = opt['data_path']
        self.patch_size = opt['patch_size']
        self.x_path = os.path.join(self.path, 'X' + str(scale))
        self.y_path = os.path.join(self.path, 'HR')

        if tf is None:
            self.tf = ToTensor()
        else:
            self.tf = tf

        self.images = self._load_csv()

    def __getitem__(self, index):
        name = self.images[index]
        x_path = os.path.join(self.x_path, name)
        y_path = os.path.join(self.y_path, name)
        tag, x = div2k(y_path, x_path, self.patch_size // self.scale, self.scale)
        tag = self.tf(tag)
        x = self.tf(x)
        return {'LR': x, 'HR': tag, 'name': name.split('.')[0]}

    def __len__(self):
        return len(self.images)

    def _load_csv(self):
        name = self.model + '_X' + str(self.scale) + '.csv'
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
    opt = {'data_path': '../data/DF2K', 'patch_size': 24}
    d = DF2K(opt)
    a = next(iter(d))

    print(a)

import os

from torch.utils.data import Dataset

from utils.data import defaultTransforms


class Set5(Dataset):

    def __init__(self, opt, model='test', tf=None):
        super(Set5, self).__init__()
        assert model in ['test'], f'Model {model} is not support!'
        self.opt = opt
        self.model = model
        self.scale = 'X' + str(opt['scale'])
        if tf is None:
            self.tf = defaultTransforms()
        else:
            self.tf = tf
        self.path = opt['data_path']
        self.x_path = os.path.join(self.path, self.scale)
        self.y_path = os.path.join(self.path, 'HR')
        self.images = sorted(os.listdir(self.y_path))
        self.x = sorted(os.listdir(self.x_path))

        assert len(self.images) == len(self.x), '数据长度不一致!'

    def __getitem__(self, index):
        hr = os.path.join(self.y_path, self.images[index])
        lr = os.path.join(self.x_path, self.x[index])
        name = self.images[index].split('.')[0]

        hr = self.tf(hr)
        lr = self.tf(lr)

        return {'LR': lr, 'HR': hr, 'name': name}

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    config = {'data_path': r'../data/Set5', 'scale': 2}
    data = Set5(config)
    print(data.images)
    print(data.x)

    item = next(iter(data))
    print(item)

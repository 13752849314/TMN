from .Set5 import Set5
from .DIV2K import DIV2K
from .DF2K import DF2K


def getdataSet(name, opt, model, **kwargs):
    tf = kwargs.get('tf', None)
    if name == 'Set5':
        data = Set5(opt)
        return data
    elif name == 'DIV2K':
        data = DIV2K(opt, model, scale='X' + str(opt['scale']), sub=opt['sub'], lr_type=opt['lr_type'], tf=tf)
        return data
    elif name == 'Set14':
        data = Set5(opt)
        return data
    elif name == 'BSDS100':
        data = Set5(opt)
        return data
    elif name == 'Manga109':
        data = Set5(opt)
        return data
    elif name == 'Urban100':
        data = Set5(opt)
        return data
    elif name == 'DF2K':
        data = DF2K(opt, opt['scale'], model, tf)
        return data
    else:
        raise ValueError(f'Dataset {name} is not exist!')

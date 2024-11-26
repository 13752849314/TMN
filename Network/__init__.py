from .MTN import MTN


def getNetwork(opt):
    in_ch = opt['in_ch']
    out_ch = opt['out_ch']
    scale = opt['scale']
    if opt['model'] == 'MTN':
        first_filters = opt['first_filters']
        filters = opt['filters']
        ratio = opt['ratio']
        drop = opt['drop']
        model = MTN(in_ch, out_ch, scale, first_filters=first_filters, filters=filters, ratio=ratio, drop=drop)
        return model

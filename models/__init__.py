from .RPCANet import *


def get_model(name, net=None):
    if name == 'rpcanet':
        net = RPCANet(stage_num=6)
    else:
        raise NotImplementedError

    return net


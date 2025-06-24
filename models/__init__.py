from .deepunfolding import *

def get_model(name, net=None):
    if name == 'rpcanet':
        net = RPCANet9(stage_num=6, slayers=6, llayers=3, mlayers=3, channel=32)

    elif name == 'rpcanet_pp':
        net = RPCANet_LSTM(stage_num=6, slayers=6, mlayers=3, channel=32)

    elif name == 'rpcanet_pp_s3':
        net = RPCANet_LSTM(stage_num=3, slayers=6, mlayers=3, channel=32)

    elif name == 'rpcanet_pp_s9':
        net = RPCANet_LSTM(stage_num=9, slayers=6, mlayers=3, channel=32)
    else:
        raise NotImplementedError

    return net


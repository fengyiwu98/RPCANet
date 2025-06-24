import os
os.environ['CUDA_VISIBLE_DEVICE'] ='0'
gpu = '--gpu 0'

train_cfg = ' --epoch 800 --lr 1e-4 --batch-size 8 --save-iter-step 100 --log-per-iter 10'

train_cfg_sirstaug = ' --epoch 400 --lr 1e-4 --batch-size 8 --save-iter-step 100 --log-per-iter 10'

train_cfg_VS = ' --epoch 400 --lr 5e-4 --batch-size 4 --save-iter-step 100 --log-per-iter 10'

train_cfg_neu = ' --epoch 200 --lr 1e-4 --save-iter-step 100 --batch-size 8 --log-per-iter 10 --base-size 200 --crop-size 200'

train_cfg_saliency = ' --epoch 400 --lr 1e-4 --save-iter-step 100 --batch-size 8 --log-per-iter 10 --base-size 200 --crop-size 200'

data_sirstaug = ' --dataset sirstaug '
data_irstd1k = ' --dataset irstd1k '
data_nudt = ' --dataset nudt '
data_sirst = ' --dataset sirst '

data_drive = ' --dataset drive '
data_stare = ' --dataset stare '
data_CHASEDB1 = ' --dataset CHASEDB1 '

data_neu = ' --dataset neu '
data_saliency = ' --dataset saliency '

for i in range(1):
    os.system(
        'python train.py --net-name rpcanet_pp' + train_cfg + data_sirst + gpu
    )
    # os.system(
    #     'python train.py --net-name rpcanet_pp' + train_cfg_sirstaug + data_sirstaug + gpu
    # )
    # os.system(
    #     'python train.py --net-name rpcanet_pp' + train_cfg_VS + data_stare + gpu
    # )
    # os.system(
    #     'python train_dd.py --net-name rpcanet_pp' + train_cfg_neu + data_neu + gpu
    # )
    # os.system(
    #     'python train_dd.py --net-name rpcanet_pp' + train_cfg_saliency + data_saliency + gpu
    # )



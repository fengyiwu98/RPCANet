import os
os.environ['CUDA_VISIBLE_DEVICE'] ='0'
gpu = '--gpu 0'

train_cfg = ' --epoch 400 --lr 1e-4 --save-iter-step 100 --log-per-iter 10 '
data_sirstaug = ' --dataset sirstaug '
data_irstd1k = ' --dataset irstd1k '
data_nudt = ' --dataset nudt '


# for i in range(3):
#     print('Train Index: ', i)
#     os.system('python train.py --net-name fpn_1' + train_cfg + data_mdfa + gpu)
#
# for i in range(3):
#     print('Train Index: ', i)
#     os.system('python train.py --net-name fpn_1' + train_cfg + data_sirstaug + gpu)

#for i in range(3):
#    os.system('python train.py --net-name rnet --batch-size 8  --rank 8' + train_cfg + data_irstd1k + gpu)

# for i in range(2):
#     os.system('python train.py --net-name rpcanet9 --batch-size 8' + train_cfg + data_irstd1k + gpu)
for i in range(1):
    os.system('python train.py --net-name rpcanet --batch-size 8' + train_cfg + data_sirstaug + gpu)
# os.system('sleep 300')
# os.system('sh ~/shutdown.sh')



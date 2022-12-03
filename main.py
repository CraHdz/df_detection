import os 
import util
from log.log import logger
from data_manager import VideoProcess
import torch.utils.data as data
from data_manager import Meta_Train_Dataset
from network.meta_learning import MetaLearner
import shutil
import torch
from torch import nn
import torch.nn.functional as F


import torch.distributed as dist
import torch.utils.data.distributed
from torch.multiprocessing import Process

from torch.utils.tensorboard import SummaryWriter
import sys

def start(config, rank=None):
    logger(config.log_dir)

    device = torch.device(config.device)

    if config.mode == "train":
        epoches = config.epoches
        model_save_path = config.model_save_path

        train_config = config.train
        meta_train_loaders = get_data_loader(train_config.meta_train_dataset, train_config.train_batch_size, "train", rank)
        meta_test_loaders = get_data_loader(train_config.meta_test_dataset, train_config.test_batch_size, "test", rank)

        meta_learner = MetaLearner(config.metalearning, meta_train_loaders, meta_test_loaders, device)

        print(get_parameter_number(meta_learner))
        writer = SummaryWriter(config.tensorb_log)
        if rank is not None:
            dist.init_process_group("gloo", rank=rank, world_size=3)
            torch.cuda.set_device(rank)
            meta_learner = torch.nn.parallel.DistributedDataParallel(meta_learner, device_ids=[rank])

        meta_learner = meta_learner.to(device)
        print("start training")
        loss_iter, loss_train, loss_test = 0, 0, 0
        for i in range(epoches):
            loss_meta_iter, loss_meta_train, loss_meta_test = meta_learner()
            logger.info("loss_meta_iter:" + str(loss_meta_iter) + ", loss_meta_train:" + str(loss_meta_train) + ", loss_meta_test:" + str(loss_meta_test))
            loss_iter = loss_iter + loss_meta_iter
            loss_train = loss_train + loss_meta_train
            loss_test = loss_test + loss_meta_test
            if i % 10 == 0:
                writer.add_scalar('loss/loss_meta_iter', loss_iter, i / 5)
                writer.add_scalar('loss/loss_meta_train', loss_train, i / 5)
                writer.add_scalar('loss/loss_meta_test', loss_test, i / 5)
                loss_iter, loss_train, loss_test = 0, 0 , 0

            if i % 100 == 0:
                model_save_path = os.path.join(model_save_path, "ckp_{}.pth".format(str(i)))
                meta_learner.save_model() 
    
    elif config.mode == "test":
        pass 
    elif config.mode == "process":
        vid_process = VideoProcess(config.video_process)
        vid_process.process()


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# class MLP(nn.Module):
#     '''
#     网络里面主要是要定义__init__()、forward()
#     '''
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(5,10)
#         self.out = nn.Linear(10,2)
     
#     def forward(self,x):
#         return self.out(F.relu(self.hidden(x)))

# def mksure_data(dataset_path_list):
#     for dataset_path in dataset_path_list:
#         vids_pkg = sorted(os.listdir(dataset_path))
#         print(len(vids_pkg))
#         for vid_pkg in vids_pkg:
#             vid_pkg_path = os.path.join(dataset_path, vid_pkg)
#             vid_imgs = os.listdir(vid_pkg_path)
#             if len(vid_imgs) != 32:
#                 logger.error(vid_pkg_path + " is not have enghout image!!!")
#                 shutil.rmtree(vid_pkg_path)

# def prepare(config):
#     pass

def get_data_loader(dataset_path_list, batch_size, mode, rank):
    loaders_list = []
    for dataset_path in dataset_path_list:
        data_set = Meta_Train_Dataset(dataset_path, mode)
        if rank is None:
            sampler = None
        else: 
            sampler = torch.utils.data.distributed.DistributedSampler(data_set)
        
        data_loader = data.DataLoader(
            data_set, 
            batch_size=batch_size, 
            num_workers=4,
            shuffle=False,
            drop_last=True,
            sampler=sampler
        )
        loaders_list.append(data_loader)                                              
    return loaders_list

def main():
    config_Path = './configs/config.yaml'
    config = util.get_config(config_Path)

    if config.is_DDP:

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12855'

        size = 2
        processes = []
        for rank in range(size):
            p = Process(target=start, args=(config, rank))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    else:
        start(config)

if __name__ == "__main__":
    main()
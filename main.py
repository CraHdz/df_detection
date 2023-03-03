import os 
import util
from log.log import logger
from data_manager import VideoProcess
import torch.utils.data as data
from data_manager import Meta_Train_Dataset
from network.meta_learning import MetaLearner
from network.MSTX_net import MSTX
from data_manager import Nor_Dataset
import shutil
import torch
from torch import nn
import torch.nn.functional as F


import torch.distributed as dist
import torch.utils.data.distributed
from torch.multiprocessing import Process

from torch.utils.tensorboard import SummaryWriter

def start(config, rank=None):
    print("init logger")
    logger(config.log_dir)

    device = torch.device(config.device)

    if config.mode == "train":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        epoches = config.epoches
        model_save_path = config.model_save_path

        train_config = config.train

        if rank is not None:
            dist.init_process_group("gloo", rank=rank, world_size=2)
            torch.cuda.set_device(rank)
        
        print("init loader and net")
        meta_train_loaders = get_data_loader(train_config.meta_train_dataset, train_config.train_batch_size, "train", config.vids_pkg_size, rank)
        meta_test_loaders = get_data_loader(train_config.meta_test_dataset, train_config.test_batch_size, "test", config.vids_pkg_size, rank)
        meta_learner = MetaLearner(config.metalearning, meta_train_loaders, meta_test_loaders, device, epoches)

        print("init tensorboard")
        writer = SummaryWriter(config.tensorb_log)
        if rank is not None:
            meta_learner = torch.nn.parallel.DistributedDataParallel(meta_learner, device_ids=[rank])

        meta_learner = meta_learner.to(device)
        print("start training")
        loss_iter, loss_train, loss_test = 0, 0, 0
        for epoch in range(epoches):
            loss_meta_iter, loss_meta_train, loss_meta_test = meta_learner(epoch)
            logger.info("loss_meta_iter:" + str(loss_meta_iter) + ", loss_meta_train:" + str(loss_meta_train) + ", loss_meta_test:" + str(loss_meta_test))
            loss_iter = loss_iter + loss_meta_iter
            loss_train = loss_train + loss_meta_train
            loss_test = loss_test + loss_meta_test

            if epoch % 10 == 0 and epoch != 0:
                writer.add_scalar('loss/loss_meta_iter', loss_iter, epoch / 10)
                writer.add_scalar('loss/loss_meta_train', loss_train, epoch / 10)
                writer.add_scalar('loss/loss_meta_test', loss_test, epoch / 10)
                loss_iter, loss_train, loss_test = 0, 0, 0

            if epoch % 500 == 0 and epoch != 0:
                ckp_path = os.path.join(model_save_path, "ckp_{}.pth".format(str(epoch)))
                meta_learner.save_model(ckp_path) 
    
    elif config.mode == "test":
        print("test start")
        model = MSTX(config.metalearning.MSTX)

        model.load_state_dict(torch.load(config.ckp_file))
        model.to(device)
        model.eval()
        
        data_set = Nor_Dataset(config.test.test_dataset, "test")

        
        data_loader = data.DataLoader(
            data_set, 
            batch_size=1, 
            num_workers=0,
            shuffle=False,
            drop_last=True,

        )

        n_classes = 2
        acc_num = 0
        target_num = torch.zeros((1, n_classes)) # n_classes为分类任务类别数量
        predict_num = torch.zeros((1, n_classes))
        acc_num = torch.zeros((1, n_classes))
        with torch.no_grad():
            for index, (frams, label) in enumerate(data_loader):
                frams = frams.to(device)
                label = label.to(device)
                outputs = model(frams)
                print(outputs)
                loss = F.cross_entropy(outputs, label)

                _, predicted = outputs.max(1)
                if predicted.item() != label.item():
                    print("index" + str(index) + "  : " + str(predicted.item()) + ":" + str(label.item())) 
                pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
                tar_mask = torch.zeros(outputs.size()).scatter_(1, label.data.cpu().view(-1, 1), 1.)
                target_num += tar_mask.sum(0)  # 得到数据中每类的数量
                acc_mask = pre_mask * tar_mask 
                acc_num += acc_mask.sum(0) # 得到各类别分类正确的样本数量

            print("test finish")
            recall = acc_num / target_num
            precision = acc_num / predict_num
            F1 = 2 * recall * precision / (recall + precision)
            accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
            print('Test Acc {}, recal {}, precision {}, F1-score {}'.format(accuracy, recall, precision, F1))

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

def mksure_data(dataset_path_list):
    for dataset_path in dataset_path_list:
        vids_pkg = sorted(os.listdir(dataset_path))
        print(len(vids_pkg))
        for vid_pkg in vids_pkg:
            vid_pkg_path = os.path.join(dataset_path, vid_pkg)
            vid_imgs = os.listdir(vid_pkg_path)
            if len(vid_imgs) != 32:
                logger.error(vid_pkg_path + " is not have enghout image!!!")
                shutil.rmtree(vid_pkg_path)

# def prepare(config):
#     pass

def get_data_loader(dataset_path_list, batch_size, mode, vids_pkg_size, rank):
    loaders_list = []
    for dataset_path in dataset_path_list:
        data_set = Meta_Train_Dataset(dataset_path, mode, vids_pkg_size)
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
        os.environ['MASTER_PORT'] = '18888'

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
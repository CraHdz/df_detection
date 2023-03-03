import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from MSTX_net import MSTX
import random
from copy import deepcopy
import sys
class MetaLearner(nn.Module):
    def __init__(self, config, meta_train_loaders, meta_test_loaders, device, epoches):
        super(MetaLearner, self).__init__()
       
        self.device = device
        self.epoches = epoches

        self.meta_train_loaders = meta_train_loaders
        self.meta_test_loaders = meta_test_loaders

        self.meta_train_loaders_iter = [iter(loader) for loader in self.meta_train_loaders]
        self.meta_test_loaders_iter = [iter(loader) for loader in self.meta_test_loaders]

        self.train_loader_len = len(self.meta_train_loaders_iter)
        self.test_loader_len = len(self.meta_test_loaders_iter)

        self.meta_train_batch_size = config.meta_train_batch_size
        self.meta_test_batch_size = config.meta_test_batch_size
        
        self.net = MSTX(config.MSTX)
        self.net = self.net.to(self.device)

        self.meta_train_lr = config.meta_train_lr
        self.meta_train_momentum = config.meta_train_momentum
        self.meta_train_weight_decay = config.meta_train_weight_decay

        self.alpha =config.alpha

        self.meta_lr = config.meta_lr
        self.meta_momentum = config.meta_momentum
        self.meta_dampening = config.meta_dampening

        self.meta_bufs = []
        # self.train_optim = torch.optim.SGD(self.net.parameters(), lr=self.meta_train_lr, momentum=self.meta_train_momentum, weight_decay=self.meta_train_weight_decay)
        self.train_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_train_lr)
        self.train_scheduler = torch.optim.lr_scheduler.StepLR(self.train_optim, step_size=100, gamma=0.5)

    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path))

    def save_model(self, model_path):
        torch.save(self.net.state_dict(), model_path)

    def gen_rand_num(self, mask_num, high, low=1):
        rand_num = mask_num
        while rand_num == mask_num:
            rand_num = random.randint(low, high)
        return rand_num

    def get_data(self, index, data_loaders, data_loaders_iter): 
        loader_iter = data_loaders_iter[index]
        try:
            input, label = next(loader_iter)
        except StopIteration:
            data_loaders_iter[index] = iter(data_loaders[index])
            input, label = next(data_loaders_iter[index])
        return input, label
    
    def SGD(self, paras, grads, lr, is_meta, meta_momentum=None, meta_dampening=None):
        for index, (para, grad) in enumerate(zip(paras, grads)):
            grad = grad.data
            if is_meta:
                if index > (len(self.meta_bufs) - 1):
                    meta_buf = torch.clone(grad).detach_()
                    self.meta_bufs.append(meta_buf)
                else:
                    meta_buf = self.meta_bufs[index]
                    if meta_momentum is not None and meta_dampening is not None:
                        meta_buf.mul_(meta_momentum).add_(grad, alpha=1-meta_dampening)
                        self.meta_bufs[index] = meta_buf
            else:
                meta_buf = grad
            para.data.add_(meta_buf, alpha=-lr)
        return paras

    def forward(self, epoches):

        torch.cuda.empty_cache()
        # if epoches != 0 and epoches // 100 == 0:
        #     self.meta_lr = self.meta_lr * 0.5
        #     self.meta_train_lr = self.meta_train_lr * 0.8


        # with torch.no_grad():
        #     origin_paras = deepcopy(list(self.net.parameters()))

        # loss_meta_iter = 0
        # meta_grad = None

        for train_domain_id in range(1, self.train_loader_len):
            #将源参数赋值给网络
            # for ori_para, net_para in zip(origin_paras, self.net.parameters()):
            #     net_para.data = ori_para.clone().detach_()
            
            iter_grad = None
            train_batch_nums = [self.gen_rand_num(train_domain_id, self.train_loader_len - 1) for i in range(int(self.meta_train_batch_size / 2))]
            train_batch_nums.extend([0 for i in range(int(self.meta_train_batch_size / 2))])
            test_batch_nums = [random.choice([0, train_domain_id, self.test_loader_len - 1]) for i in range(self.meta_test_batch_size)]

            #meta train
            scaler = torch.cuda.amp.GradScaler()
            loss_meta_train = 0

            for index in train_batch_nums:

                x, label = self.get_data(index, self.meta_train_loaders, self.meta_train_loaders_iter)
                x = x.to(self.device)
                label = label.to(self.device)
                self.train_optim.zero_grad()

                with torch.cuda.amp.autocast():
                    y = self.net(x)
                    loss = F.cross_entropy(y, label)
                    # print(y)
                    # print(label.item())
                    loss_meta_train = loss_meta_train + loss.item()
                
                # grad = torch.autograd.grad(loss, self.net.parameters(), retain_graph=False, allow_unused=True)

                scaler.scale(loss).backward()
                scaler.step(self.train_optim)
                scaler.update()
                

                # with torch.no_grad():
                #     self.SGD(self.net.parameters(), grad, self.meta_train_lr, False)

                    # if iter_grad is not None:
                    #     iter_grad = list(map(lambda p: p[0].add_(p[1].div_(self.meta_train_batch_size / 8.0)), zip(iter_grad, grad)))
                    # else :
                    #     iter_grad = [g.clone().detach_().div_(self.meta_train_batch_size / 8.0) for g in grad]
                # self.net.zero_grad()
                torch.cuda.empty_cache()
        self.train_scheduler.step()

        #     iter_grad = [g.mul_(self.alpha) for g in iter_grad]
        
        #     # meta test
        #     loss_meta_test = 0
        #     for index in test_batch_nums:
        #         x, label = self.get_data(index, self.meta_test_loaders, self.meta_test_loaders_iter)
        #         x = x.to(self.device)
        #         label = label.to(self.device)
        #         with torch.cuda.amp.autocast():
        #             y = self.net(x)
        #             loss = F.cross_entropy(y, label)
        #             loss_meta_test = loss_meta_test + loss.item()
                
        #         grad = torch.autograd.grad(loss, self.net.parameters(), retain_graph=False, allow_unused=True)

        #         with torch.no_grad():
        #             if iter_grad is not None:
        #                 iter_grad = list(map(lambda p: p[0].add_(p[1].div_(self.meta_test_batch_size / 4.0)), zip(iter_grad, grad)))
        #             else :
        #                 print("error: train grad is None")
        #                 sys.exit(1)
    
        #         self.net.zero_grad()
        #         torch.cuda.empty_cache()
            
        #     #计算任务的梯度并相加
        #     with torch.no_grad():
        #         if meta_grad is not None:
        #             meta_grad = list(map(lambda p: p[0].add_(p[1]), zip(meta_grad, iter_grad)))
        #         else:
        #             meta_grad = iter_grad

        #         loss_meta_iter = loss_meta_iter + loss_meta_train / self.meta_train_batch_size + loss_meta_test / self.meta_test_batch_size
        
        # # meta_grad = [meta_g.div_(self.train_loader_len - 1) for meta_g in meta_grad]

        # with torch.no_grad():
        #     meta_paras = self.SGD(origin_paras, meta_grad, self.meta_lr, True, self.meta_momentum, self.meta_dampening)

        #     for meta_para, net_para in zip(meta_paras, self.net.parameters()):
        #         net_para.data = meta_para
        
        # return loss_meta_iter, loss_meta_train, loss_meta_test
        return 0, loss_meta_train, 0

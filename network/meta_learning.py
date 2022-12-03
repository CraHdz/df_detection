import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from MSTX_net import MSTX
import random
from copy import deepcopy

class MetaLearner(nn.Module):
    def __init__(self, config, meta_train_loaders, meta_test_loaders, device,):
        super(MetaLearner, self).__init__()
       
        self.device = device

        self.meta_train_loaders = meta_train_loaders
        self.meta_test_loaders = meta_test_loaders

        self.meta_train_loaders_iter = [iter(loader) for loader in self.meta_train_loaders]
        self.meta_test_loaders_iter = [iter(loader) for loader in self.meta_test_loaders]

        self.train_loader_len = len(self.meta_train_loaders_iter)
        self.test_loader_len = len(self.meta_test_loaders_iter)

        self.meta_train_batch_size = config.meta_train_batch_size
        self.meta_test_batch_size = config.meta_test_batch_size
        
        self.net = MSTX(config.MSTX)
        # self.net = self.net.to(self.device)

        self.meta_train_lr = config.meta_train_lr
        self.meta_train_momentum = config.meta_train_momentum
        self.meta_train_weight_decay = config.meta_train_weight_decay

        self.alpha =config.alpha

        self.meta_lr = config.meta_lr
        self.meta_momentum = config.meta_momentum
        self.meta_dampening = config.meta_dampening
        # self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=config.meta_lr)

        self.meta_bufs = []

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, model_path):
        torch.save(self.model.module.state_dict(), model_path)
    def gen_rand_num(self, mask_num, high, low=0):
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
    
    def SGD(self, paras, grads):
        for index, (para, grad) in enumerate(zip(paras, grads)):
            if index > (len(self.meta_bufs) - 1):
                meta_buf = torch.clone(grad).detach()
                self.meta_bufs.append(meta_buf)
            else:
                meta_buf = self.meta_bufs[index]
                meta_buf.mul_(self.meta_momentum).add_(grad, alpha=1 - self.dampening)
            
            para.data.add_(meta_buf, alpha=-self.meta_lr)
        return paras

    def forward(self):
        
        origin_paras = deepcopy(self.net.parameters())

        meta_grad = None
        loss_meta_iter = 0
        for train_domain_id in range(1, self.train_loader_len):
  
            train_batch_nums = [self.gen_rand_num(train_domain_id, self.train_loader_len - 1) for i in range(self.meta_train_batch_size) ]
            test_batch_nums = [random.choice([0, train_domain_id, self.test_loader_len - 1]) for i in range(self.meta_test_batch_size) ]

            #meta train
            train_optim = torch.optim.SGD(self.net.parameters(), lr=self.meta_train_lr, momentum=self.meta_train_momentum, weight_decay=self.meta_train_weight_decay)
            scaler = torch.cuda.amp.GradScaler()
            train_grad = None
            loss_meta_train = 0

            for index in train_batch_nums:
                train_optim.zero_grad()

                x, label = self.get_data(index, self.meta_train_loaders, self.meta_train_loaders_iter)
                x = x.to(self.device)
                label = label.to(self.device)

                with torch.cuda.amp.autocast():
                    y = self.net(x)
                    loss = F.cross_entropy(y, label)
                    loss_meta_train = loss_meta_train + loss.item(0)
                
                grad = torch.autograd.grad(loss, self.net.parameters(), retain_graph=True)

                scaler.scale(loss).backward()
                scaler.step(train_optim)
                scaler.update()

                

                if train_grad is not None:
                    train_grad = list(map(lambda p: p[0] + p[1], zip(train_grad, grad)))
                else :
                    train_grad = grad

            
            #meta test
            test_grad = None
            loss_meta_test = 0
            for index in test_batch_nums:
                x, label = self.get_data(index, self.meta_test_loaders, self.meta_test_loaders_iter)
                x = x.to(self.device)
                label = label.to(self.device)
                
                with torch.cuda.amp.autocast():
                    y = self.net(x)
                    loss = F.cross_entropy(y, label)
                    loss_meta_test = loss_meta_test + loss.item(0)
                
                grad = torch.autograd.grad(loss, self.net.parameters())
                if test_grad is not None:
                    test_grad = list(map(lambda p: p[0] + p[1], zip(test_grad, grad)))
                else :
                    test_grad = grad
            
            #计算任务的梯度并相加
            if meta_grad is not None:
                mix_grad = list(map(lambda p: (p[0] * (1 - self.alpha) / self.meta_test_batch_size  + p[1] * self.alpha / self.meta_train_batch_size) / (self.train_loader_len - 1), zip(test_grad, train_grad)))
                meta_grad = list(map(lambda p: p[0] + p[1], zip(meta_grad, mix_grad)))
            else :
                meta_grad = list(map(lambda p: (p[0] * (1 - self.alpha) / self.meta_test_batch_size  + p[1] * self.alpha / self.meta_train_batch_size) / (self.train_loader_len - 1), zip(test_grad, train_grad)))

            loss_meta_iter = loss_meta_iter + loss_meta_train / self.meta_train_batch_size + loss_meta_test / self.meta_test_batch_size

        meta_paras = self.SGD(origin_paras, meta_grad)

        #将参数更新到网络
        for meta_para, net_para in zip(meta_paras, self.net.parameters()):
            net_para.data = meta_para

        return loss_meta_iter, loss_meta_train, loss_meta_test
            # loss = F.cross_entropy(y_hat, y_spt[i]) 
           
            
            # # fast_weights这一步相当于求了一个\theta - \alpha*\nabla(L)
            # fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
            # 在query集上测试，计算准确率
            # 这一步使用更新前的数据
        #     with torch.no_grad():
        #         y_hat = self.net()
        #         loss_qry = F.cross_entropy(y_hat, y_qry[i])
        #         loss_list_qry[0] += loss_qry
        #         pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
        #         correct = torch.eq(pred_qry, y_qry[i]).sum().item()
        #         correct_list[0] += correct
            
        #     # 使用更新后的数据在query集上测试。
        #     with torch.no_grad():
        #         y_hat = self.net(x_qry[i], fast_weights, bn_training = True)
        #         loss_qry = F.cross_entropy(y_hat, y_qry[i])
        #         loss_list_qry[1] += loss_qry
        #         pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
        #         correct = torch.eq(pred_qry, y_qry[i]).sum().item()
        #         correct_list[1] += correct   
            
        #     for k in range(1, self.update_step):
                
        #         y_hat = self.net(x_spt[i], params = fast_weights, bn_training=True)
        #         loss = F.cross_entropy(y_hat, y_spt[i])
        #         grad = torch.autograd.grad(loss, fast_weights)
        #         tuples = zip(grad, fast_weights) 
        #         fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
                    
        #         y_hat = self.net(x_qry[i], params = fast_weights, bn_training = True)
        #         loss_qry = F.cross_entropy(y_hat, y_qry[i])
        #         loss_list_qry[k+1] += loss_qry
                
        #         with torch.no_grad():
        #             pred_qry = F.softmax(y_hat,dim=1).argmax(dim=1)
        #             correct = torch.eq(pred_qry, y_qry[i]).sum().item()
        #             correct_list[k+1] += correct

                
        # loss_qry = loss_list_qry[-1] / task_num
        # self.meta_optim.zero_grad()
        # loss_qry.backward()
        # self.meta_optim.step()
        
        # accs = np.array(correct_list) / (query_size * task_num)
        # loss = np.array(loss_list_qry) / ( task_num)
        # return accs,loss

    
    
    # def finetunning(self, x_spt, y_spt, x_qry, y_qry):
    #     assert len(x_spt.shape) == 4
        
    #     query_size = x_qry.size(0)
    #     correct_list = [0 for _ in range(self.update_step_test + 1)]
        
    #     new_net = deepcopy(self.net)
    #     y_hat = new_net(x_spt)
    #     loss = F.cross_entropy(y_hat, y_spt)
    #     grad = torch.autograd.grad(loss, new_net.parameters())
    #     fast_weights = list(map(lambda p:p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))
        
    #     # 在query集上测试，计算准确率
    #     # 这一步使用更新前的数据
    #     with torch.no_grad():
    #         y_hat = new_net(x_qry,  params = new_net.parameters(), bn_training = True)
    #         pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
    #         correct = torch.eq(pred_qry, y_qry).sum().item()
    #         correct_list[0] += correct

    #     # 使用更新后的数据在query集上测试。
    #     with torch.no_grad():
    #         y_hat = new_net(x_qry, params = fast_weights, bn_training = True)
    #         pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
    #         correct = torch.eq(pred_qry, y_qry).sum().item()
    #         correct_list[1] += correct

    #     for k in range(1, self.update_step_test):
    #         y_hat = new_net(x_spt, params = fast_weights, bn_training=True)
    #         loss = F.cross_entropy(y_hat, y_spt)
    #         grad = torch.autograd.grad(loss, fast_weights)
    #         fast_weights = list(map(lambda p:p[1] - self.base_lr * p[0], zip(grad, fast_weights)))
            
    #         y_hat = new_net(x_qry, fast_weights, bn_training=True)
            
    #         with torch.no_grad():
    #             pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
    #             correct = torch.eq(pred_qry, y_qry).sum().item()
    #             correct_list[k+1] += correct
                
    #     del new_net
    #     accs = np.array(correct_list) / query_size
    #     return accs      

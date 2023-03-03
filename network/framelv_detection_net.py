from __future__ import print_function,  division,  absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from d2l import torch as d2l

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = Conv2d_cd(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = Conv2d_cd(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        # self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        # self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            # [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block,  self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides,  bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

#attention layer
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class AttentionBlock(nn.Module):
    def __init__(self, 
                key_size=512, 
                query_size=512, 
                value_size=512,
                num_hiddens=512,
                norm_shape=[512],
                ffn_num_input=512,
                ffn_num_hiddens=1024,
                num_heads=8,
                dropout=0.1,
                use_bias=False,
                **kwargs
        ):
        super(AttentionBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, key_values=None, valid_lens=None):
        if key_values == None:
            key_values = X
        Y = self.addnorm1(X, self.attention(X, key_values, key_values, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
    


class framelv_detection_net(nn.Module):

    def __init__(self, vids_size):
    
        super(framelv_detection_net, self).__init__()

        self.block1=Block(3, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2=Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3=Block(256, 512, 2, 2, start_with_relu=True, grow_first=True)
        self.block4=Block(512, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block5=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10=Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block11=Block(728,  512,  2,  2, start_with_relu=True, grow_first=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, 7, 1), 
            nn.BatchNorm2d(512)
        )

        self.attention1 = AttentionBlock()

        self.fc1 = nn.Sequential(
            nn.Linear(512, int(512 / vids_size)), 
            nn.ReLU(True)
        )
        self.attention2 = AttentionBlock()

        self.fc2 = nn.Sequential(
            nn.Linear(512, 128), 
            nn.ReLU(True)
        )

        self.fc3 = nn.Linear(128, 2)

        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():        
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) 		 
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)

    def get_key_value(self, x):
        x = x.clone().detach_()
        with torch.no_grad():
            sum_t = torch.sum(x, dim=2)
            indexs = torch.argmax(sum_t, dim=1)
            x = x[:, indexs, :]
        x.requires_grad = True
        return x

    def forward(self,  x):
        b, t, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = self.block11(out)

        out = self.conv1(out) 

        out = out.view(b, t, -1)

        key_value = self.get_key_value(out)
        out = self.attention1(out, key_value)
        out = self.fc1(out)

        out = out.view(b, 1, -1)
        out = self.attention2(out)

        out = out.view(b, -1)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def gram_matrix(y):
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)   #C和w*h转置
        gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t相乘
        return gram

if __name__ == "__main__":
    net = framelv_detection_net(16)
    # input = torch.randn(32, 24, 224, 224)
    # print(get_parameter_number(net))
    # net = framelv_detection_net()
    for i in range(2):
        input = torch.randn(1, 16, 3, 224, 224)
        result = net(input)
        print(result.shape)
    
    
    
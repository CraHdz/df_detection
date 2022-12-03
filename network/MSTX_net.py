import torch
from vidlv_detection_net import vidlv_detection_net
from feat_extra_net import feat_extra_net
from framelv_detection_net import framelv_detection_net
import torch.nn as nn

class MSTX(nn.Module):
    def __init__(self, config):
        super(MSTX, self).__init__()

        self.t_size = config.t_size


        self.vidlv_detection_net = vidlv_detection_net(config.vst_weight_path)
        self.feat_extra_net = feat_extra_net()
        self.framelv_detection_net = framelv_detection_net()
       
        self.fc = nn.Linear(4,  2)

        self.init_para()
        
    def init_para(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.feat_extra_net(x)

        x = x.view(b, t, c, 224, 224)

        frame_result = self.vidlv_detection_net(x.permute(0, 2, 1, 3, 4))
        video_result = self.framelv_detection_net(x)
        result = torch.cat((frame_result, video_result), dim=0)

        result = self.fc(result)
        return result
    
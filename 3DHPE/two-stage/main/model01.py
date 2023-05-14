import torch
import torch.nn as nn
from torch.nn import functional as F
# from backbone import *
from config import cfg
import os.path as osp
from backbone.pose2dnet import MobilePosNet
from loss import KLDiscretLoss
from thop import profile

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

        self.w3 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm3 = nn.BatchNorm1d(self.l_size)

        self.w4 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm4 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w3(y)
        y = self.batch_norm3(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w4(y)
        y = self.batch_norm4(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, linear_size=256, num_stage=4, p_dropout=0.5):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size = input_size  # 16 * 2
        # 3d joints
        self.output_size = output_size  # 16 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y

def get_pred(heatmaps):
    m = torch.argmax(heatmaps,dim=2,keepdim = True)
    return m

class CustomNet(nn.Module):
    def __init__(self, backbone, linear, joint_num):
        super(CustomNet, self).__init__()
        self.backbone = backbone
        self.linear = linear
        self.joint_num = joint_num

    def forward(self, input_img, target=None):
        hx, hy = self.backbone(input_img)
        joint_img = torch.cat((get_pred(hx), get_pred(hy)),2).flatten(1)
        # print(joint_img.shape)
        joint_cam = self.linear(joint_img.float()).reshape(-1, self.joint_num, 3)
        

        if target is None:
            return joint_cam, joint_img.reshape(-1, self.joint_num, 2)
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']
            target_cam = target['joint_cam']
            target_hx = target['hx']
            target_hy = target['hy']

            ## coordinate loss
            criteron = KLDiscretLoss()
            loss_hm = criteron(hx, hy, target_hx, target_hy, target_vis)
            loss_coord= torch.abs(joint_cam - target_cam) * target_vis
            loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2] * target_have_depth)/3.
            return loss_coord.mean(), loss_hm

def get_pose_net(backbone_str, is_train, joint_num):
    INPUT_SIZE = cfg.input_shape
    INTER_SIZE = (INPUT_SIZE[0]//16, INPUT_SIZE[1]//16)
    OUTPUT_SIZE = cfg.output_shape

    assert INPUT_SIZE == (256, 256)

    print("=" * 60)
    print("{} BackBone Generated".format(backbone_str))
    print("=" * 60)
    backbone = MobilePosNet(joint_num, INTER_SIZE, OUTPUT_SIZE)
    linear = LinearModel(joint_num * 2, joint_num * 3)
    model = CustomNet(backbone, linear, joint_num)
    if is_train == True:
        model.backbone.init_weights()
        model.linear.init_weights()
    return model

# if __name__ == "__main__":
#     model = get_pose_net('Gaint', True, 18)
#     dummy_input = torch.randn(1, 3, 256, 256)
#     flops, params = profile(model, (dummy_input,))
#     print('flops: ', flops, 'params: ', params)
#     print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))  
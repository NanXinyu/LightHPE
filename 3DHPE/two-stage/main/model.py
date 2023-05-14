import torch
import torch.nn as nn
from torch.nn import functional as F
# from backbone import *
from config import cfg
import os.path as osp
from backbone.pose2dnet import MobilePosNet
from semgcn import SemGraphConv
from semgcn import GraphNonLocal
from loss import KLDiscretLoss
from thop import profile
from utils.graphs_utils import adj_mx_from_skeleton
from torchsummary import summary
class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.nonlocal_ = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.nonlocal_(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class SemGCN(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj)

    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out


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
        joint_img = torch.cat((get_pred(hx), get_pred(hy)),2)#.flatten(1)
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
            loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2] ) * target_have_depth/3 #* target_have_depth)/3.
            
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
    adj = adj_mx_from_skeleton()
    semgcn = SemGCN(adj, 128, num_layers=4, p_dropout=0.0,
                       nodes_group= None)
    model = CustomNet(backbone, semgcn, joint_num)
    if is_train == True:
        model.backbone.init_weights()
        # model.linear.init_weights()
    return model

if __name__ == "__main__":
    model = get_pose_net('Gaint', True, 18).cuda()
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    summary(model, (3, 256, 256))
    adj = adj_mx_from_skeleton()  
    semgcn = SemGCN(adj.cuda(), 128, num_layers=4, p_dropout=0.0,
                       nodes_group= None).cuda()
    dummy_input = torch.randn(1, 18, 2).cuda()
    flops, params = profile(semgcn, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    summary(semgcn, (18, 2))  
import torch
from torch import nn
from torch.nn import functional as F

from models.HQS.network import extractors
from models.HQS.sync_batchnorm import SynchronizedBatchNorm2d
from models.HQS.network.aspp import ASPP_no4level

import time

def make_coord(shape, ranges=None, flatten=True, device=None): #
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device).float() # , 
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def compute_locations(h, w, stride, device): 
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device) 
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1) 
    shift_y = shift_y.reshape(-1)
    locations = torch.stack(( shift_x, shift_y), dim=1) + stride // 2 
    return locations 

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class CRMNet(nn.Module):
    def __init__(self, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.aspp_ = ASPP_no4level(backbone=backend, output_stride=8, BatchNorm=SynchronizedBatchNorm2d)
        self.imnet = MLP(in_dim=256+6, out_dim=1, hidden_list=[32, 32, 32, 32])

    def forward(self, x, seg, coord, cell, transferCoord=None, transferFeat=None, inter_s8=None, inter_s4=None):
        ####   
        torch.cuda.synchronize()
        start_r50_aspp_coord_time = torch.cuda.Event(enable_timing=True)
        end_r50_aspp_coord_time = torch.cuda.Event(enable_timing=True)
        start_r50_aspp_coord_time.record()
        
        if transferFeat is None:
            p = torch.cat((x, seg), 1)
            
            x1_feat, x2_feat, x3_feat = self.feats(p) # [6, 64, 112, 112] [6, 256, 56, 56] [6, 1024, 28, 28]
            feat = self.aspp_(x1_feat, x2_feat, x3_feat)

            feat_coord = make_coord(feat.shape[-2:], flatten=False, device=feat.device) # 
            # feat_coord = feat_coord.cuda()
            feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0)
            feat_coord = feat_coord.expand(feat.shape[0], 2, *feat.shape[-2:])
        else:
            feat = transferFeat
            feat_coord = transferCoord
        
        end_r50_aspp_coord_time.record()
        torch.cuda.synchronize()
        # print("######r50_aspp_coord_time:", start_r50_aspp_coord_time.elapsed_time(end_r50_aspp_coord_time))

        torch.cuda.synchronize()
        start_suppout_generate_time = torch.cuda.Event(enable_timing=True)
        end_suppout_generate_time = torch.cuda.Event(enable_timing=True)
        start_suppout_generate_time.record()

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rx = 2 / feat.shape[-2] / 2 # 48  0.020833333333333332
        ry = 2 / feat.shape[-1] / 2 # 48  0.020833333333333332

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                torch.cuda.synchronize()
                start_gridSample_feat_time = torch.cuda.Event(enable_timing=True)
                end_gridSample_feat_time = torch.cuda.Event(enable_timing=True)
                start_gridSample_feat_time.record()

                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord, coord], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

                end_gridSample_feat_time.record()
                torch.cuda.synchronize()
                # print("######gridSample_feat_time:", start_gridSample_feat_time.elapsed_time(end_gridSample_feat_time))

                torch.cuda.synchronize()
                start_imnet_time = torch.cuda.Event(enable_timing=True)
                end_imnet_time = torch.cuda.Event(enable_timing=True)
                start_imnet_time.record()

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

                end_imnet_time.record()
                torch.cuda.synchronize()
                # print("######imnet_time:", start_imnet_time.elapsed_time(end_imnet_time))
        
        end_suppout_generate_time.record()
        torch.cuda.synchronize()
        # print("######suppout_generate_time:", start_suppout_generate_time.elapsed_time(end_suppout_generate_time))

        torch.cuda.synchronize()
        start_weightedArea_time = torch.cuda.Event(enable_timing=True)
        end_weightedArea_time = torch.cuda.Event(enable_timing=True)
        start_weightedArea_time.record()

        tot_area = torch.stack(areas).sum(dim=0)
        # if self.local_ensemble:
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0

        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        
        pred_224 = torch.sigmoid(ret) # [6, 1, 224, 224]

        images = {}
        images['out_224'] = ret
        images['pred_224'] = pred_224
        
        # end_weightedArea_time.record()
        # torch.cuda.synchronize()
        # print("######weightedArea_time:", start_weightedArea_time.elapsed_time(end_weightedArea_time))
        if transferFeat is None:
            return images, feat_coord, feat
        else:
            return images        

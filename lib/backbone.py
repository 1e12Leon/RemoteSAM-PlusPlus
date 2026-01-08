import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from .cross_scale_interaction import CIM
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .mmcv_custom import load_checkpoint
# from mmseg.utils import get_root_logger
from typing import Optional
from torch import Tensor
from .various_receptive import VariousReceptive
import math
# from .logger import create_logger
import os
import sys
import logging
import functools
from termcolor import colored

from mmcv.cnn import build_norm_layer, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from torch.distributions.normal import Normal



@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


class Conv3x3_FFN(nn.Module):
    def __init__(self, 
                 embed_dims=256,
                 feedforward_channels=1024,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,):
        super().__init__() 
        self.embed_dims = embed_dims
        self.k = 3
        self.feedforward_channels = feedforward_channels 
 
        conv1 = [nn.Conv2d(embed_dims, feedforward_channels, self.k),
                    build_activation_layer(act_cfg), 
                    nn.Dropout(ffn_drop),]
        self.conv1 = nn.Sequential(*conv1)
        ffn = [Linear(feedforward_channels, embed_dims),nn.Dropout(ffn_drop)]
        self.ffn = nn.Sequential(*ffn)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

        self.gamma2 = nn.Identity()

    def forward(self, x, identity=None):
        # x expected to be (B, C, H, W) for Conv2d, but inputs might be flattened or different shape
        # The SparseDispatcher below reshapes inputs to (N, k, k, C) and permutes to (N, C, k, k)
        out = self.conv1(x).squeeze(-1).squeeze(-1) 
        out = self.ffn(out)
        out = self.gamma2(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x[:,:,(self.k-1)//2,(self.k-1)//2] 
        return identity + self.dropout_layer(out)

class Conv5x5_FFN(Conv3x3_FFN):
    def __init__(self, 
                 embed_dims=256,
                 feedforward_channels=1024,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,):
        super().__init__(embed_dims,
                 feedforward_channels,
                 act_cfg,
                 ffn_drop,
                 dropout_layer,
                 add_identity) 
        self.k = 5
        conv1 = [nn.Conv2d(embed_dims, feedforward_channels, self.k),
                    build_activation_layer(act_cfg), 
                    nn.Dropout(ffn_drop),]
        self.conv1 = nn.Sequential(*conv1)


class CosineTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, init_t=0.5):
        super(CosineTopKGate, self).__init__()
        proj_dim = min(model_dim//2, 256)
        # proj_dim=256
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = torch.nn.Linear(model_dim, proj_dim)
        self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x):
        cosine_projector = self.cosine_projector
        sim_matrix = self.sim_matrix
        logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
                              F.normalize(sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits

class MoE_layer(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts."""

    def __init__(self, moe_cfg):
        super(MoE_layer, self).__init__() 

        self.noisy_gating = moe_cfg['noisy_gating']
        self.input_size = moe_cfg['embed_dims']
        self.k = moe_cfg['top_k']
        self.gating = moe_cfg['gating']
        # instantiate experts
        self.num_experts = moe_cfg['num_experts']
        self.squad_num = None
        self.squads = None

        if 'squad_num' in moe_cfg.keys() and moe_cfg['squad_num'] > 0:
            self.squad_num = moe_cfg['squad_num']
            self.squads = moe_cfg['squads']
            self.num_experts = self.squad_num * len(self.squads)

            self.experts  = nn.ModuleList([])
            for squad_idx in range(self.squad_num):
                for each in self.squads:
                    # Using eval to instantiate class from string name
                    if each == 'FFN':
                        net = FFN
                    elif each == 'Conv3x3_FFN':
                        net = Conv3x3_FFN
                    elif each == 'Conv5x5_FFN':
                        net = Conv5x5_FFN
                    else:
                        raise ValueError(f"Unknown expert type: {each}")
                    
                    if each == 'FFN':
                        self.experts.append(net(
                            embed_dims=self.input_size,
                            feedforward_channels=moe_cfg['feedforward_channels'],
                            num_fcs=2,
                            ffn_drop=moe_cfg['drop_rate'],
                            dropout_layer=dict(type='DropPath', drop_prob=moe_cfg['drop_path_rate']),
                            act_cfg=moe_cfg['act_cfg'],
                            add_identity=True,
                            init_cfg=None))
                    else:
                        self.experts.append(net(
                            embed_dims=self.input_size,
                            feedforward_channels=moe_cfg['feedforward_channels'], 
                            ffn_drop=moe_cfg['drop_rate'],
                            dropout_layer=dict(type='DropPath', drop_prob=moe_cfg['drop_path_rate']),
                            act_cfg=moe_cfg['act_cfg'],
                            add_identity=True,
                        ))           
            
        else:
            self.experts = nn.ModuleList([FFN(
                embed_dims=self.input_size,
                feedforward_channels=moe_cfg['feedforward_channels'],
                num_fcs=2,
                ffn_drop=moe_cfg['drop_rate'],
                dropout_layer=dict(type='DropPath', drop_prob=moe_cfg['drop_path_rate']),
                act_cfg=moe_cfg['act_cfg'],
                add_identity=True,
                init_cfg=None) for i in range(self.num_experts)]) 
        self.infer_expert = None
 
        if moe_cfg['gating'] == 'linear':
            self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
        elif moe_cfg['gating'] == 'cosine':
            self.w_gate = CosineTopKGate(self.input_size, self.num_experts)
        self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        if len(x.shape) == 2:
            x = x.sum(dim=0)
        return x.float().var() / (x.float().mean()**2 + eps)


    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten() # (bs x m)
        threshold_positions_if_in = torch.arange(batch) * m + self.k # bs
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in.to(top_values_flat.device)), 1)

        if len(noisy_values.shape) == 3:
            threshold_if_in = threshold_if_in.unsqueeze(1)

        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out.to(top_values_flat.device)), 1)
        if len(noisy_values.shape) == 3:
            threshold_if_out = threshold_if_out.unsqueeze(1)

        # is each value currently in the top k.

        normal = Normal(self.mean.to(noise_stddev.device), self.std.to(noise_stddev.device))
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    
    def random_k_gating(self, features, train):
        if train:
            idx = torch.randint(0, self.num_experts, 1)
            results = self.experts[idx](features)

        else:
            results = []
            for i in range(self.num_experts):
                tmp = self.num_experts[i](features)
                results.append(tmp)
            
            results = torch.stack(results, dim=0).mean(dim=0)

        return results

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        if self.gating == 'linear':
            clean_logits = x @ self.w_gate
        elif self.gating == 'cosine':
            clean_logits = self.w_gate(x)

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim= -1)  
        
        top_k_logits = top_logits[:, :self.k] if len(top_logits.shape) == 2 else top_logits[:, :, :self.k]     
        top_k_indices = top_indices[:, :self.k] if len(top_indices.shape) == 2 else top_indices[:, :, :self.k]
        
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)  

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


    def forward(self, x, identity=None, loss_coef=1e-2, hwshape=None):
        train = self.training
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0) 
            identity = identity.unsqueeze(0)
        x_shape = x.shape
        x = x.reshape(-1,x.shape[-1])
        identity = identity.reshape(-1,identity.shape[-1]) 
        gates, load = self.noisy_top_k_gating(x, train) 
        # calculate importance loss
        importance = gates.sum(dim=0)
        
        # calculate loss
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates, self.squads ) 
        expert_inputs, identity = dispatcher.dispatch(x,identity, shape=[x_shape[0], *hwshape, x_shape[-1]]) 
        gates = dispatcher.expert_to_gates() 
        expert_outputs = [self.experts[i](expert_inputs[i], identity[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs).reshape(x_shape)

        return y, loss
      
class SparseDispatcher(object):
    def __init__(self, num_experts, gates, squads=None):
        self.squads = None
        if squads:
            assert num_experts%len(squads)==0
            self.squads = squads*(num_experts//len(squads))
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)  # torch.nonzero: 
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp, identity=None, shape=None):
        
        if identity is not None:
            identity = identity[self._batch_index].squeeze(1)
        inp_exp = inp[self._batch_index].squeeze(1)

        if self.squads is None:
            return torch.split(inp_exp, self._part_sizes, dim=0), torch.split(identity, self._part_sizes, dim=0)
        
        else:
            split_idx = torch.split(self._batch_index, self._part_sizes, dim=0)
            splits = []
            for i, sq in enumerate(self.squads):
                if sq == 'FFN':
                    splits.append(inp[split_idx[i]])
                else:
                    k = 3 if sq == 'Conv3x3_FFN' else 5
                    inp_ = torch.cat([inp, torch.zeros(1,shape[-1], device=inp.device)], dim=0)
                    b,h,w,d = shape
                    x_idx = torch.tensor(range(b*h*w), device=inp.device).reshape(b, h, w).unsqueeze(-1)

                    def get_kernel_feat(ind, k=3):
                        i, j = ind %(h*w)// w, ind % w
                        pad_margin = tuple((k-1)//2 for i in range(4)) 
                        padded_x = torch.nn.functional.pad(x_idx, (0, 0, *pad_margin), "constant", -1).squeeze(-1)  # pad with 1 zero on each side
                        neighbor_features = padded_x[ind//(h*w), i: i+k, j:j+k] 
                        return neighbor_features 
            
                    result = torch.stack([get_kernel_feat(x,k) for x in split_idx[i]], dim=0)
                    result_ = result.flatten() 
                    ys = inp_[result_].reshape(len(result),k,k,shape[-1]).permute(0,3,1,2).contiguous()
                    splits.append(ys)

            return splits, torch.split(identity, self._part_sizes, dim=0)
                

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            if len(stitched.shape) == 3:
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(1))
            else:
                stitched = stitched.mul(self._nonzero_gates)

        if len(stitched.shape) == 3:
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(-1), requires_grad=True, device=stitched.device)
        else:
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined


    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)



class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # cat op
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, moe_cfg=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # --- MoE Modification ---
        self.moe_cfg = moe_cfg
        if moe_cfg is not None:
            # Update moe_cfg with layer specific details
            moe_cfg_copy = moe_cfg.copy()
            moe_cfg_copy.update({
                'embed_dims': dim,
                'feedforward_channels': mlp_hidden_dim,
                'drop_rate': drop,
                'drop_path_rate': drop_path, # passing float, Dict handled in MoE_layer
                'act_cfg': dict(type='GELU')
            })
            self.mlp = MoE_layer(moe_cfg_copy)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # ------------------------

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN feed-forward network
        x = shortcut + self.drop_path(x)

        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # --- MoE Logic ---
        loss = 0.0
        norm_x = self.norm2(x)
        if self.moe_cfg is not None:
            x, loss = self.mlp(norm_x, identity=x, hwshape=(H, W))
        else:
            x = x + self.drop_path(self.mlp(norm_x))
        # -----------------

        return x, loss


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x



class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, fv1:NestedTensor):
        x = fv1.tensors
        mask = fv1.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32,device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MultiModalSwinTransformer(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.0,
                 # --- MoE args ---
                 moe_block_indexes = [[],[],[],[]],
                 noisy_gating=True,
                 num_experts=8,
                 gate='cosine',
                 top_k=2,
                 squads=None,
                 squad_num=0
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
       
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MMBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                num_heads_fusion=num_heads_fusion[i_layer],
                fusion_drop=fusion_drop,
                # --- MoE args ---
                moe_block_ind=moe_block_indexes[i_layer],
                noisy_gating=noisy_gating,
                num_experts=num_experts,
                gate=gate,
                top_k=top_k,
                squads=squads,
                squad_num=squad_num
                # ----------------
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # cross-scale interaction module
        self.cim = CIM(dim=sum(num_features))
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x, l, l_mask):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        gate_losses = [] # Store MoE losses
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww, layer_aux_loss = layer(x, Wh, Ww, l, l_mask)
            
            if layer_aux_loss is not None:
                gate_losses.append(layer_aux_loss)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)  # output of a Block has shape (B, H*W, dim)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        outs = self.cim(outs)

        # Calculate mean MoE loss
        if len(gate_losses) > 0:
            moe_loss = sum(gate_losses) / len(gate_losses)
        else:
            moe_loss = torch.tensor(0.0, device=x.device)

        return tuple(outs), moe_loss
    
    def swin_converter(self, ckpt):
        """
        将标准 Swin Transformer 的权重转换为 MoE 权重 (Warm Start)。
        核心逻辑：
        1. 修复 Downsample 层的权重形状 (兼容官方 Swin)。
        2. 将原本的 mlp 权重复制给 moe_layer 中的每一个 expert。
        """
        from collections import OrderedDict
        import re
        import torch
        import torch.nn as nn

        new_ckpt = OrderedDict()

        # --- 辅助函数：如果是 ConvFFN，需要把全连接权重重塑为卷积权重 ---
        def init_conv_from_ffn(weight, k):
            # weight shape: [out, in] -> [out, in, k, k]
            conv_weight = torch.zeros(weight.shape[0], weight.shape[1], k, k)
            center = (k - 1) // 2
            conv_weight[:, :, center, center] = weight
            return conv_weight

        # --- 辅助函数：修复 PatchMerging 权重形状 ---
        def correct_unfold_reduction_order(x):
            out_channel, in_channel = x.shape
            x = x.reshape(out_channel, 4, in_channel // 4)
            x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
            return x

        def correct_unfold_norm_order(x):
            in_channel = x.shape[0]
            x = x.reshape(4, in_channel // 4)
            x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
            return x

        for k, v in ckpt.items():
            # 1. 优先处理 Downsample (PatchMerging) 的权重不匹配问题
            if 'downsample' in k:
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
                else:
                    new_v = v
                new_ckpt[k] = new_v
                continue

            # 2. 处理 MLP -> MoE Experts 的转换
            if 'mlp.' in k:
                # 解析层级结构，例如 layers.0.blocks.1.mlp.fc1.weight
                # 使用 split 更加稳健
                parts = k.split('.')
                
                # 尝试找到 layer 和 block 的索引
                layer_idx = -1
                block_idx = -1
                try:
                    # 兼容 layers (mmcv) 或 stages (swin official)
                    if 'layers' in parts:
                        l_i = parts.index('layers')
                    elif 'stages' in parts:
                        l_i = parts.index('stages')
                    else:
                        # 如果找不到层级标识，跳过解析
                        raise ValueError
                    
                    layer_idx = int(parts[l_i+1])
                    
                    if 'blocks' in parts:
                        b_i = parts.index('blocks')
                        block_idx = int(parts[b_i+1])
                except:
                    pass

                # 判断该 block 在当前模型中是否被定义为 MoE block
                is_moe_block = False
                if layer_idx != -1 and block_idx != -1:
                    if layer_idx < len(self.layers):
                        if block_idx in self.layers[layer_idx].moe_block_ind:
                            is_moe_block = True
                
                if not is_moe_block:
                    # === Case A: 普通 Swin Block ===
                    # 直接保留原样
                    new_ckpt[k] = v
                else:
                    # === Case B: MoE Block (Warm Start) ===
                    # 将权重复制给所有 Expert
                    
                    # 获取当前 block 的引用以查看配置 (expert数量, 类型等)
                    curr_block = self.layers[layer_idx].blocks[block_idx]
                    
                    # 安全获取 expert 参数
                    num_experts = getattr(curr_block.mlp, 'num_experts', 1)
                    squads = getattr(curr_block.mlp, 'squads', None)
                    squad_num = getattr(curr_block.mlp, 'squad_num', 0)
                    
                    for expert_idx in range(num_experts):
                        new_v_ = v
                        
                        # 确定当前 expert 的类型 (FFN vs ConvFFN)
                        expert_type = 'FFN'
                        if squads is not None and squad_num > 0:
                            # 按照 SM3Det 逻辑计算当前 expert 类型
                            full_squads = squads * (num_experts // len(squads))
                            expert_type = full_squads[expert_idx]

                        # 根据类型构建子模块路径
                        sub_net = ''
                        
                        # 处理 fc1 (第一层)
                        if 'mlp.fc1.' in k:
                            if expert_type == 'FFN':
                                sub_net = '.layers.0.0.' # mmcv FFN Linear1
                            elif expert_type in ['Conv3x3_FFN', 'Conv5x5_FFN']:
                                sub_net = '.conv1.0.' # ConvFFN Conv1
                                # 如果是权重，需要将 FC 权重 reshape 塞入 Conv 核中心
                                if 'weight' in k:
                                    kernel_size = 3 if '3' in expert_type else 5
                                    new_v_ = init_conv_from_ffn(v, kernel_size)
                            
                            # 替换 key: mlp.fc1.weight -> mlp.experts.0.layers.0.0.weight
                            new_k = k.replace('mlp.fc1.', f'mlp.experts.{expert_idx}{sub_net}')

                        # 处理 fc2 (第二层)
                        elif 'mlp.fc2.' in k:
                            if expert_type == 'FFN':
                                sub_net = '.layers.1.' # mmcv FFN Linear2
                            elif expert_type in ['Conv3x3_FFN', 'Conv5x5_FFN']:
                                sub_net = '.ffn.0.' # ConvFFN Linear2
                            
                            new_k = k.replace('mlp.fc2.', f'mlp.experts.{expert_idx}{sub_net}')
                        
                        else:
                            new_k = k.replace('mlp.', f'mlp.experts.{expert_idx}.')

                        new_ckpt[new_k] = new_v_
                    
                    continue 

            # 3. 其他非 backbone 或不需要特殊处理的层
            new_ckpt[k] = v

        return new_ckpt

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone."""
        
        # 基础随机初始化
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # 1. 先对所有层应用随机初始化
        self.apply(_init_weights)

        # 2. 加载预训练权重
        if isinstance(pretrained, str):
            try:
                logger = create_logger(output_dir='logs', name=f"swin_init")
            except:
                import logging
                logger = logging.getLogger()

            print(f"Loading pretrained weights from {pretrained}...")
            checkpoint = torch.load(pretrained, map_location='cpu')

            if 'state_dict' in checkpoint:
                _state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                _state_dict = checkpoint['model']
            else:
                _state_dict = checkpoint

            # Warm Start
            print("Converting weights for MoE structure...")
            _state_dict = self.swin_converter(_state_dict)

            msg = self.load_state_dict(_state_dict, strict=False)
            
            print(f"Weights loaded successfully.")
            print(f"Missing keys: {len(msg.missing_keys)} (Expected: w_gate, w_noise, fusion layers)")
            print(f"Unexpected keys: {len(msg.unexpected_keys)}")
            
            expert_missing = [k for k in msg.missing_keys if 'experts' in k]
            if not expert_missing:
                print("Success: All experts initialized from pretrained weights!")
            else:
                print(f"Warning: Some experts missing: {expert_missing[:3]}...")

        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(MultiModalSwinTransformer, self).train(mode)
        self._freeze_stages()


class MMBasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 num_heads_fusion=1,
                 fusion_drop=0.0,
                 # --- MoE params ---
                 moe_block_ind=None,
                 noisy_gating=True,
                 num_experts=8,
                 gate='cosine',
                 top_k=2,
                 squads=None,
                 squad_num=0
                 # ------------------
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.dim = dim

        self.moe_block_ind = moe_block_ind if moe_block_ind is not None else []

        # build blocks
        # self.blocks = nn.ModuleList([
        #     SwinTransformerBlock(
        #         dim=dim,
        #         num_heads=num_heads,
        #         window_size=window_size,
        #         shift_size=0 if (i % 2 == 0) else window_size // 2,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop,
        #         attn_drop=attn_drop,
        #         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #         norm_layer=norm_layer)
        #     for i in range(depth)])
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Check if this block is an MoE block
            moe_cfg = None
            if i in self.moe_block_ind:
                moe_cfg = {
                    'noisy_gating': noisy_gating,
                    'num_experts': num_experts,
                    'top_k': top_k,
                    'gating': gate,
                    'squads': squads,
                    'squad_num': squad_num
                }
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    moe_cfg=moe_cfg # Pass MoE config
                )
            )

        # Cross-modal Alignment Branch
        self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
                           dim,  # v_in
                           768,  # l_in
                           dim,  # key
                           dim,  # value
                           num_heads=num_heads_fusion,
                           dropout=fusion_drop)

        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )

        # Various Receptive Branch
        self.visual_residual = VariousReceptive(dim)

        self.visual_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W, l, l_mask):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        aux_losses = []
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                # x = checkpoint.checkpoint(blk, x, attn_mask)
                x, loss = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                # x = blk(x, attn_mask)  # output of a Block has shape (B, H*W, dim)
                x, loss = blk(x, attn_mask)
            
            if loss != 0.0:
                aux_losses.append(loss)
        
        total_aux_loss = sum(aux_losses) if aux_losses else None

        v_residual = self.visual_residual(x)

        # PWAM fusion
        x_residual = self.fusion(x, l, l_mask)
        # apply a gate on the residual
        x = x + (self.res_gate(x_residual) * x_residual) + (self.visual_gate(v_residual) * v_residual)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_residual, H, W, x_down, Wh, Ww, total_aux_loss
        else:
            return x_residual, H, W, x, H, W, total_aux_loss


class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(PWAM, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )

        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=value_channels,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, dim)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)
        # print(x.shape, l.shape, l_mask.shape)

        lang = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim)

        lang = lang.permute(0, 2, 1)  # (B, dim, H*W)

        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        return mm


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out

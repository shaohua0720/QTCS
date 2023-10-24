import math
import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from itertools import repeat
import torch.nn.functional as F
# from vqlayer import VQLayer
import collections.abc as container_abcs
from torch.nn.modules.module import Module
import models
from .vqlayer import VQLayer


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


class Conv(Module):
    def __init__(self, config, ic, oc):
        super(Conv, self).__init__()
        self.config = config
        self.ic = ic
        self.oc = oc
        self.w = nn.Parameter(torch.Tensor(oc, ic, 9))
        self.padding = _pair(1)
        self.init = nn.Parameter(torch.zeros([ic, 9, 9], dtype=torch.float32))
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if config.DDP:
            self.device = int(os.environ["LOCAL_RANK"])
        else:
            self.device = config.device

    def forward(self, inputs):
        init = self.init + torch.eye(9, dtype=torch.float32).unsqueeze(0).repeat((self.ic, 1, 1)).to(self.device)
        weight = torch.reshape(torch.einsum('abc, dac->dab', init, self.w), (self.oc, self.ic, 3, 3))
        outputs = F.conv2d(inputs, weight, None, 1, self.padding)
        return outputs


class pre_layer(nn.Module):
    def __init__(self, config):
        super(pre_layer, self).__init__()

        self.num = 4

        self.conv_in = nn.Sequential(
            Conv(config, 1, 32),
            nn.BatchNorm2d(32),
            nn.ELU())

        self.conv = nn.ModuleList()
        for i in range(self.num):
            self.conv.append(nn.Sequential(
                Conv(config, 32, 32),
                nn.BatchNorm2d(32),
                nn.ELU())
            )

        self.conv_out = nn.Sequential(
            Conv(config, 32, 1))

    def forward(self, x_recon):
        x_recon = torch.transpose(x_recon, 0, 1).reshape([-1, 1, 32, 32])
        x_input = self.conv_in(x_recon)
        x_mid = x_input
        for i in range(self.num):
            x_mid = self.conv[i](x_mid)
        x_output = self.conv_out(x_mid)
        x_output = torch.transpose(x_output.reshape(-1, 1024), 0, 1)
        return x_output


class post_layer(nn.Module):
    def __init__(self, config):
        super(post_layer, self).__init__()

        self.num = 4

        self.conv_in = nn.Sequential(
            Conv(config, 1, 32),
            nn.BatchNorm2d(32),
            nn.ELU())

        self.conv = nn.ModuleList()
        for i in range(self.num):
            self.conv.append(nn.Sequential(
                Conv(config, 32, 32),
                nn.BatchNorm2d(32),
                nn.ELU())
            )

        self.conv_out = nn.Sequential(
            Conv(config, 32, 1))

    def forward(self, x_recon):
        x_input = self.conv_in(x_recon)
        x_mid = x_input
        for i in range(self.num):
            x_mid = self.conv[i](x_mid)
        x_output = self.conv_out(x_mid)
        return x_output


class Trans(nn.Module):
    def __init__(self, config, dim):
        super(Trans, self).__init__()
        self.config = config
        self.threshold = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)
        self.encoder = models.Encoder(dim=dim)
        self.decoder = models.Decoder(dim=dim)

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        outputs = torch.mul(torch.sign(outputs), F.relu(torch.abs(outputs) - self.threshold))
        outputs = self.decoder(inputs, outputs)
        return outputs


class HybridNet(nn.Module):
    def __init__(self, config):
        super(HybridNet, self).__init__()
        self.config = config
        self.phi_size = 32
        points = self.phi_size ** 2
        phi_init = np.random.normal(0.0, (1 / points) ** 0.5, size=(int(config.ratio * points), points))
        self.phi = nn.Parameter(torch.from_numpy(phi_init).float(), requires_grad=True)
        self.Q = nn.Parameter(torch.from_numpy(np.transpose(phi_init)).float(), requires_grad=True)

        self.quant = VQLayer(config.n_embed,embedding_dim=config.embed_d)

        self.num_layers = 6
        self.pre_block = nn.ModuleList()
        for i in range(self.num_layers):
            self.pre_block.append(pre_layer(config))

        self.post_block = nn.ModuleList()
        for i in range(self.num_layers):
            self.post_block.append(post_layer(config))

        self.trans = nn.ModuleList()
        for i in range(self.num_layers):
            self.trans.append(Trans(config, dim=8 ** 2))

        self.weights = []
        self.etas = []
        for i in range(self.num_layers):
            self.weights.append(nn.Parameter(torch.tensor(1.), requires_grad=True))
            self.register_parameter("eta_" + str(i + 1), nn.Parameter(torch.tensor(0.1), requires_grad=True))  # todo
            self.etas.append(eval("self.eta_" + str(i + 1)))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        inputs = torch.unsqueeze(inputs,1)
        y = self.sampling(inputs, self.phi_size)
        y_quant,quan_loss,commit_loss = self.quant(y)
        recon = self.recon(y_quant, self.phi_size, batch_size)
        recon= torch.squeeze(recon,1)
        return recon,quan_loss,commit_loss

    def sampling(self, inputs, init_block): # B C H W
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=init_block, dim=3), dim=0) # (B W/IB) C H IB
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=init_block, dim=2), dim=0) # (B W/IB H/IB) C IB IB
        inputs = torch.reshape(inputs, [-1, init_block ** 2]) # (B W/IB H/IB C) (IB IB)
        inputs = torch.transpose(inputs, 0, 1) #(IB IB) (B W/IB H/IB C)
        y = torch.matmul(self.phi, inputs) # (M=R*IB^2) (B W/IB H/IB C)
        return y

    def recon(self, y, init_block, batch_size):
        idx = int(self.config.block_size / init_block)

        recon = torch.matmul(self.Q, y) # (IB IB) x M x M x (B W/IB H/IB C) = (IB IB) (B W/IB H/IB C)
        for i in range(self.num_layers):
            recon = recon - self.weights[i] * torch.mm(torch.transpose(self.phi, 0, 1), (torch.mm(self.phi, recon) - y))
            recon = recon - self.pre_block[i](recon)
            recon = torch.reshape(torch.transpose(recon, 0, 1), [-1, 1, init_block, init_block])  # (B W/IB H/IB C) 1 IB IB = M 1 IB IB
            recon = torch.cat(torch.split(recon, split_size_or_sections=idx * batch_size, dim=0), dim=2) # K 1 IB*M/K IB
            recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3) # IDX 1 IB/M/K IB/IDX
            recon = self.size256to8(recon) # IDX (S L) 8 8
            recon = recon - self.etas[i] * self.trans[i](recon)
            recon = self.size8to256(recon)
            recon = recon - self.post_block[i](recon)

            recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=3), dim=0)
            recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=2), dim=0)
            recon = torch.reshape(recon, [-1, init_block ** 2])
            recon = torch.transpose(recon, 0, 1)

        recon = torch.reshape(torch.transpose(recon, 0, 1), [-1, 1, init_block, init_block])
        recon = torch.cat(torch.split(recon, split_size_or_sections=idx * batch_size, dim=0), dim=2)
        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)
        return recon

    def size8to256(self, inputs):
        idx = int(self.config.block_size / 8)
        outputs = torch.cat(torch.split(inputs, split_size_or_sections=idx, dim=1), dim=2)
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=1, dim=1), dim=3)
        return outputs

    def size256to8(self, inputs):
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=8, dim=3), dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=8, dim=2), dim=1)
        return inputs

def QCSLoss(x, reco_x,quant_loss,commit_loss):
    mse_loss = F.mse_loss(reco_x, x,reduction='sum')/x.shape[0]
    loss = mse_loss+quant_loss+0.25*commit_loss
    return loss
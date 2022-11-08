"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

This code borrows heavily from the DimeNet implementation as part of
pytorch-geometric: https://github.com/rusty1s/pytorch_geometric. License:

---

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from math import sqrt
from typing import Optional

import torch
from cugraph_ops_binding import MMAOp, ib_agg_edge, radial_basis
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad, get_edge_offsets, get_pbc_distances
from torch import nn


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= (tensor.size(-2) + tensor.size(-1)) * tensor.var()
        tensor.data *= scale.sqrt()


@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


@torch.jit.script
def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


@torch.jit.script
def besselBasisLayer(dist, vec, env):
    return env * (vec * dist).sin()


@torch.jit.script
def swish(x):
    return x * x.sigmoid()


@torch.jit.script
def envelope(x):
    x_pow_p0 = torch.pow(x, 5)
    x_pow_p1 = x_pow_p0 * x
    x_pow_p2 = x_pow_p1 * x
    return (1.0 / x - 28 * x_pow_p0 + 48 * x_pow_p1 - 21 * x_pow_p2) * (x < 1.0)


@torch.jit.script
def output_mul(x, rbf, i, num_nodes):
    return scatter(rbf * x, i, dim=0, dim_size=num_nodes)


class ResidualLayer(torch.jit.ScriptModule):
    def __init__(self, hidden_channels=192):
        super().__init__()
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    @torch.jit.script_method
    def forward(self, x):
        return x + swish(self.lin2(swish(self.lin1(x))))


class EmbeddingBlock(torch.jit.ScriptModule):
    def __init__(self, num_radial, hidden_channels):
        super().__init__()
        self.emb = nn.Embedding(95, hidden_channels)
        self.bias = nn.Parameter(torch.zeros(hidden_channels))
        self.bound = 1 / sqrt(num_radial)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        nn.init.uniform_(self.bias, -self.bound, self.bound)

    @torch.jit.script_method
    def forward(self, x, rbf, i, j):
        x, rbf = self.emb(x), swish(rbf + self.bias)
        return swish(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class InteractionPPBlock_pre(torch.jit.ScriptModule):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size,
    ):
        super(InteractionPPBlock_pre, self).__init__()
        self.hidden_channels = hidden_channels
        # Transformations of Bessel and spherical basis representations.
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        # Dense transformations of input messages.
        self.lin_x = nn.Linear(hidden_channels, 2 * hidden_channels)
        self.lin_x_split = [hidden_channels, hidden_channels]
        # Embedding projections for interaction triplets.
        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        for w in torch.split(self.lin_x.weight, self.lin_x_split, dim=0):
            glorot_orthogonal(w, scale=2.0)
        self.lin_x.bias.data.fill_(0)
        glorot_orthogonal(self.lin_down.weight, scale=2.0)

    @torch.jit.script_method
    def forward(self, x, rbf):
        # Initial transformations.
        rbf = self.lin_rbf2(rbf)
        x_ji, x_kj = torch.split(swish(self.lin_x(x)), self.lin_x_split, dim=1)
        # Transformation via Bessel basis.
        x_kj = x_kj * rbf
        # Down-project embeddings and generate interaction triplet embeddings.
        x_kj = swish(self.lin_down(x_kj))

        return x_ji, x_kj


class InteractionPPBlock_post(torch.jit.ScriptModule):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        num_before_skip,
        num_after_skip,
    ):
        super(InteractionPPBlock_post, self).__init__()
        self.hidden_channels = hidden_channels
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)
        # Residual layers before and after skip connection.
        self.layers_before_skip = torch.nn.ModuleList([ResidualLayer(hidden_channels) for _ in range(num_before_skip)])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([ResidualLayer(hidden_channels) for _ in range(num_after_skip)])
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    @torch.jit.script_method
    def forward(self, x, x_ji, x_kj):
        # Up-project embeddings.
        x_kj = swish(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = swish(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputPPBlock(torch.jit.ScriptModule):
    def __init__(
        self,
        hidden_channels,
        out_emb_channels,
        num_layers,
    ):
        super(OutputPPBlock, self).__init__()
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)

    @torch.jit.script_method
    def forward(self, x, rbf, i, num_nodes):
        x = rbf * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = swish(lin(x))
        return x


class DimeNetPlusPlus(torch.nn.Module):
    r"""DimeNet++ implementation based on https://github.com/klicperajo/dimenet.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size (int): Embedding size used in the basis transformation
        out_emb_channels(int): Embedding size used for atoms in the output block
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion.
            (default: :obj:`swish`)
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_blocks,
        int_emb_size,
        basis_emb_size,
        out_emb_channels,
        num_spherical,
        num_radial,
        cutoff=6.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        device="cuda",
    ):
        super(DimeNetPlusPlus, self).__init__()
        self.emb = EmbeddingBlock(num_radial, hidden_channels)
        self.cutoff = cutoff
        self.num_blocks = num_blocks
        self.hidden_channels = hidden_channels
        rbf_splits = [[basis_emb_size, hidden_channels] for _ in range(num_blocks)]
        rbf_splits.insert(0, [hidden_channels, hidden_channels])
        self.rbf_splits = list(itertools.chain(*rbf_splits))
        self.vec = nn.Parameter(torch.arange(1, num_radial + 1, device=device) * 3.141592653589793)
        self.num_spherical = num_spherical
        self.num_radial = num_radial

        self.lin_rbf = nn.Linear(
            num_radial, num_blocks * basis_emb_size + (num_blocks + 2) * hidden_channels, bias=False
        )
        self.lin_sbf = nn.Parameter(torch.empty(num_blocks, num_spherical * num_radial + int_emb_size, basis_emb_size))
        self.output_blocks = torch.nn.ModuleList(
            [
                OutputPPBlock(
                    hidden_channels,
                    out_emb_channels,
                    num_output_layers,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks_pre = torch.nn.ModuleList(
            [
                InteractionPPBlock_pre(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                )
                for _ in range(num_blocks)
            ]
        )
        self.interaction_blocks_post = torch.nn.ModuleList(
            [
                InteractionPPBlock_post(
                    hidden_channels,
                    int_emb_size,
                    num_before_skip,
                    num_after_skip,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_feat = out_emb_channels * (num_blocks + 1)
        self.out_emb_channels = out_emb_channels
        self.out_channels = out_channels
        self.output_lin = nn.Linear(self.out_feat, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for w in torch.split(self.lin_rbf.weight, self.rbf_splits, dim=0):
            glorot_orthogonal(w, scale=2.0)
        nn.init.kaiming_uniform_(self.lin_rbf.weight[: self.hidden_channels], a=sqrt(5))
        sbf_split = self.num_spherical * self.num_radial
        for w in self.lin_sbf:
            glorot_orthogonal(w[:sbf_split, :], scale=2.0)
            glorot_orthogonal(w[sbf_split:, :].T, scale=2.0)
        self.output_lin.weight.data.fill_(0)

    def forward(self, z, pos, batch=None):
        """"""
        raise NotImplementedError


@registry.register_model("dimenetplusplus")
class DimeNetPlusPlusWrap(DimeNetPlusPlus):
    def __init__(
        self,
        num_atoms,
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        hidden_channels=128,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        otf_graph=False,
        cutoff=10.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        device="cuda",
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.device = device

        super(DimeNetPlusPlusWrap, self).__init__(
            hidden_channels=hidden_channels,
            out_channels=num_targets,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            device=device,
        )

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        pos, edge_index, batch = data.pos, data.edge_index, data.batch
        j, i = edge_index
        dist = get_pbc_distances(pos[j], pos[i], data.offsets)

        # Calculate RBF and radial part of SBF features
        rbf, sbf_rad = radial_basis(dist, self.vec)
        rbf = self.lin_rbf(rbf)
        rbf = torch.split(rbf, self.rbf_splits, dim=1)
        rbf_out, rbf_inter = rbf[1::2], rbf[2::2]

        # Get pos_diff & edge offsets for interaction block aggregation
        pos_d = pos.detach()
        pos_diff = pos_d[j] - pos_d[i] + data.offsets
        edge_offsets = get_edge_offsets(edge_index, pos.size(0))
        # Embedding, interaction and output blocks.
        num_nodes = torch.tensor(pos.size(0))
        x = self.emb(data.atomic_numbers.long(), rbf[0], i, j)
        xs = [self.output_blocks[0](x, rbf_out[0], i, num_nodes=num_nodes)]
        for block_idx in range(self.num_blocks):
            x_ji, x_kj = self.interaction_blocks_pre[block_idx](x, rbf_inter[block_idx])
            # apply spherical basis features and aggregate embeddings
            x_kj = ib_agg_edge(
                pos_diff,
                sbf_rad,
                x_kj,
                self.lin_sbf[block_idx],
                edge_index,
                edge_offsets[0],
                edge_offsets[1],
                edge_offsets[2],
                MMAOp.HighPrecision,
            )
            x = self.interaction_blocks_post[block_idx](x, x_ji, x_kj)
            xs.append(self.output_blocks[block_idx + 1](x, rbf_out[block_idx + 1], i, num_nodes=num_nodes))
        energy = scatter(self.output_lin(torch.cat(xs, dim=1)), batch, dim=0)

        return energy

    def forward(self, data):
        data.pos.requires_grad_(True)
        energy = self._forward(data)
        forces = -1 * (
            torch.autograd.grad(
                energy,
                data.pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )
        return energy, forces

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

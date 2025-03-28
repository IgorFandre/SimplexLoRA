# Copyright 2023-present the HuggingFace Inc. team.
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

import math
from typing import Any, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lycoris_utils import LycorisLayer


class WeightLoraLayer(nn.Module, LycorisLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = (
        "weight_lora_A",
        "weight_lora_B",
        "weight_lora_w",
    )
    # other_param_names is defined on parent class

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__()
        LycorisLayer.__init__(self, base_layer)

        # Weight Lora info
        self.weight_lora_A = nn.ParameterDict({})
        self.weight_lora_B = nn.ParameterDict({})
        self.weight_lora_w = nn.ParameterDict({})

    @property
    def _available_adapters(self) -> Set[str]:
        return {
            *self.weight_lora_A,
            *self.weight_lora_B,
            *self.weight_lora_w,
        }

    def create_adapter_parameters(
        self,
        adapter_name: str,
        r: int,
        shape
    ):
        self.weight_lora_A[adapter_name] = nn.Parameter(torch.empty(shape[0], r))
        self.weight_lora_B[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))
        self.weight_lora_w[adapter_name] = nn.Parameter(torch.empty(1))

    def reset_adapter_parameters(self, adapter_name: str):
        nn.init.kaiming_uniform_(self.weight_lora_A[adapter_name], a=math.sqrt(5))
        nn.init.zeros_(self.weight_lora_B[adapter_name])
        nn.init.ones_(self.weight_lora_w[adapter_name])

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        rank_dropout: float,
        module_dropout: float,
        **kwargs,
    ) -> None:
        """Internal function to create weight lora adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            lora_alpha (`float`): Alpha for the added adapter.
            rank_dropout (`float`): The dropout probability for rank dimension during training
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize adapter weights.
            use_effective_conv2d (`bool`): Use parameter effective decomposition for Conv2d with ksize > 1.
            decompose_both (`bool`): Perform rank decomposition of left kronecker product matrix.
            decompose_factor (`int`): Kronecker product decomposition factor.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r
        self.rank_dropout[adapter_name] = rank_dropout
        self.module_dropout[adapter_name] = module_dropout
        base_layer = self.get_base_layer()

        # Determine shape of WeightLora weights
        if isinstance(base_layer, nn.Linear):
            shape = (base_layer.in_features, base_layer.out_features)
        else:
            raise TypeError(f"WeightLora is not implemented for base layers of type {type(base_layer).__name__}")

        # Create weights with provided shape
        self.create_adapter_parameters(adapter_name, r, shape)

        # Initialize weights
        self.reset_adapter_parameters(adapter_name)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/e4259b870d3354a9615a96be61cb5d07455c58ea/lycoris/modules/lokr.py#L224
        device = self.weight_lora_B[adapter_name].device
        dtype = self.weight_lora_B[adapter_name].dtype
        w_A = self.weight_lora_A[adapter_name]
        w_B = self.weight_lora_B[adapter_name]
        w = self.weight_lora_w[adapter_name]

        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
        if cast_to_fp32:
            w_A = w_A.float()
            w_B = w_B.float()

        # Combine marixes
        weight = w * w_A @ w_B * self.scaling[adapter_name]
        weight = weight.T
        if cast_to_fp32:
            weight = weight.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter_name].weight.data = w_A.to(dtype)
            self.lora_B[adapter_name].weight.data = w_B.to(dtype)
        # print(self.get_base_layer().weight.shape, type(self.get_base_layer()))
        # weight = weight.reshape(self.get_base_layer().weight.shape)

        # Perform rank dropout during training - drop rows of addition weights
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (torch.rand(weight.size(0)) > rank_dropout).float()
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            drop /= drop.mean()
            weight *= drop

        return weight

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)

            # Execute all the adapters
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    result = result + self._get_delta_activations(active_adapter, x, *args, **kwargs)

        result = result.to(previous_dtype)
        return result

    def update_lora_rank_QR(self, new_rank, adapter_name):
        # A: m x r, B: r x n, W (base): n x m
        device = self.weight_lora_B[adapter_name].device
        w_A = self.weight_lora_A[adapter_name]
        w_B = self.weight_lora_B[adapter_name]

        current_rank = self.r[adapter_name]
        n, m = self.get_base_layer().weight.shape

        if new_rank == 0:
            # why not?
            self.get_base_layer().weight += self.get_delta_weight(adapter_name)

            # TODO check for bugs
            # TODO disable_adapters 
            # self.disable_adapters = True
            self.weight_lora_A[adapter_name].data = torch.zeros((m, 0), requires_grad=False, device=device)
            self.weight_lora_B[adapter_name].data = torch.zeros((0, n), requires_grad=False, device=device)
            
        elif new_rank > current_rank:
            Q, R = torch.linalg.qr(w_A, mode="reduced")
            N = torch.randn((m, new_rank - current_rank), device=device)
            I = torch.eye(m, device=device)
            O = torch.zeros((new_rank - current_rank, n), device=device)

            self.weight_lora_A[adapter_name].data = torch.concat([Q, (I - Q @ Q.T) @ N], dim=1)
            self.weight_lora_B[adapter_name].data = torch.concat([R @ w_B, O], dim=0)

            self.weight_lora_A[adapter_name].requires_grad = True
            self.weight_lora_B[adapter_name].requires_grad = True

        elif new_rank < current_rank:
            # TODO debug
            Q_A, R_A = torch.linalg.qr(w_A, mode="reduced")
            Q_B, R_B = torch.linalg.qr(w_B.T, mode="reduced")
            U, S, V = torch.linalg.svd(R_A @ R_B.T)
            print("Q_A, Q_B, R_A, R_B: ", Q_A.shape, Q_B.shape, R_A.shape, R_B.shape)
            print("before: ", U.shape, S.shape, V.shape, new_rank)

            dim_S = new_rank
            if len(S) < dim_S:
                S = torch.diag(
                    torch.concat((S, torch.zeros(dim_S - len(S), device=device)))
                )[:dim_S, :dim_S]
            else:
                S = torch.diag(S)

            U_r = U[:, :new_rank]
            S_r = S[:new_rank, :new_rank]
            V_r = V[:new_rank, :]

            print("after: ", U.shape, S.shape, V.shape)


            self.weight_lora_A[adapter_name].data = Q_A @ U_r
            self.weight_lora_B[adapter_name].data = S_r @ V_r @ Q_B.T

            self.weight_lora_A[adapter_name].requires_grad = True
            self.weight_lora_B[adapter_name].requires_grad = True

        self.r[adapter_name] = new_rank

        if new_rank == 0:
            self.scaling[adapter_name] = 0
        else:
            self.scaling[adapter_name] = self.lora_alpha[adapter_name] / new_rank
                        

    def final_lora_rank_update(self, new_rank, adapter_name):
        # A: m x r, B: r x n, W (base): n x m
        device = self.weight_lora_B[adapter_name].device

        n, m = self.get_base_layer().weight.shape

        # TODO multiply dropout
        self.get_base_layer().weight += self.get_delta_weight(adapter_name)

        # TODO rank 0 case
        self.weight_lora_A[adapter_name].data = torch.randn((m, new_rank), requires_grad=True, device=device)
        self.weight_lora_B[adapter_name].data = torch.zeros((new_rank, n), requires_grad=True, device=device)
        self.weight_lora_w[adapter_name].data = torch.tensor(1., requires_grad=False, device=device)

        self.r[adapter_name] = new_rank

        if new_rank == 0:
            self.scaling[adapter_name] = 0
        else:
            self.scaling[adapter_name] = self.lora_alpha[adapter_name] / new_rank
    
    def update_alpha(self, multiplier, adapter_name):
        self.lora_alpha[adapter_name] *= multiplier
        self.scaling[adapter_name] = self.lora_alpha[adapter_name] / self.r[adapter_name]


class Linear(WeightLoraLayer):
    """WeightLora implemented in Linear layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        adapter_name: str = "default",
        r: int = 0,
        lora_alpha: float = 1.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, rank_dropout, module_dropout, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)
        # don't add bias here, because the bias is already included in the output of the base_layer
        # print(input.weight.shape(), delta_weight.weight.shape(), end="-----------\n")
        delta_weight = delta_weight.to(input.dtype)
        return F.linear(input, delta_weight)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "weight_lora." + rep

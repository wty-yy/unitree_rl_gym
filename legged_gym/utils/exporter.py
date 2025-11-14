# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
from typing import Optional


def export_policy_as_jit(policy: object, path: str, normalizer: Optional[object] = None, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: Optional[object] = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "student_encoder"):
            self.student_encoder = copy.deepcopy(policy.student_encoder)
            self.history = torch.zeros_like(policy.history)
            self.forward = self.forward_cts
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))
    
    def forward_cts(self, x):  # x is single observations
        x = self.normalizer(x)
        latent = self.student_encoder(x)
        x = torch.cat([latent, x], dim=1)
        return self.actor(x)

    @torch.jit.export
    def reset(self):
        if hasattr(self, 'history'):
            self.history = torch.zeros_like(self.history)

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        self.input_dim = None
        # copy policy parameters
        if hasattr(policy, 'student_encoder'):
            self.student_encoder = copy.deepcopy(policy.student_encoder)
            self.num_actions = policy.num_actions
            self.forward = self.forward_cts
            self.input_dim = self.student_encoder[0].in_features
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
            if self.input_dim is None:
                 self.input_dim = self.actor[0].in_features
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
            if self.input_dim is None:
                 self.input_dim = self.actor[0].in_features
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x))
    
    # def forward_cts(self, x):  # x is stack observations by terms
    #     x = self.normalizer(x)
    #     term_dims = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
    #     obs_dim = sum(term_dims)
    #     frames = x.shape[1] // obs_dim
    #     frame_terms = []
    #     start = 0
    #     for dim in term_dims:
    #         terms = []
    #         for _ in range(frames):
    #             terms.append(x[:, start:dim])
    #             start += dim
    #         frame_terms.append(terms)
    #     history = []
    #     for i in range(frames):
    #         for j in range(len(term_dims)):
    #             history.append(frame_terms[j][i])
    #     history = torch.cat(history, dim=1)
    #     last_obs = history[:, -obs_dim:]
    #     latent = self.student_encoder(history)
    #     x = torch.cat([latent, last_obs], dim=1)
    #     return self.actor(x)

    def forward_cts(self, x):  # x is stack observations by terms
        x = self.normalizer(x)
        term_dims = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
        obs_dim = sum(term_dims)
        if x.shape[1] % obs_dim != 0:
            raise ValueError(f"x.shape[1] ({x.shape[1]}) 不是 obs_dim ({obs_dim}) 的整数倍")
            
        frames = x.shape[1] // obs_dim
        split_sizes = [dim * frames for dim in term_dims]
        # [B, dim0*frames], [B, dim1*frames], ...
        term_chunks = torch.split(x, split_sizes, dim=1)

        # [ [B, frames, dim0], [B, frames, dim1], ... ]
        frame_terms_reshaped = [
            chunk.view(-1, frames, dim) 
            for chunk, dim in zip(term_chunks, term_dims)
        ]

        history_by_frame = []
        for i in range(frames):
            # [ [B, dim0], [B, dim1], ... ]
            terms_for_this_frame = [ftr[:, i, :] for ftr in frame_terms_reshaped]
            history_by_frame.append(torch.cat(terms_for_this_frame, dim=1))
        # [B, (Frame0_AllTerms), (Frame1_AllTerms), ...]
        history = torch.cat(history_by_frame, dim=1)

        last_obs = history[:, -obs_dim:]
        latent = self.student_encoder(history)
        x = torch.cat([latent, last_obs], dim=1)
        
        return self.actor(x)

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.input_dim)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )

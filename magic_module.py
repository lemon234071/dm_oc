import math
import torch
import torch.nn as nn

from pytorch_pretrained_bert.modeling import BertSelfAttention
from transformers.modeling_gpt2 import Attention

import operator
import copy


class MagicModule(nn.Module):
    def __init__(self, module):
        nn.Module.__init__(self)
        self._type = type(module)

        for key, value in module._parameters.items():
            self.register_parameter('_origin_' + key, value)
            if value is not None:
                self.register_buffer(key, value.data)
            else:
                self.register_buffer(key, None)

        for key, value in module._buffers.items():
            self.register_buffer(key, copy.deepcopy(value))

        for key, value in module._modules.items():
            self.add_module(key, MagicModule(value))

        for key, value in module.__dict__.items():
            if (not key in self.__dict__) and\
                    (not key in self._buffers) and\
                    (not key in self._modules):
                self.__setattr__(key, value)

    def forward(self, *args, **kwargs):
        return self._type.forward(self, *args, **kwargs)

    def update_params(self, deltas):
        sub_params = {}
        for key, delta in deltas.items():
            if not ('.' in key):
                self._buffers[key] = self._buffers[key] + delta
            else:
                attr = key.split('.')[0]
                if not (attr in sub_params):
                    sub_params[attr] = {}
                sub_params[attr]['.'.join(key.split('.')[1:])] = delta
        for key, value in sub_params.items():
            self._modules[key].update_params(value)

    def check_forward_args(self, *args, **kwargs):
        assert issubclass(self._type, nn.RNNBase)
        return nn.RNNBase.check_forward_args(self, *args, **kwargs)

    @property
    def _flat_weights(self):
        assert issubclass(self._type, nn.RNNBase)
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):
        assert issubclass(self._type, nn.RNNBase)
        return [[getattr(self, weight) for weight in weights] for weights in
                self._all_weights]

    def _get_abs_string_index(self, idx):
        assert issubclass(self._type, nn.ModuleList)
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        assert issubclass(self._type, nn.ModuleList)
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __len__(self):
        assert issubclass(self._type, nn.ModuleList)
        return len(self._modules)

    # Model methods
    def transpose_for_scores(self, x):
        assert issubclass(self._type, BertSelfAttention)
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def split_heads(self, x, k=False):
        assert issubclass(self._type, Attention)
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        assert issubclass(self._type, Attention)
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        assert issubclass(self._type, Attention)
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states
from transformers import LlamaForCausalLM
from .configuration_oursvd_llama import CovSVDLlamaConfig
import torch.nn as nn
import torch.nn.functional as F
import torch

class CovSVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.weight_residual = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_residual.requires_grad = False


    def forward(self, input):
        y = self.BLinear(input)
        y = self.ALinear(y) + F.linear(input, self.weight_residual)
        return y


class CovSVDLlamaForCausalLM(LlamaForCausalLM):
    config_class = CovSVDLlamaConfig
    def __init__(self, config:CovSVDLlamaConfig):
        super().__init__(config)
        
        self.lora_r = config.lora_r
        full_name_dict = {module: name for name, module in self.named_modules()}
        linear_info = {}
        modules = [self]
        while len(modules) > 0:
            submodule = modules.pop()
            for name, raw_linear in submodule.named_children():
                if isinstance(raw_linear, nn.Linear):
                    full_name = full_name_dict[raw_linear]
                    linear_info[raw_linear] = {
                        "father": submodule,
                        "name": name,
                        "full_name": full_name,
                    }
                else:
                    modules.append(raw_linear)


        for name,module in self.named_modules():
            if "lm_head" not in name and isinstance(module, nn.Linear):
                info=linear_info[module]
                new_layer=CovSVDLinear(module.in_features, module.out_features, self.lora_r, bias=module.bias is not None)
                setattr(info["father"], info["name"], new_layer)

                
        
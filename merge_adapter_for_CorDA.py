import numpy as np
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from cordalib.evaluate_utils import evaluate_model
from mapping.modeling_oursvd_llama import CovSVDLinear
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
from transformers.models.llama.modeling_llama import LlamaMLP
import os

def main(args):
    # Load model
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    print("\n---- model before merge ---\n")
    print(model)
    
    # evaluate
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "",
        eval_ppl="wikitext2,ptb",
        limit=-1,
    )
    print("Wiki PTB perplexity before merge (used to check the difference before and after merging) ")
    print(result)
    
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            #if isinstance(raw_linear, nn.Linear):
            if name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)
    #print(linear_info)
    ## merge =======
    print("\nbegin merge. \n")
    module_dict = {module: name for name, module in model.named_modules()}
    for module in module_dict.keys():
        #if isinstance(module, LlamaSdpaAttention) or isinstance(module, LlamaMLP):
        #    for name, sub_module in module.named_children():
        name = module_dict[module]
        #if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "gate_proj" in name or "up_proj" in name or "down_proj" in name:
        if type(module).__name__ == "CovSVDLinear":
            #print(name, module)
            info = linear_info[module]
                #if isinstance(sub_module, CovSVDLinear):
                #if name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            in_features = module.BLinear.in_features
            out_features = module.ALinear.out_features
            new_linear = nn.Linear(in_features, out_features, bias=False)
            merged_weight = module.ALinear.weight.data @ module.BLinear.weight.data + module.weight_residual # A: out, r B: r, in 
            new_linear.weight.data = merged_weight
            delattr(info["father"], info["name"])
            setattr(info["father"], info["name"], new_linear)

    print("\n---- model after merge ---\n")
    print(model)
    
    # evaluate again:
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "",
        eval_ppl="wikitext2,ptb",
        limit=-1,
    )
    print("Wiki PTB perplexity afater merge (used to check the difference before and after merging) ")
    print(result)
    


    ## save as hugging face model 
    if args.save_model:
        assert args.save_path is not None
        save_path = args.save_path

        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        config = model.config.to_dict()
        #config["lora_r"] = args.r
        #config["atten_diag"] = args.atten_diag
        #config["auto_map"] = {
        #    "AutoConfig": "configuration_oursvd_llama.CovSVDLlamaConfig",
        #    "AutoModelForCausalLM": "modeling_oursvd_llama.CovSVDLlamaForCausalLM",
        #}
        #config["architectures"] = ["CovSVDLlamaForCausalLM"]
        config["architectures"] = ["LlamaForCausalLM"]
        del config["lora_r"]
        del config["auto_map"]
        del config["_name_or_path"]
        #os.system(
        #    "cp ./configuration_oursvd_llama.py ./modeling_oursvd_llama.py ./"
        #    + save_path
        #)
        import json

        json.dump(config, open(save_path + "/config.json", "w"), indent=2)

        print(f"Done merging adapter into the original model architecture in {save_path}")
        del model
        del tokenizer
    # finished

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--save_model",
        default=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    main(args)

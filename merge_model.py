import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml
import shutil
import os

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def merge_and_save():
    config = load_config()
    base_model_name = config['model']['base_model_name']
    adapter_model_name = config['model']['new_model_name']
    output_dir = adapter_model_name + "-merged"

    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": "cpu"} # Load on CPU to save VRAM during merge
    )

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"Merging with adapter: {adapter_model_name}")
    model = PeftModel.from_pretrained(base_model, adapter_model_name)
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    merge_and_save()

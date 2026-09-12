import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    config = load_config()
    base_model_name = config['model']['base_model_name']
    new_model_name = config['model']['new_model_name']

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print("Loading PEFT adapters...")
    model = PeftModel.from_pretrained(base_model, new_model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("\n--- LLM Inference Testing ---")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        # Format instruction if applicable
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        response = generate_response(prompt, model, tokenizer)
        
        # Extract response part
        clean_response = response.split("### Response:\n")[-1].strip()
        print(f"Assistant: {clean_response}\n")

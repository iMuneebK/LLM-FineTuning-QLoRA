import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train():
    config = load_config()
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 2. Configure Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['training']['load_in_4bit'],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    
    # 3. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model_name'],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    
    # 4. Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        r=config['lora']['r'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    # 5. Load Dataset
    from data_prep import load_and_prepare_data
    dataset = load_and_prepare_data(config['data']['dataset_path'])
    
    # 6. Set Training Arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )
    
    # 7. Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config['training']['max_seq_length'],
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    
    print("Starting training...")
    trainer.train()
    
    # 8. Save Model
    trainer.model.save_pretrained(config['model']['new_model_name'])
    print(f"Model saved to {config['model']['new_model_name']}")

if __name__ == "__main__":
    train()

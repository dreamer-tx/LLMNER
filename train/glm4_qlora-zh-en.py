import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用两张显卡
import torch
import wandb
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch.nn as nn

# 数据处理
tokenizer = AutoTokenizer.from_pretrained('.../glm-4-9b-chat', use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载数据集
df = pd.read_json('/LLM-NER/data/train-allinone.json')
ds = Dataset.from_pandas(df)

# 数据集处理函数
def process_func(example):
    MAX_LENGTH = 1024
    input_ids, attention_mask, labels = [], [], []
    
    instruction = tokenizer(
        (f"[gMASK]<sop><|system|>\nYou are an expert in medical named entity recognition.<|user|>\n"
         f"{example['instruction']+example['input']}<|assistant|>\n").strip(), 
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 做截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    print(lora_module_names)
    return lora_module_names

# 数据处理
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForCausalLM.from_pretrained(
    '.../glm-4-9b-chat',
    quantization_config=quantization_config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)
if 'output_router_logits' in model.config.to_dict():
    model.config.output_router_logits = True

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Lora微调配置
target_modules = find_all_linear_names(model, 'qlora')
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

args = TrainingArguments(
    output_dir="/home/Users/lvtx/cner/LLM-NER/output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=3,
    logging_steps=10,
    num_train_epochs=16,
    save_steps=15446,
    learning_rate=2e-4,
    save_total_limit=16,
    gradient_checkpointing=True,
    bf16=True,
    logging_dir='./logs',
    load_best_model_at_end=False,
    evaluation_strategy="no",
    eval_steps=15446,
    report_to= 'none',
    fp16=False,
)



trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()


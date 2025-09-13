import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(".../glm-4-9b-chat", trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained(
    ".../glm-4-9b-chat",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()


model = PeftModel.from_pretrained(model, model_id=".../output/checkpoint")

import json
import torch

with open("/LLM-NER/data/test-allinone.json", "r", encoding="utf-8") as input_file, \
     open("/LLM-NER/result/reslut-tem0.5.jsonl", "w", encoding="utf-8") as output_file:
    
    i = 0
    # 读取整个 JSON 文件
    data = json.load(input_file)
    
    for entry in data:
        # 从每个条目中获取 instruction 和 input
        query = entry['instruction'] + "\n" + entry['input']
        print("提问：", query)
        
        inputs = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are an expert in medical named entity recognition."},
             {"role": "user", "content": query}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )

        inputs = inputs.to(device)

        gen_kwargs = {"max_length": 8192, "do_sample": True, "top_k": 8,"temperature":0.5}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            outputs_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            i += 1
            print(i, "Factory：{}".format(outputs_result))
            
            # 将英文双引号替换为中文双引号
            outputs_result = outputs_result.replace('"', '“').replace('"', '”')

            if isinstance(outputs_result, dict): 
                # 写入文件
                output_file.write("{\"answer\":\"" + outputs_result["name"] + "\n" + outputs_result["content"] + "\"}" + "\n")
            elif isinstance(outputs_result, str):
                # 如果 outputs_result 是字符串，直接写入文件
                output_file.write(f'{{"answer":"{outputs_result}"}}\n')
            else:
                print("错误: outputs_result 既不是字典也不是字符串。")



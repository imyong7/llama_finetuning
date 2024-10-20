from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import pandas as pd
import torch

if __name__ == "__main__" :
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    input_text = 'What is the Kubernetes configuration for a simple apache web server?'

    print(input_text)


    use_model_name = "llama-3.2-3b-instruct"

    # token_path 경로 수정 필요
    token_path = "D:/py_workspace/llama_finetuning/model/{}/original".format(use_model_name)

    # model_path 경로 수정 필요
    model_path = "D:/py_workspace/llama_finetuning/model/{}".format(use_model_name)


    tokenizer = AutoTokenizer.from_pretrained(token_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()  # 추론 모드로 설정 (필수)
    model.to(device)


    # 입력 텍스트를 토큰화
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # 모델에 입력을 넣고 텍스트 생성
    output = model.generate(input_ids, max_length=10000, num_return_sequences=1, no_repeat_ngram_size=2)

    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Generated text:", generated_text)
    
    
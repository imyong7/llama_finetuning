from datetime import datetime

from datasets import Dataset
from transformers import  LlamaForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from peft import LoraConfig, get_peft_model

def tokenize_function(text):
    if isinstance(text, str):  # 각 값이 문자열인지 확인
        return tokenizer(
            text,
            padding="max_length",  # max_length 기준으로 패딩
            truncation=True,       # 길이가 초과되면 자름
            max_length=512         # 최대 토큰 길이를 512로 설정
        )
    else:
        raise ValueError(f"Invalid input type: {type(text)}. Expected a string.")



if __name__ == "__main__" :
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    use_model_name = "llama-3.2-3b"

    data_parquet = pd.read_parquet('data/train.parquet')

    # token_path 경로 수정 필요
    token_path = "D:/py_workspace/llama_finetuning/model/{}/original".format(use_model_name)

    tokenizer = AutoTokenizer.from_pretrained(token_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    # print(tokenizer)

    train_data, validation_data = train_test_split(data_parquet, test_size=0.2)


    # text_data = data_parquet['text']

    # tokenized_data = text_data.apply(tokenize_function)
    # tokenized_data = []
    # for text in text_data:
    #     tokenized_data.append(tokenize_function(text))

    tokenized_train_data = []
    for text in train_data['text']:
        dict = tokenize_function(text)
        dict['labels'] = dict['input_ids']
        tokenized_train_data.append(dict)

    dataset_train = Dataset.from_list(tokenized_train_data)


    #
    tokenized_validation_data = []
    for text in validation_data['text']:
        dict = tokenize_function(text)
        dict['labels'] = dict['input_ids']
        tokenized_validation_data.append(dict)

    dataset_validation = Dataset.from_list(tokenized_validation_data)


    #
    # 6. 토크나이즈된 데이터 확인
    # model_path 경로 수정 필요
    model_path = "D:/py_workspace/llama_finetuning/model/{}".format(use_model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map='auto'
    )

    #    for param in model.parameters():
    #        if param.dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
    #            param.requires_grad = True  # 부동소수점 텐서에 대해서만 requires_grad 설정
    # param.requires_grad = True  # 파라미터 고정 해제

    # 모든 파라미터를 학습 가능하도록 고정 해제
    #    model.gradient_checkpointing_enable()  # 그라디언트 체크포인트를 활성화하여 메모리 사용량 절감
    # model = model.train()  # 모델을 학습 모드로 설정

    # Lora 어댑터 구성 (LoRA는 학습 가능한 어댑터를 추가하는 방식 중 하나)
    lora_config = LoraConfig(
        r=16,                       # 랭크 값, 어댑터의 차원
        lora_alpha=32,               # 로라 알파, 학습 속도를 조절
        target_modules=["q_proj", "v_proj"],  # LoRA를 적용할 대상 모듈 (Q, V에 적용)
        lora_dropout=0.1,            # 드롭아웃 확률
        bias="none",                 # 바이어스를 학습할지 여부
        task_type="CAUSAL_LM"        # 작업 유형 (언어 모델)
    )


    # 어댑터를 모델에 추가
    model = get_peft_model(model, lora_config)

    # 어댑터가 추가되었는지 확인
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8, # batch_size를 조절할수록 그래픽 용량 많이 차지
        per_device_eval_batch_size=8, # batch_size를 조절할수록 그래픽 용량 많이 차지
        num_train_epochs=5, # epochs 수
        weight_decay=0.01,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_validation,
    )

    trainer.train()


    #####
    trainer.evaluate()



    ########
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_save = "D:/py_workspace/llama_finetuning/result/fine_tuned_{}.pth".format(current_time)
    tokenizer_save = "D:/py_workspace/llama_finetuning/result/tokenizer_{}.pth".format(current_time)

    model.save_pretrained(model_save)
    tokenizer.save_pretrained(tokenizer_save)
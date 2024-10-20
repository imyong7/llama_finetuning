from transformers import LlamaForCausalLM, AutoTokenizer
import torch

if __name__ == "__main__" :
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    input_text = 'Please change the pro c code below into java code.\
    #include <stdio.h> \
    int main() { \
        int rows = 5; \
        int cols = 5; \
        int array[5][5]; \
        // 이중 for문을 사용하여 배열 초기화 \
        for (int i = 0; i < rows; i++) { \
            for (int j = 0; j < cols; j++) { \
                array[i][j] = i * j;  // 배열의 값을 행과 열의 곱으로 설정 \
            } \
        } \
        // 이중 for문을 사용하여 배열 출력 \
        for (int i = 0; i < rows; i++) { \
            for (int j = 0; j < cols; j++) { \
                printf("%d ", array[i][j]); \
            } \
            printf("\n"); \
        } \
        return 0; \
    } \
    '

    print(input_text)


    use_model_name = "llama-3.2-3b"
    
    # token_path 경로 수정 필요
    token_path = "D:/py_workspace/llama_finetuning/model/{}/original".format(use_model_name)

    tokenizer = AutoTokenizer.from_pretrained(token_path, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token

    # model_path 경로 수정 필요
    model_path = "D:/py_workspace/llama_finetuning/model/{}".format(use_model_name)
    
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.to(device)


    # 입력 텍스트를 토큰화
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # 모델에 입력을 넣고 텍스트 생성
    output = model.generate(input_ids, max_length=10000, num_return_sequences=1, no_repeat_ngram_size=2)

    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Generated text:", generated_text)
    
    
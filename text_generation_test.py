import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 設定 ---
MODEL_NAME = "Qwen/Qwen3-0.6B-Base" 
INPUT_TXT = "東京大学の研究チームが新しいAI"
SAMPLING_METHOD = "top-k" # "greedy", "top-k", "top-p"
TEMPERATURE = 1
N_STEPS = 20
CHOICES_PER_STEP = 15 # top-kのk値、top-pの場合は上位何%までを候補にするか
PROBABILITY_THRESHOLD = 1.0 # top-p用,上位何%までを候補にするか

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    dtype=torch.bfloat16, 
    device_map="auto",
)

inputs = tokenizer(INPUT_TXT, return_tensors="pt").to(model.device)
input_len = inputs.input_ids.shape[1]

# generateの設定を整理
gen_config = {
    "max_new_tokens": N_STEPS,
    "return_dict_in_generate": True,
    "output_scores": True,
    "do_sample": (SAMPLING_METHOD != "greedy"),
    "temperature": TEMPERATURE,
}

if SAMPLING_METHOD == "top-k":
    gen_config["top_k"] = CHOICES_PER_STEP
elif SAMPLING_METHOD == "top-p":
    gen_config["top_p"] = PROBABILITY_THRESHOLD

# 生成実行
outputs = model.generate(**inputs, **gen_config)

generated_tokens = outputs.sequences[0]
scores = outputs.scores 

print(f"\nInitial input: {INPUT_TXT}")
print("-" * 50)

# 各ステップの分析
for step, logits in enumerate(scores):
    # 1. Temperatureを適用して確率分布を計算
    scaled_logits = logits[0] / TEMPERATURE
    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    
    # 2. 上位候補の取得
    top_probs, top_ids = torch.topk(probs, CHOICES_PER_STEP)
    
    # 3. 実際に選ばれたトークンの特定
    chosen_token_id = generated_tokens[input_len + step].item()
    chosen_token_str = tokenizer.decode([chosen_token_id])
    
    print(f"--- Step {step + 1} ---")
    # 文脈を表示 (直近のトークンのみ)
    current_context = tokenizer.decode(generated_tokens[:input_len + step], skip_special_tokens=True)
    print(f"  Context: ...{current_context[-30:]}")
    
    for i in range(CHOICES_PER_STEP):
        token_id = top_ids[i].item()
        prob = top_probs[i].item()
        token_str = tokenizer.decode([token_id], errors='replace') 
    
        marker = " ★SELECTED" if token_id == chosen_token_id else ""
        print(f"    Rank {i+1}: ID[{token_id:6}] -> '{token_str}' ({prob*100:.2f}%){marker}") 


print("\n" + "="*50)
print("✨ Final Output:")
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
print("="*50)
import os
import json
import datetime
import csv
import random
import argparse
import torch
import numpy as np
from rouge_score import rouge_scorer
from utils.janome_tokenizer import JanomeRougeTokenizer

# vLLMのインポート
from vllm import LLM, SamplingParams

# シードの固定
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ==========================================
# 設定
# ==========================================
TEST_FILE = "./data/test.jsonl"
SYSTEM_PROMPT_FILE = "system_prompt.txt"

# --- 生成パラメータ ---
MAX_NEW_TOKENS = 128      

# ==========================================
#  メイン処理
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="タイトル生成モデルの評価スクリプト")
    parser.add_argument("--model", type=str, default="sft_output", help="評価するモデルのパス")
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=SYSTEM_PROMPT_FILE,
        help="使用するシステムプロンプトファイルのパス",
    )
    args = parser.parse_args()

    model_name = args.model
    system_prompt_file = args.system_prompt_file

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ログディレクトリの作成
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    detail_log_file = os.path.join(log_dir, f"{timestamp}_details.jsonl")
    csv_log_file = "output.csv"

    print(f"🚀 Loading model with vLLM: {model_name}...")
    
    # 1. vLLMエンジンの初期化
    llm = LLM(
        model=model_name,
        dtype="float16",
        seed=seed
    )

    if not os.path.exists(TEST_FILE):
        print(f"❌ Test data {TEST_FILE} not found.")
        return
    if not os.path.exists(system_prompt_file):
        print(f"❌ System prompt {system_prompt_file} not found.")
        return

    with open(system_prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()

    print("Loading test dataset...")
    test_data = []
    prompts = []
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            test_data.append(record)
            # 事前にすべてのプロンプトを作成しておく
            prompt = f"{system_prompt}:\n{record['text']}\n答え:\n"
            prompts.append(prompt)

    # 2. サンプリングの設定 (HFのgen_configに相当)
    # do_sample=False (Greedy) は、vLLMでは temperature=0.0 に該当します
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=MAX_NEW_TOKENS
    )

    # Rouge評価器のセットアップ
    custom_tokenizer = JanomeRougeTokenizer(use_stemmer=True)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=custom_tokenizer)

    all_rouge_l_scores = []
    log_records = []

    print("Generating predictions... (vLLM will show its own progress bar)")
    
    # 3. vLLMによる一括生成
    # promptsのリストを丸ごと渡すだけで、自動的に最速の並列処理（Continuous Batching）を行います
    outputs = llm.generate(prompts, sampling_params)

    # 4. 結果の処理とスコア計算
    # outputsには生成結果がリストで返ってきます
    for i, output in enumerate(outputs):
        record = test_data[i]
        
        # vLLMは自動でプロンプト部分を削り、新しく生成したテキストだけを返してくれます
        prediction = output.outputs[0].text.strip()
        reference = record["target"].strip()

        # Rougeスコア計算
        score = scorer.score(reference, prediction)
        rouge_l_f1 = score['rougeL'].fmeasure
        all_rouge_l_scores.append(rouge_l_f1)

        # 詳細ログ用に保存
        log_records.append({
            "original_text": record["text"],
            "reference_title": reference,
            "predicted_title": prediction,
            "rouge_L_f1": rouge_l_f1
        })

    # ==========================================
    # 📊 結果の集計と保存
    # ==========================================
    mean_score = np.mean(all_rouge_l_scores)
    std_score = np.std(all_rouge_l_scores)

    print("\n" + "="*50)
    print("✨ Evaluation Complete (vLLM powered)!")
    print(f"Model: {model_name}")
    print(f"ROUGE-L F1: {mean_score:.4f} ± {std_score:.4f}")
    print("="*50)

    # 1. 詳細ログの保存 (Logディレクトリ内)
    with open(detail_log_file, 'w', encoding='utf-8') as f:
        for rec in log_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"📁 Detailed logs saved to: {detail_log_file}")

    # 2. output.csv への追記
    csv_exists = os.path.exists(csv_log_file)
    with open(csv_log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["Timestamp", "Model_Path", "ROUGE-L_Mean", "ROUGE-L_Std"])
        
        writer.writerow([timestamp, model_name, f"{mean_score:.4f}", f"{std_score:.4f}"])
    print(f"📁 Summary appended to: {csv_log_file}")

if __name__ == "__main__":
    main()

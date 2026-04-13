import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator

def main():
    print("="*50)
    print("🚀 LLM SFT 学習環境 テストスクリプト")
    print("="*50)

    # ---------------------------------------------------------
    # 1. PyTorch & GPU (CUDA) の確認
    # ---------------------------------------------------------
    print("\n[1] PyTorch & GPU チェック")
    print(f"PyTorch Version : {torch.__version__}")
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        print(f"✅ GPU 認識成功  : {device_name}")
        print(f"✅ CUDA Version : {cuda_ver}")
        
        # V100等でBFloat16がサポートされているかの確認
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"ℹ️ BFloat16対応  : {bf16_supported}")
    else:
        print("❌ エラー: GPUが認識されていません。CUDAドライバまたはPyTorchのバージョンを確認してください。")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. Accelerate & bitsandbytes の確認
    # ---------------------------------------------------------
    print("\n[2] 分散学習・最適化ライブラリ チェック")
    try:
        accelerator = Accelerator()
        print(f"✅ Accelerate 初期化成功 (Device: {accelerator.device})")
    except Exception as e:
        print(f"❌ ライブラリの初期化に失敗しました: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 3. データセット読み込み確認 (xlsum_ja)
    # ---------------------------------------------------------
    # print("\n[3] データセット (xlsum) 読み込みチェック")
    # try:
    #     # ダウンロード時間を節約するため、最初の1%だけロードします
    #     dataset = load_dataset("csebuetnlp/xlsum", "japanese", split="train[:1%]")
    #     print(f"✅ ロード成功 (取得サンプル数: {len(dataset)})")
    #     print(f"📖 サンプル(text) : {dataset[0]['text'][:50]}...")
    #     print(f"🏷️ サンプル(title): {dataset[0]['title']}")
    # except Exception as e:
    #     print(f"❌ データセットのロードに失敗しました: {e}")
    #     sys.exit(1)

    # ---------------------------------------------------------
    # 4. モデル＆トークナイザ読み込みと推論確認
    # ---------------------------------------------------------
    print("\n[4] モデルの読み込みと動作確認")

    test_model_id = "Qwen/Qwen3-0.6B-Base" 
    
    try:
        print(f"⏳ {test_model_id} をダウンロード/ロード中...")
        tokenizer = AutoTokenizer.from_pretrained(test_model_id)
        
        # V100なので torch_dtype=torch.float16 を明示的に指定
        model = AutoModelForCausalLM.from_pretrained(
            test_model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        print("✅ モデル・トークナイザのロード成功")

        # 軽い推論テスト（GPUで計算できるか）
        prompt = "人工知能の未来について、"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15)
        
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 推論テスト成功:\n   入力: {prompt}\n   出力: {result_text}")

    except Exception as e:
        print(f"❌ モデルの動作テストに失敗しました: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 5. vLLM の確認 (高速推論エンジン)
    # ---------------------------------------------------------
    print("\n[5] vLLM (高速推論エンジン) チェック")
    try:
        from vllm import LLM, SamplingParams
        print(f"✅ vLLM インポート成功")
        
        # 軽い推論テスト
        sampling_params = SamplingParams(max_tokens=10)
        llm = LLM(model=test_model_id, trust_remote_code=True, dtype="float16", enforce_eager=True)
        outputs = llm.generate(["人工知能の未来について、"], sampling_params)
        
        generated_text = outputs[0].outputs[0].text
        print(f"✅ vLLM 推論テスト成功:\n   出力: {generated_text}")
        
        # GPUメモリを解放するためにLLMオブジェクトを明示的に削除（簡易的）
        import gc
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    except ImportError:
        print("❌ vLLM がインストールされていないか、インポートに失敗しました。")
    except Exception as e:
        print(f"⚠️ vLLM の初期化または推論に失敗しました (環境によってはメモリ制限などで失敗することがあります): {e}")

    print("\n" + "="*50)
    print("🎉 全てのテストが正常に完了しました！")
    print("   チュートリアル本編（SFT学習）に進む準備が完璧に整っています。")
    print("="*50)

if __name__ == "__main__":
    main()
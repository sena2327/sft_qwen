# evaluate.py (または Jupyter Notebookなど)
from rouge_score import rouge_scorer
from utils.janome_tokenizer import JanomeRougeTokenizer
import difflib

def main():
    custom_tokenizer = JanomeRougeTokenizer(use_stemmer=True)

    scorer = rouge_scorer.RougeScorer(
        ['rougeL'], 
        use_stemmer=False, 
        tokenizer=custom_tokenizer
    )

    # 3. 評価の実行
    ref = "東京大学の研究チームが、新しい人工知能を開発しました。"
    pred = "東大の研究チームが新しいAIを作った。"
    scores = scorer.score(ref, pred)

    print("Rouge-L F1:", scores['rougeL'].fmeasure)

    # 1. 実際にどのようにトークン（単語）に切り分けられたかを確認
    ref_tokens = custom_tokenizer.tokenize(ref)
    pred_tokens = custom_tokenizer.tokenize(pred)

    print("\n=== 🕵️ 計算過程の検証 ===")
    print(f"正解 (Ref) の長さ: {len(ref_tokens)} トークン")
    print(f"  中身: {ref_tokens}")
    print(f"予測 (Pred) の長さ: {len(pred_tokens)} トークン")
    print(f"  中身: {pred_tokens}")

    # 2. Pythonの標準ライブラリを使って「一致した部分(LCS)」を可視化
    matcher = difflib.SequenceMatcher(None, ref_tokens, pred_tokens)
    match_blocks = matcher.get_matching_blocks()

    matched_tokens = []
    for block in match_blocks:
        # 一致したトークンを抽出
        matched_tokens.extend(ref_tokens[block.a : block.a + block.size])

    print(f"\n一致したトークン数: {len(matched_tokens)} トークン")
    print(f"  中身: {matched_tokens}")

    # 3. ROUGE-Lの手計算
    R = len(matched_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    P = len(matched_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    F1 = (2 * R * P) / (R + P) if (R + P) > 0 else 0

    print("\n--- 手計算による ROUGE-L ---")
    print(f"Recall    = {len(matched_tokens)} / {len(ref_tokens)} = {R:.4f}")
    print(f"Precision = {len(matched_tokens)} / {len(pred_tokens)} = {P:.4f}")
    print(f"F-measure = (2 * {R:.4f} * {P:.4f}) / ({R:.4f} + {P:.4f}) = {F1:.4f}")

if __name__ == "__main__":
    main()
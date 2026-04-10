from janome.tokenizer import Tokenizer as JanomeAnalyzer
from rouge_score.tokenizers import Tokenizer

class JanomeRougeTokenizer(Tokenizer):
    """rouge-score用のJanomeを用いたTokenizerクラス"""
    
    def __init__(self, use_stemmer=False):
        self.use_stemmer = use_stemmer
        self.analyzer = JanomeAnalyzer()

    def tokenize(self, text):
        text = text.strip()
        if not text:
            return []

        tokens = []
        # textを解析して、1単語ずつ処理
        for token in self.analyzer.tokenize(text):
            if self.use_stemmer:
                # 原形を取得。未知語などで '*' の場合はそのままの文字を使う
                base_form = token.base_form if token.base_form != "*" else token.surface
                tokens.append(base_form)
            else:
                # 単純な分かち書き（そのままの文字）
                tokens.append(token.surface)
                
        return tokens
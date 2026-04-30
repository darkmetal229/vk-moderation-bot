import re
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

SPAM_PATTERNS = [
    r"(купи[тье]?|заказ(ать|ывай)|скидк[аи]|акци[яи]|бесплатно|дёшево)",
    r"(t\.me|telegram|whatsapp|viber|instagram)[\s:/]",
    r"(подпишись|подписывайся|лайк|репост)",
    r"(http|https|www)\S+",
    r"(.)\1{4,}",
]

NEGATIVE_PATTERNS = [
    r"(ужас|отстой|кошмар|позор|бред|чушь)",
    r"(ненавиж|бесит|раздражает|достал[аи]?)",
    r"(плохо|плохой|хуже|худший|отвратительн)",
]

NEGATION_WORDS = ["не", "нет", "никогда", "без"]

@dataclass
class AnalysisResult:
    spam_score: float
    negative_score: float
    verdict: str
    method: str
    details: str

class TextAnalyzer:
    def __init__(self, spam_threshold: float = 0.7, negative_threshold: float = 0.65):
        self.spam_threshold = spam_threshold
        self.negative_threshold = negative_threshold
        self._blacklist: set = set()
        self._ml_model = None
        self._try_load_ml()

    def _try_load_ml(self):
        try:
            from train_model import MLModel
            model = MLModel()
            if model.load():
                self._ml_model = model
                logger.info("✅ Random Forest модель загружена в анализатор")
        except Exception as e:
            logger.warning(f"⚠️ ML модель не загружена: {e}")

    def update_thresholds(self, spam: float, negative: float):
        self.spam_threshold = spam
        self.negative_threshold = negative

    def update_blacklist(self, words: list):
        self._blacklist = {w.lower().strip() for w in words}

    def analyze(self, text: str) -> AnalysisResult:
        clean = text.lower().strip()

        for word in self._blacklist:
            if word in clean:
                return AnalysisResult(
                    spam_score=1.0, negative_score=0.5,
                    verdict="spam", method="blacklist",
                    details=f"Слово из чёрного списка: '{word}'"
                )

        if self._ml_model is not None:
            try:
                ml_result = self._ml_model.predict(text)
                if ml_result:
                    probs = ml_result["probabilities"]
                    spam_score = probs.get("spam", 0.0)
                    neg_score = probs.get("negative", 0.0)
                    ok_score = probs.get("ok", 0.0)
                    verdict = self._decide_verdict(spam_score, neg_score)
                    return AnalysisResult(
                        spam_score=round(spam_score, 3),
                        negative_score=round(neg_score, 3),
                        verdict=verdict,
                        method="random_forest",
                        details=(
                            f"RF: spam={spam_score:.2f} "
                            f"neg={neg_score:.2f} "
                            f"ok={ok_score:.2f} "
                            f"conf={ml_result['confidence']:.2f}"
                        )
                    )
            except Exception as e:
                logger.warning(f"⚠️ Ошибка ML предсказания: {e}")

        return self._rule_analyze(clean, text)

    def _rule_analyze(self, clean: str, original: str) -> AnalysisResult:
        spam_score = self._score_patterns(clean, SPAM_PATTERNS)
        neg_score = self._score_patterns(clean, NEGATIVE_PATTERNS)

        for neg_word in NEGATION_WORDS:
            if neg_word in clean.split():
                neg_score *= 0.4
                break

        verdict = self._decide_verdict(spam_score, neg_score)
        return AnalysisResult(
            spam_score=round(spam_score, 3),
            negative_score=round(neg_score, 3),
            verdict=verdict, method="rules",
            details=f"Rule-based: spam={spam_score:.2f} neg={neg_score:.2f}"
        )

    def _score_patterns(self, text: str, patterns: list) -> float:
        hits = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        return min(hits / len(patterns) * 2.5, 1.0)

    def _decide_verdict(self, spam: float, negative: float) -> str:
        if spam >= self.spam_threshold:
            return "spam"
        if negative >= self.negative_threshold:
            return "negative"
        if spam > self.spam_threshold * 0.6 or negative > self.negative_threshold * 0.6:
            return "pending"
        return "ok"

_analyzer: Optional[TextAnalyzer] = None

def get_analyzer() -> TextAnalyzer:
    global _analyzer
    if _analyzer is None:
        from config import settings
        _analyzer = TextAnalyzer(
            spam_threshold=settings.spam_threshold,
            negative_threshold=settings.negative_threshold,
        )
    return _analyzer

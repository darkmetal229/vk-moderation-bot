"""
Загрузка предобученной Random Forest модели.
Модель должна быть предварительно обучена локально и сохранена как rf_model.joblib
"""

import os
import joblib
import logging
import re
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Функция построения признаков (ДОЛЖНА СОВПАДАТЬ С ТРЕНИРОВОЧНОЙ)
# ============================================================
def build_features(text: str) -> dict:
    """Ручные признаки для улучшения Random Forest."""
    t = text.lower()
    return {
        "len": len(text),
        "words": len(text.split()),
        "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "url_count": len(re.findall(r"https?://\S+|www\.\S+", t)),
        "excl_count": text.count("!"),
        "has_phone": int(bool(re.search(r"\+7|8\s?\(?\d{3}", t))),
        "has_tg": int(bool(re.search(r"t\.me|telegram", t))),
        "has_promo": int(bool(re.search(r"скидк|акци|купи|заказ|бесплатно", t))),
        "repeat_chars": int(bool(re.search(r"(.)\1{3,}", t))),
        "digit_ratio": sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        "has_negative": int(bool(re.search(r"ужас|отстой|кошмар|ненавиж|бесит|плохо|позор", t))),
        "emoji_count": len(re.findall(r"[😀-🙏🌀-🗿]", text)),
    }


class MLModel:
    """Random Forest классификатор — только загрузка и предсказание."""
    
    def __init__(self, model_path: str = "rf_model.joblib"):
        self.model_path = model_path
        self.clf = None
        self.tfidf = None
        self.label_encoder = None
        self.is_loaded = False

    def load(self) -> bool:
        """Загружает предобученную модель из файла."""
        if not os.path.exists(self.model_path):
            logger.warning(f"⚠️ Файл модели {self.model_path} не найден!")
            logger.warning("   Запустите train_model.py локально для обучения модели")
            return False
        
        try:
            data = joblib.load(self.model_path)
            self.clf = data["clf"]
            self.tfidf = data["tfidf"]
            self.label_encoder = data["label_encoder"]
            self.is_loaded = True
            logger.info(f"✅ Модель загружена: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False

    def predict(self, text: str) -> dict:
        """Предсказывает класс для одного текста."""
        if not self.is_loaded:
            return None
        
        import scipy.sparse as sp
        
        # Векторизация текста
        tfidf_f = self.tfidf.transform([text])
        hand_f = np.array([list(build_features(text).values())])
        X = sp.hstack([tfidf_f, hand_f])
        
        # Предсказание
        proba = self.clf.predict_proba(X)[0]
        classes = self.label_encoder.classes_
        
        result = {cls: float(prob) for cls, prob in zip(classes, proba)}
        predicted = classes[np.argmax(proba)]
        
        return {
            "verdict": predicted,
            "probabilities": result,
            "confidence": float(np.max(proba))
        }


def get_model() -> MLModel:
    """Создаёт и загружает модель (синглтон)."""
    model = MLModel()
    if model.load():
        return model
    return None


if __name__ == "__main__":
    # Проверка загрузки модели
    model = get_model()
    if model:
        print("✅ Модель готова к использованию")
        # Тестовое предсказание
        test_text = "Купи iPhone дёшево!"
        result = model.predict(test_text)
        print(f"Тест: '{test_text}' → {result}")
    else:
        print("❌ Модель не найдена. Сначала обучите её локально.")
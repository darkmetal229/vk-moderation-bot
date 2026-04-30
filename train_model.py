"""
Тренировка Random Forest на датасете с комментариями из HuggingFace.
"""

import re
import os
import json
import joblib
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUSSIAN_EXAMPLES = {
    "spam": [
        "Купи сейчас! Лучшие цены в городе скидка 50%",
        "Переходи по ссылке t.me/casino_win и выигрывай",
        "Подпишись на мой канал telegram получи бонус",
        "АКЦИЯ! Купить дёшево iPhone 15 без предоплаты",
        "Заработай 100000 рублей в день без вложений!",
        "Репост этого поста и выиграй приз подписывайся",
        "http://bit.ly/free-iphone кликай сюда бесплатно",
        "Лучший онлайн казино вывод за 5 минут регистрация",
        "Работа на дому 500 рублей за час без опыта",
        "Продаю аккаунты ВКонтакте оптом дёшево пишите",
        "БЕСПЛАТНО получи курс по трейдингу ссылка в профиле",
        "Скидка 70% на всё только сегодня торопись",
        "Инвестиции в крипту гарантированный доход 300%",
        "Дешёвые подписчики лайки репосты раскрутка",
        "Займи до 100000 без отказа онлайн за 5 минут",
    ],
    "negative": [
        "Это просто ужас как можно так работать",
        "Отвратительный сервис никогда сюда не вернусь",
        "Ненавижу этот университет полный бред и позор",
        "Всё плохо, расписание ужасное, преподы отстой",
        "Кошмарная столовая, еда отвратительная, невозможно есть",
        "Бред какой-то, эти правила бесят до невозможности",
        "Худший университет в области, стыдно признаваться",
        "Достала эта очередь в деканате, ужасная организация",
        "Раздражает что никто не отвечает на вопросы",
        "Плохое качество обучения, жалею что поступил сюда",
        "Омерзительно как относятся к студентам здесь",
        "Позорище, третий раз переносят защиту диплома",
        "Ненавижу эту систему записи невозможно попасть",
        "Всё что делает администрация — это отстой полный",
        "Ужасная общага, условия жизни просто кошмар",
    ],
    "ok": [
        "Спасибо за информацию, очень полезно",
        "Когда будет следующее мероприятие в университете?",
        "Отличная новость! Поздравляю всех студентов",
        "Подскажите пожалуйста расписание экзаменов",
        "Хорошая работа преподавательского состава",
        "Интересная лекция, спасибо профессору",
        "Где можно узнать о стипендиальной программе?",
        "Университет проводит день открытых дверей когда?",
        "Замечательный праздник получился у всех",
        "Рад что поступил в ВГТУ, хорошее образование",
        "Спортивные соревнования были очень интересными",
        "Нормально, всё понятно и доступно объяснено",
        "Хотел бы узнать про программу обмена студентами",
        "Молодцы организаторы мероприятия, всё прошло хорошо",
        "Отличная команда КВН, болею за вас!",
    ],
}

def load_huggingface_data():
    texts, labels = [], []
    try:
        from datasets import load_dataset
        logger.info("📦 Загружаем датасет с HuggingFace...")

        try:
            ds = load_dataset("mteb/amazon_reviews_multi", "ru",
                              split="train[:3000]", trust_remote_code=True)
            for item in ds:
                text = item.get("text", "") or item.get("review_body", "")
                stars = item.get("label", item.get("stars", 3))
                if not text:
                    continue
                label = "negative" if stars <= 2 else "ok"
                texts.append(text)
                labels.append(label)
            logger.info(f"✅ amazon_reviews_multi: {len(texts)} примеров")
        except Exception as e:
            logger.warning(f"⚠️ amazon_reviews_multi недоступен: {e}")
            try:
            before = len(texts)
            ds2 = load_dataset("tweet_eval", "hate",
                               split="train[:2000]", trust_remote_code=True)
            for item in ds2:
                text = item.get("text", "")
                label_id = item.get("label", 0)
                if not text:
                    continue
                label = "negative" if label_id == 1 else "ok"
                texts.append(text)
                labels.append(label)
            logger.info(f"✅ tweet_eval/hate: {len(texts)-before} примеров")
        except Exception as e:
            logger.warning(f"⚠️ tweet_eval/hate недоступен: {e}")

        try:
            before = len(texts)
            ds3 = load_dataset("ucirvine/sms_spam",
                               split="train[:2000]", trust_remote_code=True)
            for item in ds3:
                text = item.get("sms", "")
                label_id = item.get("label", 0)
                if not text:
                    continue
                label = "spam" if label_id == 1 else "ok"
                texts.append(text)
                labels.append(label)
            logger.info(f"✅ sms_spam: {len(texts)-before} примеров")
        except Exception as e:
            logger.warning(f"⚠️ sms_spam недоступен: {e}")

    except ImportError:
        logger.warning("⚠️ datasets не установлен, используем только встроенные данные")

    return texts, labels

def build_features(text: str) -> dict:
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
    def __init__(self, model_path: str = "rf_model.joblib"):
        self.model_path = model_path
        self.pipeline = None
        self.label_encoder = None
        self.is_loaded = False

    def _combine_features(self, texts):
        import scipy.sparse as sp
        tfidf_features = self.tfidf.transform(texts)
        hand_features = np.array([list(build_features(t).values()) for t in texts])
        return sp.hstack([tfidf_features, hand_features])

    def train(self, texts, labels):
        logger.info(f"🏋️ Обучение Random Forest на {len(texts)} примерах...")

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=30000,
            sublinear_tf=True,
            min_df=1,
        )
        self.tfidf.fit(texts)

        import scipy.sparse as sp
        tfidf_features = self.tfidf.transform(texts)
        hand_features = np.array([list(build_features(t).values()) for t in texts])
        X = sp.hstack([tfidf_features, hand_features])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        self.clf.fit(X_train, y_train)

        y_pred = self.clf.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,output_dict=True
        )
        logger.info("📊 Результаты:\n" + classification_report(
            y_test, y_pred, target_names=self.label_encoder.classes_))

        self.is_loaded = True
        return report

    def predict(self, text: str) -> dict:
        if not self.is_loaded:
            return None
        import scipy.sparse as sp
        tfidf_f = self.tfidf.transform([text])
        hand_f = np.array([list(build_features(text).values())])
        X = sp.hstack([tfidf_f, hand_f])
        proba = self.clf.predict_proba(X)[0]
        classes = self.label_encoder.classes_
        result = {cls: float(prob) for cls, prob in zip(classes, proba)}
        predicted = classes[np.argmax(proba)]
        return {"verdict": predicted, "probabilities": result, "confidence": float(np.max(proba))}

    def save(self):
        data = {
            "clf": self.clf,
            "tfidf": self.tfidf,
            "label_encoder": self.label_encoder,
        }
        joblib.dump(data, self.model_path)
        logger.info(f"💾 Модель сохранена: {self.model_path}")

    def load(self):
        if not os.path.exists(self.model_path):
            return False
        data = joblib.load(self.model_path)
        self.clf = data["clf"]
        self.tfidf = data["tfidf"]
        self.label_encoder = data["label_encoder"]
        self.is_loaded = True
        logger.info(f"✅ Модель загружена: {self.model_path}")
        return True

def train_and_save():
    hf_texts, hf_labels = load_huggingface_data()

    ru_texts, ru_labels = [], []
    multiplier = max(1, len(hf_texts) // (len(RUSSIAN_EXAMPLES["spam"]) * 3 * 5))
    for label, examples in RUSSIAN_EXAMPLES.items():
        for _ in range(multiplier + 3):
            ru_texts.extend(examples)
            ru_labels.extend([label] * len(examples))

    all_texts = hf_texts + ru_texts
    all_labels = hf_labels + ru_labels

    logger.info(f"📊 Итого данных: {len(all_texts)} примеров")
    from collections import Counter
    logger.info(f"   Распределение: {Counter(all_labels)}")

    model = MLModel()
    report = model.train(all_texts, all_labels)
    model.save()

    with open("ml_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(all_texts),
            "distribution": dict(Counter(all_labels)),
            "report": report,
        }, f, ensure_ascii=False, indent=2)
    logger.info("✅ Метрики сохранены в ml_metrics.json")
    return model

if __name__ == "__main__":
    train_and_save()

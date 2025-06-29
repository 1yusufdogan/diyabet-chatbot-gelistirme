import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import numpy as np

class DiabetesModelBERT:
    def __init__(self):
        load_dotenv()
        # CSV dosya yolunu data klasöründen oku
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'diabetes_chatbot_dataset_varied.csv')
        self.dataset = pd.read_csv(csv_path)

        # Intent örneklerini ve yanıtlarını hazırla
        self.intent_examples = {}
        self.intent_responses = {}
        all_examples = []
        for intent in self.dataset['Intent'].unique():
            examples = self.dataset[self.dataset['Intent'] == intent]['Example'].tolist()
            responses = self.dataset[self.dataset['Intent'] == intent]['Response'].tolist()
            self.intent_examples[intent] = examples
            self.intent_responses[intent] = responses
            all_examples.extend(examples)

        # TF-IDF vektörizeri eğit
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2))
        self.vectorizer.fit(all_examples)
        self.intent_vectors = {
            intent: self.vectorizer.transform(examples)
            for intent, examples in self.intent_examples.items()
        }

    def normalize(self, text):
        # Küçük harfe çevir, noktalama işaretlerini at
        return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

    def find_best_match(self, query, threshold=0.3):
        norm = self.normalize(query)
        q_vec = self.vectorizer.transform([norm])
        best_intent, best_score = None, 0.0
        for intent, vecs in self.intent_vectors.items():
            sim = cosine_similarity(q_vec, vecs).max()
            if sim > best_score:
                best_score, best_intent = sim, intent
        return best_intent if best_score >= threshold else None

    def get_response(self, query):
        intent = self.find_best_match(query)
        if intent:
            # O intent için rastgele bir yanıt seç
            return np.random.choice(self.intent_responses[intent])
        # Eşleşme yoksa kullanıcıya yeniden sor
        return "Üzgünüm, bu soruyu anlayamadım. Lütfen farklı ifade edin."

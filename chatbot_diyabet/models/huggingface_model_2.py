import os
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DiabetesModelT5:
    def __init__(self):
        load_dotenv()

        self.dataset = pd.read_csv("data/diabetes_chatbot_dataset_varied.csv")

        self.intent_examples = {}
        self.intent_responses = {}
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))

        all_examples = []
        for intent in self.dataset['Intent'].unique():
            examples = self.dataset[self.dataset['Intent'] == intent]['Example'].fillna('').tolist()
            responses = self.dataset[self.dataset['Intent'] == intent]['Response'].fillna('').tolist()
            self.intent_examples[intent] = examples
            self.intent_responses[intent] = responses
            all_examples.extend(examples)

        self.vectorizer.fit(all_examples)

        self.intent_vectors = {
            intent: self.vectorizer.transform(examples)
            for intent, examples in self.intent_examples.items()
        }

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        self.generator = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def normalize(self, text):
        return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

    def find_best_match(self, query):
        norm_query = self.normalize(query)
        query_vector = self.vectorizer.transform([norm_query])

        best_intent = None
        best_score = 0

        for intent, vectors in self.intent_vectors.items():
            similarities = cosine_similarity(query_vector, vectors).flatten()
            max_sim = np.max(similarities)
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent

        return best_intent if best_score > 0.3 else None

    def get_response(self, query):
        intent = self.find_best_match(query)
        if intent:
            prompt = f"Diyabet chatbotu olarak '{intent}' niyetine yanit ver: {query}"
            result = self.generator(prompt, max_length=100, do_sample=True)[0]["generated_text"]
            return result
        else:
            return "Uzgunum, sorunu anlamadim. Lutfen tekrar sorar misiniz."
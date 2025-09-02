import pandas as pd
import string
import nltk
from django.http import JsonResponse

SMS_DATASETS = "sms-datasets.txt"

# Ensure nltk resources are available
def setup_nltk():
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords"
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

def read_dataset_file():
    data = pd.read_csv(SMS_DATASETS, sep="\t", header=None, names=["label", "sms"])
    return data

def remove_punctuation_and_stopwords():
    setup_nltk()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation)
    data = read_dataset_file()

    def clean_text(text):
        txt = text.lower()
        tokens = nltk.word_tokenize(txt)
        tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
        return " ".join(tokens)

    data["clean_sms"] = data["sms"].apply(clean_text)
    return data

def categorize_word(data):
    spam_data = []
    ham_data = []

    for sms in data[data["label"] == "spam"]["clean_sms"]:
        spam_data.extend(sms.split())
    for sms in data[data["label"] == "ham"]["clean_sms"]:
        ham_data.extend(sms.split())
    return spam_data, ham_data

def predict_message(user_input, spam_words, ham_words):
    spam_counter = sum(spam_words.count(word) for word in user_input.lower().split())
    ham_counter = sum(ham_words.count(word) for word in user_input.lower().split())

    if ham_counter > spam_counter:
        accuracy = (ham_counter / (ham_counter + spam_counter)) * 100
        result = {"prediction": "ham", "accuracy": round(accuracy, 2)}
    elif spam_counter > ham_counter:
        accuracy = (spam_counter / (ham_counter + spam_counter)) * 100
        result = {"prediction": "spam", "accuracy": round(accuracy, 2)}
    else:
        result = {"prediction": "unknown", "accuracy": 0.0}

    return result

# Preprocess dataset once at server start
cleaned_data = remove_punctuation_and_stopwords()
spam_words, ham_words = categorize_word(cleaned_data)
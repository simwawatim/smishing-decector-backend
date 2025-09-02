import os
import string
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from datasets.models import SMSMessage
import pandas as pd
import nltk
from nltk.tokenize import TreebankWordTokenizer

BASE_DIR = os.path.dirname(__file__)
SMS_DATASETS = os.path.join(BASE_DIR, "sms-datasets.txt")

# ------------------- VIEWS -------------------

def home(request):
    total_messages = SMSMessage.objects.count()
    scam_messages = SMSMessage.objects.filter(label='scam').count()
    ham_messages = SMSMessage.objects.filter(label='ham').count()
    context = {
        'total_messages': total_messages,
        'scam_messages': scam_messages,
        'ham_messages': ham_messages,
    }
    return render(request, 'datasets/home.html', context)


def update_data_sets(request):
    return render(request, 'datasets/update.html')


def create_datasets(request):
    if request.method == "POST":
        msg_text = request.POST.get('message')
        label = request.POST.get('label')

        if not msg_text or not label:
            messages.warning(request, 'Both message and label are required.')
            return redirect('datasets:update_data_sets')

        SMSMessage.objects.create(message=msg_text, label=label)
        messages.success(request, 'Message added successfully!')
        return redirect('datasets:update_data_sets')


def messages_list(request):
    total_messages = SMSMessage.objects.count()
    scam_messages = SMSMessage.objects.filter(label='scam').count()
    ham_messages = SMSMessage.objects.filter(label='ham').count()
    recent_messages = SMSMessage.objects.order_by('-created_at')[:50]

    context = {
        'total_messages': total_messages,
        'scam_messages': scam_messages,
        'ham_messages': ham_messages,
        'recent_messages': recent_messages,
    }
    return render(request, 'datasets/messages.html', context)


# ------------------- NLTK PREPROCESSING -------------------

def setup_nltk():
    # Download stopwords if not already present
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


def read_dataset_file():
    data = pd.read_csv(SMS_DATASETS, sep="\t", header=None, names=["label", "sms"])
    return data


def remove_punctuation_and_stopwords():
    setup_nltk()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokenizer = TreebankWordTokenizer()  # safer than punkt
    data = read_dataset_file()

    def clean_text(text):
        txt = str(text).lower()
        tokens = tokenizer.tokenize(txt)
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
    words = user_input.lower().split()
    spam_counter = sum(spam_words.count(word) for word in words)
    ham_counter = sum(ham_words.count(word) for word in words)

    if ham_counter > spam_counter:
        accuracy = (ham_counter / (ham_counter + spam_counter)) * 100
        result = {"prediction": "ham", "accuracy": round(accuracy, 2)}
    elif spam_counter > ham_counter:
        accuracy = (spam_counter / (ham_counter + spam_counter)) * 100
        result = {"prediction": "spam", "accuracy": round(accuracy, 2)}
    else:
        result = {"prediction": "unknown", "accuracy": 0.0}

    return result


# ------------------- PRELOAD DATA -------------------

# Preprocess dataset once when server starts
cleaned_data = remove_punctuation_and_stopwords()
spam_words, ham_words = categorize_word(cleaned_data)

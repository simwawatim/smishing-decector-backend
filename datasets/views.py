import os
import string
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from datasets.models import SMSMessage
from django.views.decorators.csrf import csrf_exempt
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

# ------------------- IMPORTS -------------------
import re
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import SMSMessage

# ------------------- PREPROCESSING -------------------
def preprocess_bemba_datasets():
    """
    Load Bemba messages from the SMSMessage model and clean the text.
    Returns a list of dictionaries with 'label' and 'message_cleaned'.
    """
    messages = SMSMessage.objects.all().values("label", "message")  # Fetch all messages
    cleaned_data = []

    for msg in messages:
        text = str(msg["message"]).lower()
        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
        text = text.strip()
        cleaned_data.append({
            "label": msg["label"].lower(),
            "message_cleaned": text
        })

    return cleaned_data

# ------------------- WORD CATEGORIZATION -------------------
def categorize_bemba_words(cleaned_data):
    """
    Create lists of scam and ham words from dataset.
    """
    scam_words = set()
    ham_words = set()
    
    for row in cleaned_data:
        words = row["message_cleaned"].split()
        if row["label"] == "scam":
            scam_words.update(words)
        else:  # 'ham'
            ham_words.update(words)
    
    return list(scam_words), list(ham_words)

# ------------------- PREDICTION -------------------
def predict_bemba_message(message, scam_words, ham_words):
    """
    Predicts whether a Bemba message is scam or ham.
    Returns prediction and accuracy.
    """
    message_clean = message.lower()
    message_clean = re.sub(r"[^\w\s]", "", message_clean)
    words = message_clean.split()
    
    if not words:
        return {"prediction": "unknown", "accuracy": 0.0}
    
    scam_hits = [w for w in words if w in scam_words]
    ham_hits = [w for w in words if w in ham_words]
    
    if len(scam_hits) > len(ham_hits):
        prediction = "scam"
        accuracy = len(scam_hits) / len(words)
    elif len(ham_hits) > len(scam_hits):
        prediction = "ham"
        accuracy = len(ham_hits) / len(words)
    else:
        prediction = "unknown"
        accuracy = 0.0
    
    return {"prediction": prediction, "accuracy": round(accuracy, 2)}

# ------------------- PRELOAD BEMBA DATA -------------------
# Load once when server starts
bemba_cleaned_data = preprocess_bemba_datasets()
bemba_scam_words, bemba_ham_words = categorize_bemba_words(bemba_cleaned_data)

# ------------------- API ENDPOINT -------------------
@csrf_exempt
def predict_bemba_api(request):
    if request.method == "POST":
        user_input = request.POST.get("message", "")
        if not user_input:
            return JsonResponse({"error": "Message is required"}, status=400)
        
        result = predict_bemba_message(
            user_input,
            bemba_scam_words,
            bemba_ham_words
        )
        return JsonResponse(result)
    
    return JsonResponse({"error": "Only POST allowed"}, status=405)

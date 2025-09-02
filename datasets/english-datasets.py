import pandas as pd
import string
import nltk

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
        for word in sms.split():
            spam_data.append(word)

    for sms in data[data["label"] == "ham"]["clean_sms"]:
        for word in sms.split():
            ham_data.append(word)

    return spam_data, ham_data


def predict(user_input, spam_words, ham_words):
    spam_counter = 0
    ham_counter = 0

    for word in user_input.lower().split():
        spam_counter += spam_words.count(word)
        ham_counter += ham_words.count(word)

    print("\n***************************** Result ****************")

    if ham_counter > spam_counter:
        accuracy = (ham_counter / (ham_counter + spam_counter)) * 100
        print(f"Prediction: Ham  |  Accuracy: {accuracy:.2f}%")
    elif spam_counter > ham_counter:
        accuracy = (spam_counter / (ham_counter + spam_counter)) * 100
        print(f"Prediction: Spam |  Accuracy: {accuracy:.2f}%")
    else:
        print("Could not determine (tie)")


if __name__ == "__main__":
    cleaned_data = remove_punctuation_and_stopwords()
    spam_words, ham_words = categorize_word(cleaned_data)

    print("=== SMS Spam Detector ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter your message: ")
        if user_input.lower() == "exit":
            print("Goodbye! 👋")
            break
        predict(user_input, spam_words, ham_words)

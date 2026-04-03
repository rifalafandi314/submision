import pandas as pd
import re
import os
from nltk.corpus import stopwords
from imblearn.over_sampling import RandomOverSampler

stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

def label_sentiment(score):
    if score <= 2:
        return 0
    elif score == 3:
        return 1
    else:
        return 2

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df[['content', 'score']]
    df.dropna(inplace=True)

    df.rename(columns={'content': 'review'}, inplace=True)

    df['label'] = df['score'].apply(label_sentiment)

    df['clean_text'] = df['review'].apply(clean_text)

    X = df['clean_text']
    y = df['label']

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X.values.reshape(-1,1), y)

    df_balanced = pd.DataFrame({
        'clean_text': X_res.flatten(),
        'label': y_res
    })

    # buat folder output jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_balanced.to_csv(output_path, index=False)
    print("Preprocessing selesai!")

if __name__ == "__main__":
    preprocess(
        '../dataset_raw/instagram_reviews.csv',
        'dataset_preprocessing/data_clean.csv'
    )
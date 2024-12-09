import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from joblib import load
from nltk.corpus import stopwords
from textblob import TextBlob
import re
import nltk

nltk.download('stopwords') 

# Streamlit 应用
st.title("Podcast Episode Recommendation")

# 用户输入
episode_id = st.text_input("Enter Spotify Episode ID:", value="")

# 定义主要函数
def process_new_sample(episode_id):
    try:
        # 设置客户端凭证（client_id 和 client_secret）
        client_id = "a728a5852093450cacb79d3106c5002f"
        client_secret = "d9b1e1a9b0b7401e96163b2330ae2a46"

        # 使用 client credentials flow 进行身份验证
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        # 获取 episode 信息
        episode = sp.episode(episode_id)

        # 提取 episode 信息
        podcast_description = episode['show']['description']
        episode_description = episode['description']
        episode_duration_ms = episode['duration_ms']
        combined_description = podcast_description + episode_description

        # 加载模型和数据
        lda_model = gensim.models.LdaModel.load('lda_model.gensim')
        dictionary = lda_model.id2word
        scaled_features = pd.read_csv('reduced.csv', header=0)
        scaled_features = np.array(scaled_features)[:, 1:3]
        df = pd.read_csv('result.csv')
        data = df.iloc[:, 0:9]
        stop_words = set(stopwords.words('english'))
        custom_stopwords = stop_words.union({
            'podcast', 'episode', 'show', 'episodes', '’', '“', '”', 'us', 
            'one', 'get', 'new', 'week', 'every', 'join', 'like', 'also',
            'first', 'free', 'today', 'ad', 'make', 'find', 'day', 'go',
            'use', 'daily', 'instagram', 'youtube',  'hosted', "—", "n't", 
            "ad-free", "—", "com", "podcasts", "–", "?", 'support'
        })

        # 定义文本处理函数
        def remove_emoji_and_urls(text):
            if not isinstance(text, str):
                return text
            text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
            emoji_pattern = re.compile(
                "[" + 
                u"\U0001F600-\U0001F64F" + u"\U0001F300-\U0001F5FF" + 
                u"\U0001F680-\U0001F6FF" + u"\U0001F700-\U0001F77F" + 
                u"\U0001F780-\U0001F7FF" + "]", flags=re.UNICODE)
            return emoji_pattern.sub(r"", text)

        def clean_text(text, stop_words):
            text = remove_emoji_and_urls(text)
            text = re.sub(r'[^\w\s]', '', text.lower())
            tokens = [word for word in text.split() if word not in stop_words]
            return ' '.join(tokens)

        def extract_sentiment_features(text):
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            return polarity, subjectivity

        # 清理文本并提取特征
        cleaned_description = clean_text(combined_description, stop_words)
        bow = dictionary.doc2bow(cleaned_description.split())
        topic_vector = lda_model.get_document_topics(bow, minimum_probability=0)
        topic_vector = [prob for _, prob in sorted(topic_vector, key=lambda x: x[0])]
        polarity, subjectivity = extract_sentiment_features(combined_description)
        combined_features = np.hstack([topic_vector, episode_duration_ms, [polarity, subjectivity]]).reshape(1, -1)

        # 标准化和降维
        scaler = load('scaler_model.joblib')
        scaled_new_sample = scaler.transform(combined_features)
        pca_model = load('pca_model.joblib')
        transformed_data = pca_model.transform(scaled_new_sample)
        transformed_data = transformed_data[0, 0:2].reshape(1, -1)
        distances = euclidean_distances(transformed_data, scaled_features).flatten()

        # 最近的样本
        nearest_indices = distances.argsort()[:5]
        nearest_episodes = data.iloc[nearest_indices].copy()
        nearest_episodes['Distance'] = distances[nearest_indices]

        # 绘图
        pc1, pc2 = transformed_data[0, 0], transformed_data[0, 1]
        plt.figure(figsize=(8, 6))
        plt.scatter(scaled_features[:, 0], scaled_features[:, 1], color='green', s=5, marker='x', label='All Points')
        plt.scatter(pc1, pc2, color='red', s=100, label='Input Episode')
        plt.scatter(scaled_features[nearest_indices, 0], scaled_features[nearest_indices, 1], color='blue', s=150, marker='x', label='Nearest Points')
        plt.title('PCA Result (2D Projection)', fontsize=14)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend()
        plt.grid(True)

        return nearest_episodes, plt
    except Exception as e:
        st.error(f"Error processing the episode: {e}")
        return None, None

# 用户提交
if st.button("Get Recommendations"):
    if episode_id:
        nearest_episodes, plot = process_new_sample(episode_id)
        if nearest_episodes is not None:
            st.subheader("Nearest Episodes:")
            st.write(nearest_episodes)
            st.subheader("PCA Visualization:")
            st.pyplot(plot)
    else:
        st.warning("Please enter a valid Episode ID.")

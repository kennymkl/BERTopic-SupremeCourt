import streamlit as st
from bertopic import BERTopic
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
import os
import json

#Load the pre-trained BERTopic model
@st.cache_resource
def load_model():
    return BERTopic.load("topic_model_supreme")

# Load text data
def load_text_data(folder_path):
    text_data = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file_name in filenames:
            if file_name.endswith('.json'):
                file_path = os.path.join(dirpath, file_name)
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
                    text = json_data['text']
                    text_data.append({'text': text})
    return pd.DataFrame(text_data)


def main():
    st.title("Supreme Court Jurisprudence Topic Modeling")
    
    #Load model
    topic_model = load_model()
    
    #Input text from user
    user_input = st.text_area("Enter text to analyze:", "")
    
    if user_input:
        num_of_topics = 3
        similar_topics, similarity = topic_model.find_topics(user_input, top_n=num_of_topics)
        
        st.write(f'The top {num_of_topics} similar topics are {similar_topics}, and the similarities are {np.round(similarity, 2)}')
        
        for i in range(num_of_topics):
            st.write(f"**Top keywords for topic {similar_topics[i]}**:")
            st.write(topic_model.get_topic(similar_topics[i]))

        # Visualization of similar topics
        st.subheader("Visualization of Similar Topics")
        
        umap_model = UMAP(n_neighbors=2, n_components=2, metric="cosine", random_state=42)
        topic_embeddings = topic_model.topic_embeddings_[similar_topics]
        reduced_embeddings = umap_model.fit_transform(topic_embeddings)

        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'topic': [f'Topic {i}' for i in similar_topics]
        })

        # Plotly scatter plot
        st.plotly_chart({
            'data': [{
                'x': df['x'],
                'y': df['y'],
                'mode': 'markers+text',
                'marker': {'size': 10},
                'text': df['topic'],
                'textposition': 'top center'
            }],
            'layout': {'title': 'UMAP Visualization of Similar Topics'}
        })
    
if __name__ == "__main__":
    main()

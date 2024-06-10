from flask import Flask, request, render_template
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import nltk

nltk.download('stopwords')

app = Flask(__name__)

def read_article(data):
    sentences = data.split(".")
    sentences = [sentence.replace("[^a-zA-Z]", "").split(" ") for sentence in sentences]
    sentences.pop()
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    similarity_matrix += np.random.rand(len(sentences), len(sentences)) * 0.01
    return similarity_matrix

def generate_summary(data, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences = read_article(data)
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    top_n = min(top_n, len(ranked_sentence))

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    return ". ".join(summarize_text)

@app.route('/')
def index():
    return render_template('index.html', result="", data="")

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.form['data']
    top_n = int(request.form['maxL'])
    summary = generate_summary(data, top_n)
    return render_template('index.html', result=summary, data=data)

if __name__ == '__main__':
    app.run(debug=True)

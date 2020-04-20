import os 
import sys
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
sys.path.append('skip-thoughts')
import skipthoughts
sys.path.append('skip-thoughts/training')
import tools
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
import click 

SUMMARY_LENGTH = 5

class Summarizer:
    def __init__(self, trainedModel=False): #path_to_model, path_to_dictionary, path_to_word2vec, trained = False): 
        # embed_map = tools.load_googlenews_vectors(path_to_word2vec)
        self.trained = trainedModel
        print('Word Vectors have been loaded...')
        # self.model = tools.load_model(embed_map, path_to_model, path_to_dictionary)
        if self.trained:
            print('LOADING TRAINED MODEL')
            self.model = tools.load_model()
        else:
            print('LOADING PRE TRAINED MODEL')
            self.model = skipthoughts.load_model()
        # self.model = skipthoughts.load_model()
        print('loaded model')
        # input()

    def get_summary(self, path_to_file):
        sentences = self.get_sentences_from_file(path_to_file)
        # vectors = self.encode_sentences(sentences)
        # n_clusters, kmeans = self.cluster(vectors)
        # summmary = self.summarize(n_clusters, kmeans, vectors, sentences)
        if self.trained: 
            summary = self.summarize_trained(sentences)
        else:
            summary = self.summarize(sentences)
        return summary

    def encode_sentences(self, sentences):
        vectors = tools.encode(self.model, sentences)
        print('Sentences have been encoded...')
        return vectors

    def get_sentences_from_file(self, path_to_file):
        with open(path_to_file, 'rb') as f1:
            sentences = f1.readlines()
        return sentences
    
    def cluster(self, vectors):
        n_clusters = int(np.ceil(len(vectors**0.5)))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(vectors)
        return n_clusters, kmeans

    def summarize_trained(self, sentences): #n_clusters, kmeans, vectors, sentences):
        # encoder = skipthoughts.Encoder(self.model)
        # vectors = encoder.encode(sentences)
        vectors = tools.encode(self.model, sentences)
        print('Sentences have been encoded...')
        # self.draw_words(self.model, sentences, pca=True)
        n_clusters = int(np.ceil(len(vectors)**0.5))
        # n_clusters = int(np.ceil(SUMMARY_LENGTH))
        print('num of sentences ' + str(len(vectors)**0.5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(vectors)
        avg = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vectors)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        summary = ' '.join([sentences[closest[idx]] for idx in ordering])
        return summary

    def summarize(self, sentences):
        encoder = skipthoughts.Encoder(self.model)
        vectors = encoder.encode(sentences) 
        print('Sentences have been encoded...')
        n_clusters = int(np.ceil(len(vectors)**0.5))
        # n_clusters = int(np.ceil(SUMMARY_LENGTH))
        print('num of sentences ' + str(len(vectors)**0.5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(vectors)
        avg = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vectors)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        summary = ' '.join([sentences[closest[idx]] for idx in ordering])
        return summary

    def draw_embeddings(self, vectors, sentences):
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(vectors)
         
        plt.figure(figsize=(8, 8))

@click.command()
@click.option('--trained', default=False, help='True if you want to use trained model')
@click.option('--file', default=False, help='option to specify if transcribing single file')
@click.option('--folder', default=False, help='option to specify if program is trancribing files in a folder')
@click.argument('path', required=False)
def run(file, folder, trained, path):
    if file and folder:
        click.echo('please specify either file or folder not both')
    elif file:
        if trained:
            tool = Summarizer.Summarizer(trainedModel=True)
        else:
            tool = Summarizer(trainedModel=False)
        click.echo(tool.get_summary(path))
    elif folder: 
        if trained:
            tool = Summarizer(trainedModel=True)
        else:
            tool = Summarizer(trainedModel=False)
        calls = os.listdir(path)
        for call in calls:
            call = os.path.join(path, call)
            click.echo(tool.get_summary(call))
    else:
        if path:
            print('please specify wheter the path is a folder or file.')
            exit()
        elif trained:
            tool = Summarizer(trainedModel=True)
        else:
            tool = Summarizer(trainedModel=False)
        while True:
            text = raw_input()
            if text == 'exit': break
            tokenized = text.split('.')
            for token in tokenized:
                token = token.strip()
            if len(tokenized) <= 4:
                click.echo('please input a phrase with more than four sentences.')
            else:
                if trained:
                    click.echo(tool.summarize_trained(tokenized))
                else:
                    click.echo(tool.summarize(tokenized))

if __name__ == "__main__":
    run()

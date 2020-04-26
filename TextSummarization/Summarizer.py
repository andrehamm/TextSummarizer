# Andre Hamm
# Command line Text Summarizer using Skip-Thought networks
# CMPT 390

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
import click 

# SUMMARY_LENGTH = 15

class Summarizer:
    def __init__(self, trainedModel=False):
        self.trained = trainedModel
        print('Word Vectors have been loaded...')
        if self.trained:
            print('LOADING TRAINED MODEL')
            self.model = tools.load_model()
        else:
            print('LOADING PRE TRAINED MODEL')
            self.model = skipthoughts.load_model()
        print('loaded model')

    def get_summary(self, path_to_file):
        sentences = self.get_sentences_from_file(path_to_file)
        if self.trained: 
            summary = self.summarize_trained(sentences)
        else:
            summary = self.summarize(sentences)
        return summary

    def get_sentences_from_file(self, path_to_file):
        with open(path_to_file, 'rb') as f1:
            sentences = f1.readlines()
        return sentences

    def summarize_trained(self, sentences): 
        # Getting sentence embeddings
        vectors = tools.encode(self.model, sentences, verbose=False)
        print('Sentences have been encoded...')
        # Retrieving clusters
        n_clusters = int(np.ceil(len(vectors)**0.5))
        # n_clusters = int(np.ceil(SUMMARY_LENGTH))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        # print pca embeddings
        self.print_embeddings(vectors)
        kmeans.fit(vectors)
        avg = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        # Choosing sentences closest to cluster centers
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vectors)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        # Returning summary
        summary = ' '.join([sentences[closest[idx]] for idx in ordering])
        return summary

    def summarize(self, sentences):
        # Getting sentence embeddings
        encoder = skipthoughts.Encoder(self.model)
        vectors = encoder.encode(sentences, verbose=False) 
        print('Sentences have been encoded...')
        # Retrieving clusters
        n_clusters = int(np.ceil(len(vectors)**0.5))
        # n_clusters = int(np.ceil(SUMMARY_LENGTH))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        # print pca embeddings
        self.print_embeddings(vectors)
        kmeans.fit(vectors)
        avg = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        # Choosing sentences closest to cluster centers
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vectors)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        # Returning summary
        summary = ' '.join([sentences[closest[idx]] for idx in ordering])
        return summary

    def print_embeddings(self, vectors):
        X_reduced = PCA(n_components=3).fit_transform(vectors)
        print(X_reduced)


@click.command()
@click.option('--trained/--pre-trained', default=False, help='specify if you want to use trained or pre-trained model')
@click.option('--file', help='set path to a tokenized file you want to transcribe')
@click.option('--folder', help='set path to a folder of tokenized files you want to transcribe')
def run(file, folder, trained):
    #  Checking if user has entered options for both file and folder transcription
    if file and folder:
        click.echo('please specify either file or folder not both')
    # Logic handling file trasncription
    elif file:
        if not os.path.isfile(file):
            click.echo('please enter a valid file path')
            exit()
        if trained:
            tool = Summarizer(trainedModel=True)
        else:
            tool = Summarizer(trainedModel=False)
        click.echo(tool.get_summary(file))
    # Logic handling folder transcription 
    elif folder: 
        if not os.path.isdir(folder):
            click.echo('please enter a valid folder path')
            exit()
        if trained:
            tool = Summarizer(trainedModel=True)
        else:
            tool = Summarizer(trainedModel=False)
        calls = os.listdir(folder)
        for call in calls:
            call = os.path.join(folder, call)
            click.echo(tool.get_summary(call))
    # Logic handling user input 
    else:
        if trained:
            tool = Summarizer(trainedModel=True)
        else:
            tool = Summarizer(trainedModel=False)
        while True:
            click.echo('Please enter text with more than four sentences')
            text = raw_input()
            if text == 'exit': break
            if text[-1] == '.':
                text = text[:-1]
            tokenized = text.split('.')
            for token in tokenized:
                token = token.strip()
            if len(tokenized) <= 4:
                click.echo('Please enter text with more than four sentences')
            else:
                if trained:
                    click.echo(tool.summarize_trained(tokenized))
                else:
                    click.echo(tool.summarize(tokenized))

if __name__ == "__main__":
    run()

import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import mplcursors

from sklearn.cluster import KMeans

import pickle
import os.path

from sklearn.metrics.pairwise import cosine_similarity

DIMENSIONS = 300

ROWS = 8192

vocabPath = "lexvec.enwiki+newscrawl.300d.W.pos.vectors"

vocabPickle = "vocab.pickle"
tsnePickle = "tsne.pickle"

def getVocabFromFile(filename, maxRow=ROWS):
    vocab = {}

    with open(filename) as f:
        row = 1
        while line := f.readline():
            if row == 1:
                row += 1
                continue
        
            if row > maxRow:
                break
            
            tokens = line.rstrip().split()
            vocab[tokens[0]] = np.array(tokens[1:]).astype(np.float32)
            row += 1
            
    return vocab

def saveVocab(filename, maxRow=ROWS):
    vocab = getVocabFromFile(filename, maxRow)

    with open(vocabPickle, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    return vocab

def loadVocab():
    with open(vocabPickle, "rb") as f:
        vocab = pickle.load(f)

    return vocab

def getTSNE(vocab):
    words = list(vocab.keys())
    word_vectors = np.array(list(vocab.values()))

    tsne = TSNE(n_components=2, random_state=0)
    word_vectors_2d = tsne.fit_transform(word_vectors)
    
    return words, word_vectors, word_vectors_2d

def saveTSNE(vocab):
    tsneTuple = getTSNE(vocab)

    with open(tsnePickle, "wb") as f:
        pickle.dump(tsneTuple, f)
    
    return tsneTuple

def loadTSNE():
    with open(tsnePickle, "rb") as f:
        tsneTuple = pickle.load(f)

    return tsneTuple

def saveFiles():
    vocab = saveVocab(vocabPath)
    saveTSNE(vocab)

def getKMeans(word_vectors, c=10):
    if isinstance(c, np.ndarray):
        kmeans = KMeans(n_clusters=c.shape[0], init=c, random_state=0)
    else:
        kmeans = KMeans(n_clusters=c, random_state=0)
    clusters = kmeans.fit(word_vectors)
    
    return clusters

def exampleClustering(interactive=True):
    tsneTuple = loadTSNE()

    words, word_vectors, word_vectors_2d = tsneTuple

    clusters = getKMeans(word_vectors)

    indices = np.triu_indices(clusters.cluster_centers_.shape[0], k=1)

    similarities = cosine_similarity(clusters.cluster_centers_)[indices]

    clusterSimilarities = sorted(list(zip(zip(*indices), similarities)), key=lambda x: x[1])
    #print(clusterSimilarities)

    for (c1, c2), _ in clusterSimilarities[:3]:
        mask = np.where(np.logical_or(clusters.labels_ == c1, clusters.labels_ == c2))
        word_vectors_2d_cs = word_vectors_2d[mask]
        word_vectors_cs = word_vectors[mask]
        labels_cs = clusters.labels_[mask]
        words_cs = np.array(words)[mask]

        if interactive:
            fig, ax = plt.subplots()
            scatter = ax.scatter(word_vectors_2d_cs[:, 0], word_vectors_2d_cs[:, 1], c=labels_cs, alpha=0.5)
            
            mplcursors.cursor(scatter).connect(
                "add", lambda sel: sel.annotation.set_text(words_cs[sel.target.index])
            )
            
            plt.show()
        else:
            plt.scatter(word_vectors_2d_cs[:, 0], word_vectors_2d_cs[:, 1], c=labels_cs, alpha=0.5)
            
            cluster_centers = [(i, np.mean(word_vectors_2d_cs[labels_cs == i], axis=0)) for i in [c1, c2]]

            for i, center in cluster_centers:
                cluster_words = words_cs[labels_cs == i]
                #print(cluster_words)
                if len(cluster_words) > 10:
                    center_word = cluster_words[np.argsort(np.linalg.norm(word_vectors_2d_cs[labels_cs == i] - center, axis=1))[0]]
                    plt.annotate(center_word, xy=center)
                    
            plt.show()

if __name__ == '__main__':
    if not (os.path.isfile(vocabPickle) and os.path.isfile(tsnePickle)):
        saveFiles()

    exampleClustering()
    #exampleClustering(False)
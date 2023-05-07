import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import mplcursors

from sklearn.cluster import KMeans

import pickle
import os.path

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

def plotTSNE(tsneTuple):
    _, _, word_vectors_2d = tsneTuple

    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])

    plt.show()

def getKMeans(word_vectors_2d, c):
    if isinstance(c, np.ndarray):
        kmeans = KMeans(n_clusters=c.shape[0], init=c, random_state=0)
    else:
        kmeans = KMeans(n_clusters=c, random_state=0)
    clusters = kmeans.fit_predict(word_vectors_2d)
    
    return clusters

def plotKMeans(tsneTuple, c=10, interactive=True):
    words, _, word_vectors_2d = tsneTuple
    
    if not isinstance(c, list):
        c = [c]
    
    for cc in c:
        clusters = getKMeans(word_vectors_2d, cc)
        
        if interactive:
            fig, ax = plt.subplots()
            scatter = ax.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], c=clusters, alpha=0.5)
            
            mplcursors.cursor(scatter).connect(
                "add", lambda sel: sel.annotation.set_text(words[sel.target.index])
            )
            
            plt.show()
        else:
            if isinstance(cc, np.ndarray):
                clusterCount = np.shape[0]
            else:
                clusterCount = cc

            cluster_centers = []
            for i in range(clusterCount):
                cluster_points = word_vectors_2d[clusters == i]
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_centers.append(cluster_center)
            cluster_centers = np.array(cluster_centers)

            plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], c=clusters, alpha=0.5)
            
            for i, center in enumerate(cluster_centers):
                cluster_words = np.array(words)[clusters == i]
                #print(cluster_words)
                if len(cluster_words) > 10:
                    center_word = cluster_words[np.argsort(np.linalg.norm(word_vectors_2d[clusters == i] - center, axis=1))[0]]
                    plt.annotate(center_word, xy=center)
                    
            plt.show()

def exampleClustering():
    plotKMeans(loadTSNE())

def exampleClusteringInit():
    tsneTuple = loadTSNE()

    words, _, word_vectors_2d = tsneTuple

    initWords = [
        "male",
        "hot",
        "large",
        "fast",
        "loud",
        "bright",
    ]

    initCentroids = np.zeros((len(initWords), 2))
    for i, word in enumerate(initWords):
        initCentroids[i, :] = word_vectors_2d[words.index(word)]

    plotKMeans(tsneTuple, initCentroids)

if __name__ == '__main__':
    if not (os.path.isfile(vocabPickle) and os.path.isfile(tsnePickle)):
        saveFiles()

    exampleClustering()
    exampleClusteringInit()

##########################################################################

#tsneTuple = loadTSNE()

#plotTSNE(tsneTuple)

#plotKMeans(tsneTuple)
#plotKMeans(tsneTuple, 30)
#plotKMeans(tsneTuple, range(2, 11))
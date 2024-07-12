import numpy as np
from enum import Enum

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import mplcursors

from sklearn.cluster import KMeans

import pickle
import os.path

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

DIMENSIONS = 300

class Distance(Enum):
    EUCLID = 1
    COSINE = 2

distance = Distance.EUCLID
#distance = Distance.COSINE

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

def getKMeans(word_vectors, c):
    if isinstance(c, np.ndarray):
        kmeans = KMeans(n_clusters=c.shape[0], init=c, random_state=0)
    else:
        kmeans = KMeans(n_clusters=c, random_state=0)
    clusters = kmeans.fit(word_vectors)
    
    return clusters

def plotKMeans(tsneTuple, c=10):
    words, word_vectors, word_vectors_2d = tsneTuple

    clusters = getKMeans(word_vectors, c)

    indices = np.triu_indices(clusters.cluster_centers_.shape[0], k=1)

    # Either get cosine similarity or euclidean distance
    if distance == Distance.COSINE:
        similarities = cosine_similarity(clusters.cluster_centers_)[indices]
    elif distance == Distance.EUCLID:
        similarities = euclidean_distances(clusters.cluster_centers_)[indices]

    clusterSimilarities = sorted(list(zip(zip(*indices), similarities)), key=lambda x: x[1])
    #print(clusterSimilarities)

    for (c1, c2), similarity in clusterSimilarities[:3] + clusterSimilarities[-3:]:
        mask = np.where(np.logical_or(clusters.labels_ == c1, clusters.labels_ == c2))
        word_vectors_2d_cs = word_vectors_2d[mask]
        labels_cs = clusters.labels_[mask]
        words_cs = np.array(words)[mask]

        fig, ax = plt.subplots()
        scatter = ax.scatter(word_vectors_2d_cs[:, 0], word_vectors_2d_cs[:, 1], c=labels_cs, alpha=0.5)
        
        mplcursors.cursor(scatter).connect(
            "add", lambda sel: sel.annotation.set_text(words_cs[sel.target.index])
        )
        
        cluster_centers = [(i, np.mean(word_vectors_2d_cs[labels_cs == i], axis=0)) for i in [c1, c2]]

        for i, center in cluster_centers:
            cluster_words = words_cs[labels_cs == i]
            #print(cluster_words)
            if len(cluster_words) > 10:
                center_word = cluster_words[np.argsort(np.linalg.norm(word_vectors_2d_cs[labels_cs == i] - center, axis=1))[0]]
                plt.annotate(center_word, xy=center)

        plt.title(f"{similarity}")     
        plt.show()

def norm(v):
    return v / np.linalg.norm(v)

def getAngle(v1, v2):
    return np.degrees(np.arccos(np.clip(np.dot(norm(v1), norm(v2)), -1, 1)))

def getDistance(v1, v2):
    if distance == Distance.EUCLID:
        return np.linalg.norm(v1 - v2)
        
    if distance == Distance.COSINE:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def getScore(dist, angle, cc):
    return dist * pow(((90 - abs(angle - 90)) / 90), 5 * cc)

def plotKMeansOrthog(tsneTuple, c=10):
    words, word_vectors, word_vectors_2d = tsneTuple

    clusters = getKMeans(word_vectors, c)

    indices = np.triu_indices(clusters.cluster_centers_.shape[0], k=1)

    if distance == Distance.EUCLID:
        distances = euclidean_distances(clusters.cluster_centers_)[indices]
    elif distance == Distance.COSINE:
        distances = cosine_similarity(clusters.cluster_centers_)[indices]

    #clusterDistances = sorted(list(zip(zip(*indices), distances)), key=lambda x: x[1], reverse=True)
    clusterDistances = list(zip(zip(*indices), distances))
    
    clusterPairwise = []
    for i, (clusterPair1, _) in enumerate(clusterDistances[:-1]):
        for clusterPair2, _ in clusterDistances[i+1:]:
            if set(clusterPair1).isdisjoint(set(clusterPair2)):
                c11, c12 = clusterPair1
                c21, c22 = clusterPair2
                v11 = clusters.cluster_centers_[c11]
                v12 = clusters.cluster_centers_[c12]
                v21 = clusters.cluster_centers_[c21]
                v22 = clusters.cluster_centers_[c22]

                angle = getAngle(v11 - v12, v21 - v22)
                dist = getDistance(v11, v12) * getDistance(v21, v22)
                clusterPairwise.append(((clusterPair1, clusterPair2), dist, angle))

    #clusterPairwiseByAngle = sorted(clusterPairwise, key=lambda x: abs(x[2] - 90))

    #clusterPairwiseByDistance = sorted(clusterPairwise, key=lambda x: x[1], reverse=True)
    #print(clusterPairwiseByDistance)
    #clusterPairwiseByAngle = list(filter(lambda x: abs(x[2] - 90) < 0.5, clusterPairwiseByDistance))
    #print(clusterPairwiseByAngle)

    clusterPairwiseByAngle = sorted(clusterPairwise, key=lambda x: getScore(x[1], x[2], clusters.cluster_centers_.shape[0]), reverse=True)

    for ((c11, c12), (c21, c22)), dist, angle in clusterPairwiseByAngle[:3]:
        mask = np.where(np.logical_or.reduce((clusters.labels_ == c11, clusters.labels_ == c12, clusters.labels_ == c21, clusters.labels_ == c22)))
        word_vectors_2d_cs = word_vectors_2d[mask]
        labels_cs = clusters.labels_[mask]
        words_cs = np.array(words)[mask]

        fig, ax = plt.subplots()
        scatter = ax.scatter(word_vectors_2d_cs[:, 0], word_vectors_2d_cs[:, 1], c=labels_cs, alpha=0.5)
        
        mplcursors.cursor(scatter).connect(
            "add", lambda sel: sel.annotation.set_text(words_cs[sel.target.index])
        )

        cluster_centers = [(i, np.mean(word_vectors_2d_cs[labels_cs == i], axis=0)) for i in [c11, c12, c21, c22]]

        c11_center = cluster_centers[0][1]
        c12_center = cluster_centers[1][1]
        plt.plot([c11_center[0], c12_center[0]], [c11_center[1], c12_center[1]], 'k--')

        c21_center = cluster_centers[2][1]
        c22_center = cluster_centers[3][1]
        plt.plot([c21_center[0], c22_center[0]], [c21_center[1], c22_center[1]], 'k--')

        for i, center in cluster_centers:
            cluster_words = words_cs[labels_cs == i]
            #print(cluster_words)
            if len(cluster_words) > 10:
                center_word = cluster_words[np.argsort(np.linalg.norm(word_vectors_2d_cs[labels_cs == i] - center, axis=1))[0]]
                plt.annotate(center_word, xy=center)

        plt.title(f"Angle: {angle}, Product of distances: {dist}\nScore: {getScore(dist, angle, clusters.cluster_centers_.shape[0])}")     
        # plt.savefig(f"orthogonal_{angle}_{dist}.pdf")
        plt.show()

def exampleClustering(tsneTuple):
    plotKMeans(tsneTuple)

def exampleClusteringInit(tsneTuple):
    words, word_vectors, word_vectors_2d = tsneTuple

    initWords = [
        "male",
        "hot",
        "large",
        "fast",
        "loud",
        "bright",
    ]

    initCentroids = np.zeros((len(initWords), DIMENSIONS))
    for i, word in enumerate(initWords):
        initCentroids[i, :] = word_vectors[words.index(word)]

    plotKMeans(tsneTuple, initCentroids)

def exampleClusteringOrthog(tsneTuple):
    plotKMeansOrthog(tsneTuple)

if __name__ == '__main__':
    if not (os.path.isfile(vocabPickle) and os.path.isfile(tsnePickle)):
        saveFiles()

    tsneTuple = loadTSNE()

    #exampleClustering(tsneTuple)
    #exampleClusteringInit(tsneTuple)

    exampleClusteringOrthog(tsneTuple)
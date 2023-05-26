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
                clusterPairwise.append(((clusterPair1, clusterPair2), getScore(dist, angle, clusters.cluster_centers_.shape[0])))

    #print(clusterPairwise)
    (c1, c2), score = max(clusterPairwise, key=lambda x: x[1])
    print(score)
    axes = [c1, c2]

    while len(axes) < clusters.cluster_centers_.shape[0] // 2:
        clusterPairwise[:] = [x for x in clusterPairwise if not ((x[0][0] in axes and x[0][1] in axes) or any([((x[0][0][0] in axis) ^ (x[0][0][1] in axis)) or ((x[0][1][0] in axis) ^ (x[0][1][1] in axis)) for axis in axes]))]

        orthogonalities = []
        for (c1, c2), _ in clusterPairwise:
            if c1 in axes:
                c = c2
            elif c2 in axes:
                c = c1
            else:
                continue

            v1 = clusters.cluster_centers_[c[0]]
            v2 = clusters.cluster_centers_[c[1]]
            dist = getDistance(v1, v2)
            angles = [getAngle(v1 - v2, clusters.cluster_centers_[axis[0]] - clusters.cluster_centers_[axis[1]]) for axis in axes]
            score = np.prod([getScore(dist, angle, clusters.cluster_centers_.shape[0]) for angle in angles])
            orthogonalities.append((c, score))

        c, score = max(orthogonalities, key=lambda x: x[1])
        print(score)

        axes.append(c)

    print(axes)

if __name__ == '__main__':
    if not (os.path.isfile(vocabPickle) and os.path.isfile(tsnePickle)):
        saveFiles()

    tsneTuple = loadTSNE()

    plotKMeansOrthog(tsneTuple)
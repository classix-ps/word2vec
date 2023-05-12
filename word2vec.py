import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import pickle
import os.path

DIMENSIONS = 300

class Distance(Enum):
    EUCLID = 1
    COSINE = 2

#distance = Distance.EUCLID
distance = Distance.COSINE

ROWS = 65536

#vocabPath = "lexvec.enwiki+newscrawl.300d.W.pos.vectors"
vocabPath = "vectors.txt"

vocabPickle = "vocab.pickle"

def getDistance(v1, v2):
    if distance == Distance.EUCLID:
        return np.linalg.norm(v1 - v2)
        
    if distance == Distance.COSINE:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def getAvgDistance(vocab, v):
    return sum([getDistance(v, vec) for vec in vocab.values()]) / len(vocab)
    
def getWordDistances(vocab, v):
    return [(word, getDistance(v, vec)) for word, vec in vocab.items()]
    
def getClosestWord(vocab, v):
    if distance == Distance.EUCLID:
        return min(getWordDistances(vocab, v), key=lambda x: x[1])
        
    if distance == Distance.COSINE:
        return max(getWordDistances(vocab, v), key=lambda x: x[1])
    
def getClosestWords(vocab, v, n=1):
    if distance == Distance.EUCLID:
        return sorted(getWordDistances(vocab, v), key=lambda x: x[1])[:n]
    
    if distance == Distance.COSINE:
        return sorted(getWordDistances(vocab, v), key=lambda x: x[1], reverse=True)[:n]

def analyzeEquation(vocab, word1, word2, word3, word4, n=5):
    v1 = vocab[word1] - vocab[word2] + vocab[word3]
    v2 = vocab[word4]
    
    print(f"{word1} - {word2} + {word3} = {word4}")
    print("Distance:", getDistance(v1, v2))
    print("Average distance:", getAvgDistance(vocab, v1))
    print(f"{n} closest words:", "\n\t" + "\n\t".join(map(str, getClosestWords(vocab, v1, n))))
    print()
    
def plotDifferences(vocab, wordPairs):
    X = np.arange(DIMENSIONS)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    for i, (word1, word2) in enumerate(wordPairs):
        ax.bar(X + 0.25 * i, vocab[word1] - vocab[word2], width = 0.25)
        
    plt.show()
    
def calculateSingularity(v1, v2, dimension):
    difference = np.abs(v1 - v2)
    return np.sum(difference) - difference[dimension]
    
def getClosestDimensionalPairs(vocab):
    dimensionalPairs = []

    for dimension in range(DIMENSIONS):
        counterparts = []
        for i, (word, vec) in enumerate(vocab.items()):
            if not i % 100:
                print("Iteration:", i)
                
            if abs(vec[dimension]) < 0.25:
                continue
                
            counterPartVec = vec.copy()
            counterPartVec[dimension] = -counterPartVec[dimension]
            counterPartVocab = vocab.copy()
            counterPartVocab.pop(word)
            closestWord = getClosestWord(counterPartVocab, counterPartVec)[0]
            counterparts.append((word, closestWord, calculateSingularity(vec, vocab[closestWord], dimension)))
        
        
        dimensionalCounterpart = min(counterparts, key=lambda x: x[2])
        dimensionalPairs.append((dimensionalCounterpart[0], dimensionalCounterpart[1]))
        break
        
    return dimensionalPairs
    
def getDimensionalMapping(vocab, dimensionalMaps, dimensionalPairs):
    dimensionalVectors = np.zeros((DIMENSIONS, 2 * len(dimensionalMaps)))
    B = np.zeros((len(dimensionalMaps), 2 * len(dimensionalMaps)))
    for i, dimension in enumerate(dimensionalPairs):
        for word1, word2 in dimensionalPairs[dimension]:
            dimensionalVectors[:, 2*i] += vocab[word1] / len(dimensionalPairs[dimension])
            dimensionalVectors[:, 2*i+1] += vocab[word2] / len(dimensionalPairs[dimension])
            
        onehot = np.zeros(len(dimensionalMaps))
        onehot[i] = 1
        B[:, 2*i] = onehot
        B[:, 2*i+1] = -onehot
    
    #dimensionalMapping = np.linalg.solve(np.identity(len(dimensionalMaps)), dimensionalDifferences.T)
    #print(dimensionalVectors.shape, B.shape)
    dimensionalMapping = np.linalg.lstsq(dimensionalVectors.T, B.T, rcond=None)[0].T
    #print(np.matmul(dimensionalMapping, dimensionalVectors))
    return dimensionalMapping

def getDimensionalMapping2(vocab, dimensionalMaps, dimensionalPairs):
    dimensionalMapping = np.zeros((DIMENSIONS, len(dimensionalMaps)))
    
    for i, dimension in enumerate(dimensionalPairs):
        basisVector = np.zeros(DIMENSIONS)
        
        for word in dimension:
            basisVector += vocab[word]
            
        basisVector /= len(dimension)
        
        dimensionalMapping[:, i] = np.linalg.norm(basisVector)
        
    return dimensionalMapping
    
def getVocabFromFile(filename, maxRow=ROWS):
    vocab = {}

    with open(filename, encoding="utf8") as f:
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

def exampleEquations(vocab):
    print("king - queen:", getDistance(vocab["king"], vocab["queen"]), '\n')

    analyzeEquation(vocab, "king", "man", "woman", "queen") # Even worse on shakespeare training data, almost no connection to queen
    analyzeEquation(vocab, "king", "male", "female", "queen")
    analyzeEquation(vocab, "father", "man", "woman", "mother")
    analyzeEquation(vocab, "father", "son", "daughter", "mother")

    analyzeEquation(vocab, "queen", "woman", "man", "king")
    analyzeEquation(vocab, "queen", "king", "man", "woman")

    analyzeEquation(vocab, "running", "run", "make", "making")

    analyzeEquation(vocab, "husband", "man", "woman", "wife")
    analyzeEquation(vocab, "husband", "wife", "man", "woman")

    analyzeEquation(vocab, "spain", "madrid", "berlin", "germany")

def exampleDifferences(vocab):
    plotDifferences(vocab, [("king", "queen"), ("man", "woman"), ("husband", "wife")])
    
    plotDifferences(vocab, [("running", "run"), ("making", "make")])

def exampleDimensionalPairs(vocab):
    dimensionalPairs = getClosestDimensionalPairs(vocab)
    print(dimensionalPairs)
    for pair in dimensionalPairs:
        plotDifferences(vocab, [(pair[0], pair[1])])

def exampleDimensionalMapping(vocab):
    dimensionalMaps = {
        0: "gender",
        1: "temperature",
        2: "size",
        3: "velocity",
        4: "loudness",
        5: "luminance",
    }
    
    dimensionalPairs = {
        "gender": [("male", "female")],
        "temperature": [("hot", "cold")],
        "size": [("large", "small")],
        "velocity": [("fast", "slow")],
        "loudness": [("loud", "quiet")],
        "luminance": [("bright", "dark")],
    }
    
    dimensionalMapping = getDimensionalMapping(vocab, dimensionalMaps, dimensionalPairs)
    
    print("Dimensional mapping v1")
    dimensionalMappingPrinter = lambda word : print(word, dimensionalMapping.dot(vocab[word]))
    dimensionalMappingPrinter("male")
    dimensionalMappingPrinter("woman")
    
    dimensionalMapping2 = getDimensionalMapping2(vocab, dimensionalMaps, dimensionalPairs)
    
    print()
    print("Dimensional mapping v2")
    dimensionalMapping2Printer = lambda word : print(word, np.matmul(vocab[word], dimensionalMapping2))
    dimensionalMapping2Printer("male")
    dimensionalMapping2Printer("woman")
    dimensionalMapping2Printer("big")
    dimensionalMapping2Printer("small")

if __name__ == '__main__':
    if not os.path.isfile(vocabPickle):
        saveVocab(vocabPath)

    vocab = loadVocab()

    exampleEquations(vocab)
    #exampleDifferences(vocab)
    #exampleDimensionalPairs(vocab)

    #exampleDimensionalMapping(vocab)
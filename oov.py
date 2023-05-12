import model as lexvec
import numpy as np
from enum import Enum
import os
import pickle

ROWS = 65536

vocabPath = "vectors.txt"

vocabPickle = "vocab.pickle"

class Distance(Enum):
    EUCLID = 1
    COSINE = 2

#distance = Distance.EUCLID
distance = Distance.COSINE

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

def getDistance(v1, v2):
    if distance == Distance.EUCLID:
        return np.linalg.norm(v1 - v2)
        
    if distance == Distance.COSINE:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

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

def analzyeNeighbors(vocab, model, word, n=5):
    print(f"{word}")
    print(f"{n} closest words:", "\n\t" + "\n\t".join(map(str, getClosestWords(vocab, model.word_rep(word), n))))
    print()

def anchor(vocab, anchors):
    return {k: vocab[k] for k in anchors if k in vocab}

def transpose(anchorVocab, model, prompt):
    tokens = prompt.strip(".").split()

    result = []
    for token in tokens:
        anchor, distance = getClosestWord(anchorVocab, model.word_rep(token))
        if distance > 0.4:
            result.append(anchor) # TODO: add spellcheck, since OOV susceptible to typos

    return " ".join(result)

def exampleNeighbors(vocab, model):
    analzyeNeighbors(vocab, model, "marvelicious")

    analzyeNeighbors(vocab, model, "clientt")
    analzyeNeighbors(vocab, model, "customr")
    analzyeNeighbors(vocab, model, "buyyer")

def exampleTranspose(vocab, model):
    anchorVocab = anchor(vocab, ["client", "give", "Germany"])

    print(transpose(anchorVocab, model, "Give me all clients from Hamburg."))

if __name__ == '__main__':
    model = lexvec.Model("model.bin")

    if not os.path.isfile(vocabPickle):
        saveVocab(vocabPath)

    vocab = loadVocab()

    np.testing.assert_array_almost_equal(vocab["king"], model.word_rep("king"))
    
    #exampleNeighbors(vocab, model)
    exampleTranspose(vocab, model)
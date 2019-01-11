import numpy as np


def load_word_vectors(filepath, dimensions=50):
    wordVectors = {}
    f = open(filepath, 'r')
    for line in f:
        linfo = line.strip().split()
        word = linfo[0]
        # embedding = np.array(map(float, linfo[1:]))
        embedding = np.asarray(linfo[1:], dtype=float)
        wordVectors[word] = embedding
    return wordVectors

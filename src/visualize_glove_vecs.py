import sys 
import tensorflow as tf
import numpy as np
from load_data import *
from glove import *
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from matplotlib import colors as mcolors
import matplotlib.cm as cm



USAGE_STR = """

python feed_forward_nn.py /afs/ir.stanford.edu/users/a/k/akma327/cs224n/project/mbti-net/data/mbti_shuffled_data.txt 0.1

"""


mbti_index = {"ISTJ" : 0, "ISFJ" :1, "INFJ" :2, "INTJ" :3, "ISTP" :4, "ISFP" : 5, "INFP":6, "INTP":7, "ESTP": 8, "ESFP":9, "ENFP":10, "ENTP":11, "ESTJ":12, "ESFJ":13, "ENFJ":14, "ENTJ":15}
d = {}
for key in mbti_index:
    d[mbti_index[key]] = key

mbti_labels = []
for i in range(16):
    mbti_labels.append(d[i])


GLOVE_DIMENSION = 100


RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def prep_data(DATA_FILE):
    """
        n = number of data points 
        d = size of glove word embedding 
        c = 16 mbti classes 
        train_X, test_X : Generate n x d matrix with each row being an average over 
        all word embeddings within a sentence 

        train_y, test_y : Generate n x c matric with each row being a one hot 
        vector representation of the labeled personality type. 

    """

    print("prepping data ...")

    def process_rows(data, glove_vectors):
        stop_words = set(stopwords.words('english'))
        x_matrix = []
        y_matrix = []

        num_data_points = len(data)

        ### For every row 
        for i, (word_arr, y_arr) in enumerate(data):
            num_words = 0
            avg_glove = np.array([0.0]*GLOVE_DIMENSION)

            for word in word_arr:
                if(word in glove_vectors) and (word not in stop_words):
                    embedding = glove_vectors[word]
                    avg_glove += embedding
                    num_words += 1

            avg_glove /= num_words
            x_matrix.append(avg_glove)
            y_matrix.append(y_arr)

        x_matrix = np.array(x_matrix)
        y_matrix = np.array(y_matrix)

        return x_matrix, y_matrix



    data, _  = load_data(DATA_FILE, 1.0)
    glove = load_word_vectors("../data/glove.6B.100d.txt", GLOVE_DIMENSION)
    glove_vectors, labels = process_rows(data, glove)


    return glove_vectors, labels

def visualizeGloveVectors(glove_vectors, labels, num_data_points_to_plot=300):
    mbti = []
    for i in range(num_data_points_to_plot):
        for j in range(len(labels[i])):
            if labels[i][j] == 1:
                mbti.append(d[j])
                break

    temp = (glove_vectors - np.mean(glove_vectors, axis=0))
    covariance = 1.0 / len(mbti) * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance, full_matrices=False)
    coord = temp.dot(U[:,0:2])
    #print(mcolors.BASE_COLORS)
    colors = {"ISTJ" : "indianred",
              "ISFJ": "r",
              "ESFJ":"maroon",
              "ESTJ":"lightcoral",
              "INFJ" : "mediumaquamarine",
              "INFP": "darkolivegreen",
              "ENFJ": "limegreen",
              "ENFP": "y",
              "INTJ" :"darkviolet",
              "INTP": "plum",
              "ENTJ": "fuchsia",
              "ENTP": "mediumorchid",
              "ISTP" :"sienna",
              "ISFP" : "goldenrod",
              "ESTP": "sandybrown",
              "ESFP":"darkorange"}

    plots = {}
    for i in range(len(mbti)):
        if plots.get(mbti[i]) is None:
            plots[mbti[i]] = ([],[])
        plots[mbti[i]][0].append(coord[i,0])
        plots[mbti[i]][1].append(coord[i,1])
    plts = []
    for key in plots:
        plts.append(plt.scatter(plots[key][0], plots[key][1], color=colors[key]))

    #plt.scatter(coord[:num_data_points_to_plot,0], coord[:num_data_points_to_plot,1], color=[colors[m] for m in mbti])
    plt.legend(plts, plots.keys())
    # sort both labels and handles by labels
    # handles, labls = plt.get_legend_handles_labels()
    # labls, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # plt.legend(handles, labels)
    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
    plt.title("Sentence Average Glove Visualization - 300 Data Points")

    plt.savefig('word_vectors_visualization-300.png')

if __name__ == '__main__':
    #colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    #by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    #for name, color in colors.items())
    #sorted_names = [name for hsv, name in by_hsv]
    #print(sorted_names)
    DATA_FILE = r'/Users/sayakghosh/Documents/sem1/nlp/project/akma/mbti-net/data/mbti_balanced_shuffled_data.txt'
    glove_vectors, labels = prep_data(DATA_FILE)
    visualizeGloveVectors(glove_vectors, labels)
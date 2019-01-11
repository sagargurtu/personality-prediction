import os
import sys
import numpy as np

USAGE_STR = """

# Usage 
# python load_data.py <DATA_FILE> <PERCENTAGE_TRAIN>

# Arguments
# <DATA_FILE> Absolute path to shuffled data file with each row being a data point 
# <PERCENTAGE_TRAIN> Percentage of rows to use for training. One minus this percentage 
# for testing

"""

mbti_index = {"ISTJ": 0, "ISFJ": 1, "INFJ": 2, "INTJ": 3, "ISTP": 4, "ISFP": 5, "INFP": 6, "INTP": 7, "ESTP": 8,
              "ESFP": 9, "ENFP": 10, "ENTP": 11, "ESTJ": 12, "ESFJ": 13, "ENFJ": 14, "ENTJ": 15}


def load_data(DATA_FILE, PERCENTAGE_TRAIN=0.7):
    f = open(DATA_FILE, 'r')
    all_data = []
    for line in f:
        linfo = line.strip().split("\t")
        mbti, sentence_str = linfo[0], linfo[3]
        mbti_one_hot = np.array([0.0] * 16)
        mbti_one_hot[mbti_index[mbti]] = 1

        sentence_str = sentence_str.replace(".", " ").lower()
        words = [w for w in sentence_str.split(" ") if w != ""]

        all_data.append((words, mbti_one_hot))

    num_data_points = len(all_data)
    num_train_points = int(PERCENTAGE_TRAIN * num_data_points)

    train_data = all_data[:num_train_points]
    test_data = all_data[num_train_points:]

    return train_data, test_data


if __name__ == "__main__":
    INPUT_PATH = '../data/mbti_balanced_shuffled_data.txt'
    train_data, test_data = load_data(INPUT_PATH, 0.7)

    train_Y = train_data[1][1]
    test_Y = test_data[1][1]
    print(train_data[0:50], len(train_data), len(test_data))

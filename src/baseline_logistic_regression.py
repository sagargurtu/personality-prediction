from feed_forward_nn import *
from sklearn.linear_model import LogisticRegression

GLOVE_DIMENSION = 100

reverse_mbti_index = {0: "ISTJ", 1: "ISFJ", 2: "INFJ", 3: "INTJ", 4: "ISTP", 5: "ISFP", 6: "INFP", 7: "INTP", 8: "ESTP",
                      9: "ESFP", 10: "ENFP", 11: "ENTP", 12: "ESTJ", 13: "ESFJ", 14: "ENFJ", 15: "ENTJ"}

mbti_index = {"ISTJ": 0, "ISFJ": 1, "INFJ": 2, "INTJ": 3, "ISTP": 4, "ISFP": 5, "INFP": 6, "INTP": 7, "ESTP": 8,
              "ESFP": 9, "ENFP": 10, "ENTP": 11, "ESTJ": 12, "ESFJ": 13, "ENFJ": 14, "ENTJ": 15}


def seperate_mbti_classes(y):
    class_1 = []
    class_2 = []
    class_3 = []
    class_4 = []

    for instance in y:
        mbti_index = np.argmax(instance)
        mbti_type = reverse_mbti_index[mbti_index]
        if mbti_type[0] == 'I':
            class_1.append(1)
        else:
            class_1.append(0)

        if mbti_type[1] == 'S':
            class_2.append(1)
        else:
            class_2.append(0)

        if mbti_type[2] == 'T':
            class_3.append(1)
        else:
            class_3.append(0)

        if mbti_type[3] == 'J':
            class_4.append(1)
        else:
            class_4.append(0)

    return class_1, class_2, class_3, class_4


def compute_mbti(predictions_class1, predictions_class2, predictions_class3, predictions_class4, shape):
    mbti_result = []
    for i in range(0, shape):
        result = []
        if predictions_class1[i] == 1:
            result.append('I')
        else:
            result.append('E')

        if predictions_class2[i] == 1:
            result.append('S')
        else:
            result.append('N')

        if predictions_class3[i] == 1:
            result.append('T')
        else:
            result.append('F')

        if predictions_class4[i] == 1:
            result.append('J')
        else:
            result.append('P')

        mbti_type = "".join(result)
        index = mbti_index[mbti_type]
        mbti_result.append(index)
        print(mbti_type)

    values = mbti_result
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]


if __name__ == "__main__":
    CHUNK_SIZE = 5

    DATASET_FILE_PATH = '../data/mbti_balanced_shuffled_data.txt'

    dictionary = set([word.strip() for word in open('../data/words.txt', 'r')])
    glove_vectors = load_word_vectors("../data/glove.6B.100d.txt", GLOVE_DIMENSION)

    train_X, test_X, train_y, test_y = prep_data(DATASET_FILE_PATH, 0.7)

    class_1_train, class_2_train, class_3_train, class_4_train = seperate_mbti_classes(train_y)

    logreg1 = LogisticRegression(C=1e5, solver='lbfgs', multi_class='ovr')
    logreg1.fit(train_X, class_1_train)

    logreg2 = LogisticRegression(C=1e5, solver='lbfgs', multi_class='ovr')
    logreg2.fit(train_X, class_2_train)

    logreg3 = LogisticRegression(C=1e5, solver='lbfgs', multi_class='ovr')
    logreg3.fit(train_X, class_3_train)

    logreg4 = LogisticRegression(C=1e5, solver='lbfgs', multi_class='ovr')
    logreg4.fit(train_X, class_4_train)

    predictions_class1 = logreg1.predict(test_X)
    predictions_class2 = logreg2.predict(test_X)
    predictions_class3 = logreg3.predict(test_X)
    predictions_class4 = logreg4.predict(test_X)

    mbti_predictions = compute_mbti(predictions_class1, predictions_class2, predictions_class3, predictions_class4,
                                    test_X.shape[0])

    test_accuracy = np.mean(np.argmax(test_y, axis=1) == np.argmax(mbti_predictions, axis=1))

    print("Test accuracy is " + str(test_accuracy*100))

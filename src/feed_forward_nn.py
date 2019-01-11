import tensorflow as tf
from load_data import *
from glove import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

GLOVE_DIMENSION = 100
RANDOM_SEED = 42


def show_values(pc, fmt="%.2f", **kw):
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in list(zip(pc.get_paths(), pc.get_facecolors(), pc.get_array())):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def find_sentence_embedding(words, glove_vectors):
    sentence_embedding = np.array([0.0] * GLOVE_DIMENSION)
    num_words = 0
    embedding = np.zeros(GLOVE_DIMENSION)
    for word in words:
        if (word in glove_vectors):
            embedding = glove_vectors[word]
        sentence_embedding = np.add(sentence_embedding, embedding)
        # num_words += 1
        num_words = num_words + 1

    return np.divide(sentence_embedding, num_words)


def process_rows(data, glove_vectors):
    x_matrix = []
    y_matrix = []

    for (words, y_mbti) in data:
        num_words = 0
        sentence_embedding = np.array([0.0] * GLOVE_DIMENSION)

        for word in words:
            if (word in glove_vectors):
                embedding = glove_vectors[word]
                sentence_embedding = np.add(sentence_embedding, embedding)
                num_words = num_words + 1

        sentence_embedding = find_sentence_embedding(words, glove_vectors)

        x_matrix.append(sentence_embedding)
        y_matrix.append(y_mbti)

    x_matrix = np.array(x_matrix)
    y_matrix = np.array(y_matrix)

    return x_matrix, y_matrix


def prep_data(file_path, percentage_train):
    train_data, test_data = load_data(file_path, percentage_train)
    glove_vectors = load_word_vectors("../data/glove.6B.100d.txt", GLOVE_DIMENSION)
    train_X, train_y = process_rows(train_data, glove_vectors)
    test_X, test_y = process_rows(test_data, glove_vectors)

    print(train_X, len(train_X), len(train_X[0]))
    print(train_y, len(train_y), len(train_y[0]))

    return train_X, test_X, train_y, test_y


def run_baseline(file_path, percentage_train):
    mbti_index = {"ISTJ": 0, "ISFJ": 1, "INFJ": 2, "INTJ": 3, "ISTP": 4, "ISFP": 5, "INFP": 6, "INTP": 7, "ESTP": 8,
                  "ESFP": 9, "ENFP": 10, "ENTP": 11, "ESTJ": 12, "ESFJ": 13, "ENFJ": 14, "ENTJ": 15}
    d = {}
    for key in mbti_index:
        d[mbti_index[key]] = key

    mbti_labels = []
    for i in range(16):
        mbti_labels.append(d[i])

    tf.set_random_seed(RANDOM_SEED)

    train_X, test_X, train_y, test_y = prep_data(file_path, percentage_train)

    print("Running Feed Forward Neural Network ...")

    # Layer's sizes
    x_size = train_X.shape[1]
    h_size = 100  # Number of hidden nodes for 1st hidden layer
    y_size = train_y.shape[1]  # Number of outcomes

    h_size_2 = 50  # Number of hidden nodes for 2nd hidden layer

    # # Symbols
    x = tf.placeholder(tf.float32, shape=[None, x_size], name="x")
    y = tf.placeholder(tf.float32, shape=[None, y_size], name="y")

    normal_initializer = tf.initializers.truncated_normal
    #
    W1 = tf.get_variable("W1", shape=(h_size, h_size_2), initializer=normal_initializer)
    b1 = tf.get_variable("b1", shape=(h_size_2), initializer=tf.constant_initializer(0))

    W2 = tf.get_variable("W2", shape=(h_size_2, y_size), initializer=normal_initializer)
    b2 = tf.get_variable("b2", shape=(y_size), initializer=tf.constant_initializer(0))

    # Forward propagation
    ymid = tf.matmul(x, W1) + b1
    yhat = tf.add(tf.matmul(ymid, W2), b2)
    predict = tf.argmax(yhat, axis=1, name='predict')

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    lr = 0.005
    updates = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)

    training_accuracies = []
    test_accuracies = []

    for epoch in range(1000):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={x: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={x: train_X, y: train_y}))
        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                sess.run(predict, feed_dict={x: test_X, y: test_y}))
        preds = sess.run(predict, feed_dict={x: test_X, y: test_y})
        labels = []
        for i in range(len(test_y)):
            for j in range(len(test_y[i])):
                if test_y[i][j] == 1:
                    labels.append(j)
                    break
        print(np.array(labels), np.array(preds))
        cm = np.array([row / float(np.sum(row)) for row in confusion_matrix(np.array(labels), np.array(preds))])

        ### Display confusion matrix
        print("Confusion Matrix for Epoch: " + str(epoch))
        plt.rcParams["figure.figsize"] = (10, 10)

        plt.xticks(np.arange(len(mbti_labels)), mbti_labels)
        plt.yticks(np.arange(len(mbti_labels)), mbti_labels)
        plt.title("MBTI Prediction Confusion Matrix -- Epoch " + str(epoch + 1))
        # plt.show()

        plt.savefig('../data/confusion-matrices/epoch' + str(epoch + 1) + ".png")
        plt.close()

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

        training_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    epochs = [i for i in range(300)]
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    print(training_accuracies)
    print(test_accuracies)
    ax.set_title('Train vs Test Accuracy')
    plt.plot(epochs, training_accuracies, 'r', label='train')  # plotting t, a separately
    plt.plot(epochs, test_accuracies, 'b', label='test')  # plotting t, b separately
    # plt.plot(epochs, loss, 'g', label='loss')  # plotting t, c separately
    ax.set_xlabel('Epochs')
    ax.legend(loc='best')
    # plt.show()

    fig.savefig('../data/baseline/' + "final.png")

    # Save model
    saver = tf.train.Saver()
    saver.save(sess, '../models/baseline.ckpt')

    sess.close()


if __name__ == '__main__':
    INPUT_FILE = '../data/mbti_balanced_shuffled_data.txt'
    run_baseline(INPUT_FILE, 0.7)

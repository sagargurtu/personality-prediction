import sys
import tensorflow as tf
import numpy as np
from load_data import *
from glove import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn

l_size = 50
h_size = 50
h2_size = 50
batch_size = 64
y_size = 16
GLOVE_DIMENSION = 50
MAX_SEQ_LEN = 120
lam = 1e-8
training_accuracies = []
test_accuracies = []
loss = []


def prep_data(file, train_percent):
    def pad(sentence):
        if (len(sentence) > MAX_SEQ_LEN):
            return sentence[:MAX_SEQ_LEN]
        else:
            num_pads = MAX_SEQ_LEN - len(sentence)
            return sentence + [""] * num_pads

    def process_rows(data, glove_vectors):
        """
            Generate input x matrix M[num_data_points][glove_dimension][max_seq_length]
        """

        inputs, outputs = [], []
        for sentence, mbti_hot in data:
            x = []
            ### Pad sentence
            padded_sentence = pad(sentence)

            ### Convert each word in sentence to glove vector
            for w in padded_sentence:
                embedding = np.array([0.0] * GLOVE_DIMENSION)
                if (w in glove_vectors):
                    embedding = glove_vectors[w]
                x.append(embedding)

            inputs.append(x)
            outputs.append(mbti_hot)

        return inputs, outputs

    train_data, test_data = load_data(file, train_percent)
    print(len(train_data))
    valid_len = int(0.1 * len(train_data))
    valid_data = train_data[valid_len:]
    train_data = train_data[:valid_len]

    glove_vectors = load_word_vectors("../data/glove.6B.50d.txt", GLOVE_DIMENSION)
    train_x, train_y = process_rows(train_data, glove_vectors)
    test_x, test_y = process_rows(test_data, glove_vectors)
    valid_x, valid_y = process_rows(valid_data, glove_vectors)

    return np.array(train_x), np.array(test_x), np.array(valid_x), np.array(train_y), np.array(test_y), np.array(
        valid_y)


mbti_index = {"ISTJ": 0, "ISFJ": 1, "INFJ": 2, "INTJ": 3, "ISTP": 4, "ISFP": 5, "INFP": 6, "INTP": 7, "ESTP": 8,
              "ESFP": 9, "ENFP": 10, "ENTP": 11, "ESTJ": 12, "ESFJ": 13, "ENFJ": 14, "ENTJ": 15}
d = {}
for key in mbti_index:
    d[mbti_index[key]] = key

mbti_labels = []
for i in range(16):
    mbti_labels.append(d[i])


class mbti_model(object):

    def __init__(self, graph):
        print("IN INIT")

        self.build_graph(graph)

    def build_graph(self, graph):
        print("BUILDING THE GRAPH!")
        # Layer's sizes

        with graph.as_default():
            self.inputs = tf.placeholder(tf.float32, shape=[None, MAX_SEQ_LEN, GLOVE_DIMENSION])

            self.labels = tf.placeholder(tf.float32, shape=[None, y_size])

            xavier_initializer = tf.contrib.layers.xavier_initializer()
            W = tf.get_variable("W", shape=(l_size, y_size), initializer=xavier_initializer)
            b = tf.get_variable("b", shape=(y_size), initializer=tf.constant_initializer(0))

            # for bidirectional LSTM, we need 2*l_size weights
            # W = tf.get_variable("W", shape=(2 * l_size, y_size), initializer=xavier_initializer)

            # Uncomment this code to implement bi-directional RNN to get the output for feeding into MLP

            # f_cell = tf.nn.rnn_cell.LSTMCell(h_size)
            # f_cell = tf.nn.rnn_cell.DropoutWrapper(f_cell, output_keep_prob=0.5)
            # b_cell = tf.nn.rnn_cell.LSTMCell(h_size)
            # b_cell = tf.nn.rnn_cell.DropoutWrapper(b_cell, output_keep_prob=0.5)
            #
            # x_cell = tf.unstack(self.inputs, MAX_SEQ_LEN, 1)
            #
            # outputs, _, _ = rnn.static_bidirectional_rnn(f_cell, b_cell, x_cell,
            #                                              dtype=tf.float32)

            # LSTM cell has a dropout wrapper of 0.5 to prevent overfitting due to small data and complex network

            cell = tf.nn.rnn_cell.LSTMCell(h_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)

            outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32)
            outputs = tf.transpose(outputs, [1, 0, 2])

            # Instead of just taking the last output, we can also gather)
            # last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

            # Uncomment this code for bi-word-RNN to feed-forward personality neural network
            # self.pred_y = self.forward_pass(last_output, W_mlp,b_mlp,W2_mlp)

            self.pred_y = tf.matmul(tf.gather(outputs, int(outputs.get_shape()[0]) - 1), W) + b

            self.pred_labels = tf.argmax(self.pred_y, axis=1)

            # Backward propagation
            # reg_loss = 0
            # for w in [self.inputs, W, b]:
            #     reg_loss += 0.5*lam*tf.nn.l2_loss(w)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.pred_y))

            lr = 0.1
            optimizer = tf.train.GradientDescentOptimizer(lr)
            self.app = optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()

    # def forward_pass(self, input, weights_input, bias_input, weights_output):
    #
    #     res = tf.add(tf.matmul(input, weights_input), bias_input)
    #     h = tf.nn.relu(res)
    #
    #     out = tf.matmul(h, weights_output)
    #     return out

    def train(self, sess, train_x, train_y, test_x, test_y, valid_x, valid_y):

        self.init.run()
        print("Right before first epoch!")

        for epoch in range(300):
            valid = True
            # Train with each example
            average_loss = 0
            for step in range(0, len(train_x), batch_size):
                batch_inputs, batch_labels = train_x[step: step + batch_size], train_y[step: step + batch_size]

                feed_dict = {self.inputs: batch_inputs, self.labels: batch_labels}
                _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
                average_loss += loss_val
                if step >= len(train_x) / 2 and valid == True:
                    valid = False
                    val_pred = sess.run(self.pred_labels, feed_dict={self.inputs: valid_x})
                    val_accuracy = np.mean(np.argmax(valid_y, axis=1) == val_pred)
                    print("Validation accuracy: ", 100 * val_accuracy, "%")
            print("Finished Training! Epoch:", epoch + 1, " Avg. loss: ", average_loss)
            loss.append(average_loss)
            self.evaluate(epoch, sess, train_x, train_y, test_x, test_y)

    def evaluate(self, epoch, sess, train_x, train_y, test_x, test_y):

        train_pred = sess.run(self.pred_labels, feed_dict={self.inputs: train_x, self.labels: train_y})
        train_accuracy = np.mean(np.argmax(train_y, axis=1) == train_pred)
        training_accuracies.append(train_accuracy)

        test_pred = sess.run(self.pred_labels, feed_dict={self.inputs: test_x, self.labels: test_y})
        test_accuracy = np.mean(np.argmax(test_y, axis=1) == test_pred)
        test_accuracies.append(test_accuracy)

        labels = []
        for i in range(len(test_y)):
            for j in range(len(test_y[i])):
                if test_y[i][j] == 1:
                    labels.append(j)
                    break
        print(np.array(labels), np.array(test_pred))
        cm = np.array([row / float(np.sum(row)) for row in confusion_matrix(np.array(labels), np.array(test_pred))])

        ### Display confusion matrix
        print("Confusion Matrix for Epoch: " + str(epoch))
        plt.rcParams["figure.figsize"] = (10, 10)
        heatmap = plt.pcolor(cm, cmap=plt.cm.Blues)
        if epoch == 200:
            plt.colorbar(heatmap)
        # show_values(heatmap)

        plt.xticks(np.arange(len(mbti_labels)), mbti_labels)
        plt.yticks(np.arange(len(mbti_labels)), mbti_labels)
        plt.title("MBTI Prediction Confusion Matrix -- Epoch " + str(epoch + 1))
        # plt.show()

        plt.savefig('../data/rnn_lstm/' + str(epoch + 1) + ".png")
        # plt.close()

        print("Epoch:", epoch + 1, " Train Accuracy:", 100 * train_accuracy, "% Test Accuracy:", 100 * test_accuracy,
              "%")


if __name__ == '__main__':
    print("Started Process!")

    # Get train, test, and valid data
    train_x, test_x, valid_x, train_y, test_y, valid_y = prep_data(r'../data/mbti_balanced_shuffled_data.txt', 0.7)

    # Train the Model
    graph = tf.Graph()
    model = mbti_model(graph)
    with tf.Session(graph=graph) as sess:
        model.train(sess, train_x, train_y, test_x, test_y, valid_x, valid_y)

    import matplotlib.pyplot as plt

    # Finally, we print the figure for our Train vs Test accuracy by using the accuracies accumulated
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

    fig.savefig('../data/rnn_lstm/' + "final.png")

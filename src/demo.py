import string

from load_data import *
from glove import *

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

K_MIN_ARG = 1
DEFAULT_CHUNK_SIZE = 5

GLOVE_DIMENSION = 100

MAX_SEQ_LEN = 120

reverse_mbti_index = {0: "ISTJ", 1: "ISFJ", 2: "INFJ", 3: "INTJ", 4: "ISTP", 5: "ISFP", 6: "INFP", 7: "INTP", 8: "ESTP",
                      9: "ESFP", 10: "ENFP", 11: "ENTP", 12: "ESTJ", 13: "ESFJ", 14: "ENFJ", 15: "ENTJ"}


def pad(sentence):
    if (len(sentence) > MAX_SEQ_LEN):
        return sentence[:MAX_SEQ_LEN]
    else:
        num_pads = MAX_SEQ_LEN - len(sentence)
        return sentence + "".join([" "] * num_pads)


def strip_numerics_special_chars(input_str):
    """
        Remove numerics from string
    """

    processed_str = []
    inclusion_char = []
    for c in string.ascii_lowercase:
        inclusion_char.append(c)
    for c in string.ascii_uppercase:
        inclusion_char.append(c)
    inclusion_char += ["?", ".", "!", " "]
    for c in input_str:
        if (c in inclusion_char):
            processed_str.append(c)
    return "".join(processed_str)


def remove_outlier_strings(input_str, dictionary):
    """
        Split upon spaces to get individual "words" and get rid of following
            - Empty strings
            - url containing www or http
            - Gutenberg

        Removes long strings that may associate with URLs rather than actual text.
        Also discard "words" that are a combination of floating punctuations.
    """
    words = input_str.split(" ")
    filtered_words = []
    for w in words:
        if (w == "" or w == "."): continue
        if ("www" in w or "http" in w): continue
        if (".." in w or ".," in w or ",." in w or "..." in w): continue
        if ("Gutenberg" in w or "gutenberg" in w): continue
        if (".org" in w or ".com" in w): continue
        if (w.strip(".?!").lower() not in dictionary): continue
        filtered_words.append(w)

    return " ".join(filtered_words)


def filter_sentences(input_str):
    """
        Splits upon punctuation and segments into sentence chunks.
    """

    input_str = input_str.replace("?", ".")
    input_str = input_str.replace("!", ".")

    sentences = input_str.split(".")
    filtered_sentences = []
    for s in sentences:
        words = s.split(" ")
        num_words = len(words)
        if (num_words > 4 and num_words < 70):
            filtered_sentences.append(" ".join(words))

    return filtered_sentences


def chunk_sentences(filtered_sentences, CHUNK_SIZE=DEFAULT_CHUNK_SIZE):
    """
        Group sentences of CHUNK_SIZE and output the concatenated string.
    """

    sentence_chunks = []
    while filtered_sentences:
        chunk = filtered_sentences[:CHUNK_SIZE]
        filtered_sentences = filtered_sentences[CHUNK_SIZE:]
        sentence_chunks.append(".".join(chunk) + ".\n")
    return sentence_chunks


def process_text(INPUT_FILE, dictionary, CHUNK_SIZE):
    f = open(INPUT_FILE, 'r')
    input_str = ""
    for line in f:
        input_str += " " + line.strip()

    processed_str = strip_numerics_special_chars(input_str)
    processed_str = remove_outlier_strings(processed_str, dictionary)
    filtered_sentences = filter_sentences(processed_str)
    return chunk_sentences(filtered_sentences, CHUNK_SIZE)


def process_input_sentence(sentence, dictionary, CHUNK_SIZE):
    processed_str = strip_numerics_special_chars(sentence)
    processed_str = remove_outlier_strings(processed_str, dictionary)
    filtered_sentences = filter_sentences(processed_str)
    return chunk_sentences(filtered_sentences, CHUNK_SIZE)


def get_embedding_baseline(chunk, glove_vectors):
    x = []
    sentence_str = chunk.replace(".", " ").lower()
    padded_sentence = pad(sentence_str)
    words = [w for w in padded_sentence.split(" ") if w != ""]
    print(len(words))

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


def find_mbti_type(sentence_chunks, glove_vectors, sess, W1, b1, W2, b2):
    if sess is None:
        sess = tf.Session()
        checkpoint = tf.train.latest_checkpoint('../models')
        saver = tf.train.import_meta_graph(checkpoint + '.meta')
        saver.restore(sess, checkpoint)
        sess.run(tf.global_variables_initializer())

    print_tensors_in_checkpoint_file(file_name='../models/baseline.ckpt', tensor_name='',
                                     all_tensors=True)
    results = []
    for chunk in sentence_chunks:
        input = get_embedding_baseline(chunk, glove_vectors)
        x_arr = np.array(input).astype(np.float32)
        x_arr = x_arr.reshape(1, GLOVE_DIMENSION)

        if W1 is None:
            W1 = tf.get_default_graph().get_tensor_by_name("W1:0")
        if b1 is None:
            b1 = tf.get_default_graph().get_tensor_by_name("b1:0")
        if W2 is None:
            W2 = tf.get_default_graph().get_tensor_by_name("W2:0")
        if b2 is None:
            b2 = tf.get_default_graph().get_tensor_by_name("b2:0")

        ymid = tf.matmul(x_arr, W1) + b1
        yhat = tf.add(tf.matmul(ymid, W2), b2)
        predict = tf.argmax(yhat, 1)

        mbti_result = reverse_mbti_index[sess.run(predict)[0]]
        print(mbti_result)
        results.append(mbti_result)
    return results


def find_prediction(sentence, dictionary, CHUNK_SIZE, sess, glove_vectors, W1, b1, W2, b2):
    chunks = process_input_sentence(sentence, dictionary, CHUNK_SIZE)
    mbti_types = find_mbti_type(chunks, glove_vectors, sess, W1, b1, W2, b2)
    if len(mbti_types) == 0:
        return "Insufficient Data"
    return mbti_types[0]


if __name__ == "__main__":
    num_args = len(sys.argv)
    CHUNK_SIZE = DEFAULT_CHUNK_SIZE

    dictionary = set([word.strip() for word in open('../data/words.txt', 'r')])
    glove_vectors = load_word_vectors("../data/glove.6B.100d.txt", GLOVE_DIMENSION)
    sentence_chunks = process_text('../input/input.txt',
                                   dictionary,
                                   CHUNK_SIZE)

    find_mbti_type(sentence_chunks, glove_vectors, None, None, None, None, None)

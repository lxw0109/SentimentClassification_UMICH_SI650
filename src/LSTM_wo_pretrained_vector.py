#!/usr/bin/env python3
# coding: utf-8
# File: LSTM_wo_pretrained_vector.py
# Author: lxw
# Date: 6/20/18 10:07 AM
"""
References:
[利用 Keras下的 LSTM 进行情感分析](https://blog.csdn.net/william_2015/article/details/72978387)

说明:
1. 没有去停用词
"""

import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


BATCH_SIZE = 128  # 32
# 把词汇表的大小设为一个定值，并且对于不在词汇表里的单词，把它们用UNK代替
MAX_VOCAB_SIZE = 2000  # 2000 -> 2300: 效果没有明显提升
# 根据句子的最大长度max_len，我们可以统一句子的长度，把短句用 0 填充
MAX_SENTENCE_LENGTH = 50  # 40 -> 50: 效果没有明显提升


def preprocessing():
    max_len = 0
    word_freqs = collections.Counter()
    sample_count = 0

    with open("../data/input/training.txt", "r") as f:
        for line in f:
            label, sentence = line.strip().split("\t")  # split()要求必须是str类型，不能是bytes类型
            words = nltk.word_tokenize(sentence.lower())  # type(words): list
            length = len(words)
            if length > max_len:
                max_len = length
            for word in words:
                word_freqs[word] += 1
            sample_count += 1

    print(f"Length of the longest sentence in the training set: {max_len}")  # 42
    print(f"vocabulary size: {len(word_freqs)}")  # 2329. 包括标点符号

    vocab_size = min(MAX_VOCAB_SIZE, len(word_freqs))
    # word_freqs.most_common(MAX_VOCAB_SIZE): <list of tuple>. [("i", 4705), ",", 4194, ".": 3558, "the": 3221, ...]
    word2index = {word[0]: idx+2 for idx, word in enumerate(word_freqs.most_common(MAX_VOCAB_SIZE))}
    word2index["PAD"] = 0  # "PAD"没有实际意义
    word2index["UNK"] = 1
    vocab_size += 2  # 加上"PAD", "UNK"
    index2word = {v: k for k, v in word2index.items()}
    return sample_count, vocab_size, word2index, index2word


def gen_train_val_data(sample_count, word2index, index2word):
    X = np.empty(sample_count, dtype=list)  # <ndarray of list>
    y = np.zeros(sample_count)
    idx = 0
    with open("../data/input/training.txt", "r") as f:
        for line in f:
            label, sentence = line.strip().split("\t")
            words = nltk.word_tokenize(sentence.lower())
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[idx] = seqs
            y[idx] = int(label)
            idx += 1

    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH, value=0)  # default: 从前面补0, 从前面截取(v2.3: 99.2%)
    # 从后面补0, 从后面截取. NOTE: 改成从后面补零和截取后，结果变差了一点.(v2.3: 99.0%)
    # X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH, value=0, padding="post", truncating="post")
    # 感觉shuffle=True必须的吧？训练样本中前面的label全是1后面的label全是0. shuffle=False好像对结果的准确率没有什么影响
    return train_test_split(X, y, test_size=0.333, random_state=1, shuffle=True)


def model_build(vocab_size):
    model = Sequential()
    # model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_SENTENCE_LENGTH, mask_zero=True,
    #                     name="embedding"))  # NOTE: mask_zero=True后，准确率降低了(v2.3: 99.2% -> 98.9%)
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_SENTENCE_LENGTH, mask_zero=False,
                        name="embedding"))
    """
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_SENTENCE_LENGTH))
    # the model will take as input an integer **matrix** of size (batch, input_length).
    # the largest integer(i.e. word index) in the input should be no larger than vocab_size (vocabulary size).
    # now model.output_shape == (None, input_length, output_dim), where None is the batch dimension.
    """
    model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, name="lstm"))
    model.add(Dense(units=1, activation="sigmoid", name="dense"))  # OK.
    # model.add(Dense(units=1, activation="softmax", name="dense"))  # OK. 效果变得非常差(v2.3: 99.2% -> 55.5%)
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
    return model


def model_train(model, X_train, y_train, X_val, y_val):
    NUM_EPOCHS = 100
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    lr_reduction = ReduceLROnPlateau(monitor="val_loss", patience=5, verbose=1, factor=0.2, min_lr=1e-5)
    model_path = "../data/output/models/best_model.hdf5"  # 保存到1个模型文件(因为文件名相同)
    # model_path = "../data/output/models/best_model_{epoch:02d}_{val_loss:.2f}.hdf5"  # 保存到多个模型文件
    checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    hist_obj = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1,
                         validation_data=(X_val, y_val), callbacks=[early_stopping, lr_reduction, checkpoint])

    # 绘制训练集和验证集的曲线
    plt.plot(hist_obj.history["acc"], label="Training Accuracy", color="green", linewidth=2)
    plt.plot(hist_obj.history["loss"], label="Training Loss", color="red", linewidth=1)
    plt.plot(hist_obj.history["val_acc"], label="Validation Accuracy", color="purple", linewidth=2)
    plt.plot(hist_obj.history["val_loss"], label="Validation Loss", color="blue", linewidth=1)
    plt.grid(True)  # 设置网格形式
    plt.xlabel("epoch")
    plt.ylabel("acc-loss")  # 给x, y轴加注释
    plt.legend(loc="upper right")  # 设置图例显示位置
    plt.show()


def model_evaluate(model, index2word, X_val, y_val):
    score, acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE)  # model.metrics: ["accuracy"]
    print(f"\nValidation score: {score:.3f}, accuracy: {acc:.3f}")
    print("预测\t真实\t句子")
    for i in range(5):
        idx = np.random.randint(len(X_val))
        x_test = X_val[idx].reshape(1, 40)
        y_label = y_val[idx]
        ypred = model.predict(x_test)[0][0]
        sent = " ".join([index2word[x] for x in x_test[0] if x != 0])
        print(f"{int(round(ypred))}\t{int(y_label)}\t{sent}")


def model_testing(model, word2index):
    input_sentences = ["I love reading.", "You are so boring.", "The orange doesn't taste very sweet.", "What a game."]
    X_test = np.empty(len(input_sentences), dtype=list)
    idx = 0
    for sentence in input_sentences:
        words = nltk.word_tokenize(sentence.lower())
        seq = []
        for word in words:
            if word in word2index:
                seq.append(word2index[word])
            else:
                seq.append(word2index["UNK"])
        X_test[idx] = seq
        idx += 1

    X_test = sequence.pad_sequences(X_test, maxlen=MAX_SENTENCE_LENGTH, value=0)  # shape: (4, 40)
    labels = [int(round(x[0])) for x in model.predict(X_test) ]
    label2word = {1: "积极", 0: "消极"}
    print()
    for i in range(len(input_sentences)):
        print(f"{label2word[labels[i]]}\t{input_sentences[i]}")


if __name__ == "__main__":
    # For reproducibility
    np.random.seed(2)
    tf.set_random_seed(2)

    sample_count, vocab_size, word2index, index2word = preprocessing()

    X_train, X_val, y_train, y_val = gen_train_val_data(sample_count, word2index, index2word)

    model = model_build(vocab_size)

    model_train(model, X_train, y_train, X_val, y_val)

    model_path = "../data/output/models/best_model.hdf5"
    model = load_model(model_path)

    model_evaluate(model, index2word, X_val, y_val)
    model_testing(model, word2index)
    """
    """

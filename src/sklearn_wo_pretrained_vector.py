#!/usr/bin/env python3
# coding: utf-8
# File: sklearn_wo_pretrained_vector.py
# Author: lxw
# Date: 6/21/18 11:32 AM

import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


BATCH_SIZE = 16
# 把词汇表的大小设为一个定值，并且对于不在词汇表里的单词，把它们用UNK代替
MAX_VOCAB_SIZE = 2000  # 2000 -> 2300: 效果没有明显提升(v2.3: 99.2% -> 99.1%)
# 根据句子的最大长度max_len，我们可以统一句子的长度，把短句用 0 填充
MAX_SENTENCE_LENGTH = 40  # 40 -> 50: 效果没有明显提升(v2.3: 99.2% -> 99.1%)


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

    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH, value=0)  # default: 从前面补0, 从前面截取
    print(f"X.shape: {X.shape}")  # X.shape: (7086, 40)
    # "shuffle=True" is essential for RF.
    # return train_test_split(X, y, test_size=0.333, random_state=1, shuffle=False)
    return train_test_split(X, y, test_size=0.333, random_state=1, shuffle=True)


# TODO: 1
def model_evaluate(model, index2word, X_val, y_val):
    score, acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE)  # model.metrics: ["accuracy"]
    print(f"\nValidation score: {score:.3f}, accuracy: {acc:.3f}")
    print("预测\t真实\t句子")
    for i in range(5):
        idx = np.random.randint(len(X_val))
        x_test = X_val[idx].reshape(1, MAX_SENTENCE_LENGTH)
        y_label = y_val[idx]
        ypred = model.predict(x_test)[0][0]
        sent = " ".join([index2word[x] for x in x_test[0] if x != 0])
        print(f"{int(round(ypred))}\t{int(y_label)}\t{sent}")


# TODO: 2
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


def train_val_predict(X_train, X_val, y_train, y_val):
    '''
    # 1. [NO]LR: LR算法的优点是可以给出数据所在类别的概率
    model = linear_model.LogisticRegression(C=1e5)
    """
    C: default: 1.0
    Inverse of regularization strength; must be a positive float. Like in support vector machines,
    smaller values specify stronger regularization.
    """
    # 2. [NO]NB: 也是著名的机器学习算法, 该方法的任务是还原训练样本数据的分布密度, 其在多分类中有很好的效果
    from sklearn import naive_bayes
    model = naive_bayes.GaussianNB()  # 高斯贝叶斯
    # 3. [OK]KNN:
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()  # 非常慢，感觉没法用(跑了半个多小时没反应)
    # 4. [OK]决策树: 分类与回归树(Classification and Regression Trees, CART)算法常用于特征含有类别信息
    # 的分类或者回归问题，这种方法非常适用于多分类情况
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    # 5. [NO]SVM: SVM是非常流行的机器学习算法，主要用于分类问题，
    # 如同逻辑回归问题，它可以使用一对多的方法进行多类别的分类
    from sklearn.svm import SVC
    model = SVC()

    # 6. [OK]MLP: 多层感知器(神经网络)
    from sklearn.neural_network import MLPClassifier
    # model = MLPClassifier(activation="relu", solver="adam", alpha=0.0001)
    # model = MLPClassifier(activation="identity", solver="adam", alpha=0.0001)
    # model = MLPClassifier(activation="logistic", solver="adam", alpha=0.0001)
    model = MLPClassifier(activation="tanh", solver="adam", alpha=0.0001)
    '''

    # 7. RF: 随机森林
    from sklearn.ensemble import RandomForestClassifier
    # n_jobs: If -1, the number of jobs is set to the number of cores.
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1, random_state=0)

    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    # print(f"\nmodel.feature_importances_: {model.feature_importances_}\n")

    print(f"classification_report:\n{classification_report(y_val, y_val_pred)}")  # y_true, y_pred
    print(f"confusion_matrix:\n{confusion_matrix(y_val, y_val_pred, labels=range(2))}")
    print("Mean accuracy score:", accuracy_score(y_val, y_val_pred))
    # cv = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Accuracy: {scores.mean():.2f}(+/-{scores.std() * 2:.2f})")
    print("model.score:", model.score(X_val, y_val))

    """
    predicted = model.predict(X_test)
    # print(predicted)
    # 把categorical数据转为numeric值，得到分类结果
    predicted = np.argmax(predicted, axis=1)
    predicted = pd.Series(predicted, name="Sentiment")
    submission = pd.concat([X_test_id, predicted], axis=1)
    # submission.to_csv("../data/output/submissions/sk_knn_submission.csv", index=False)
    submission.to_csv("../data/output/submissions/sk_rf_submission_matrix.csv", index=False)
    """


if __name__ == "__main__":
    # For reproducibility
    np.random.seed(2)

    sample_count, vocab_size, word2index, index2word = preprocessing()

    X_train, X_val, y_train, y_val = gen_train_val_data(sample_count, word2index, index2word)

    print(f"\nX_train.shape:{X_train.shape}\nX_val.shape:{X_val.shape}\n"
          f"y_train.shape:{y_train.shape}\ny_val.shape:{y_val.shape}\n")
    # X_train.shape: (4726, 40). X_val.shape: (2360, 40). y_train.shape: (4726,). y_val.shape: (2360,)

    train_val_predict(X_train, X_val, y_train, y_val)

    # TODO
    """
    model = None
    model_evaluate(model, index2word, X_val, y_val)
    model_testing(model, word2index)
    """

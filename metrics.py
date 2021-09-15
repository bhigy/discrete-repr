from collections import Counter
from itertools import groupby
from math import log2
import numpy as np


def segments_start(array):
    return [i for i in range(len(array)) if i == 0 or array[i] != array[i-1]]


def split_sequences(array, start):
    end = start[1:] + [len(array)]
    return [array[s:e] for s, e in zip(start, end)]


def coverage_top_1(labels, codes):
    '''
    Computes the coverage of label segments by the most frequent co-occuring
    code.
    '''
    start = segments_start(labels)
    segments = split_sequences(codes, start)
    return [sorted(Counter(s).values())[-1] / len(s) for s in segments]


def compute_joint_probability(x, y):
    labels_x = np.unique(x)
    idx_x = {v: i for i, v in enumerate(labels_x)}
    labels_y = np.unique(y)
    idx_y = {v: i for i, v in enumerate(labels_y)}
    counts_xy = np.zeros([len(labels_x), len(labels_y)])
    for xi, yi in zip(x, y):
        counts_xy[idx_x[xi], idx_y[yi]] += 1
    return labels_x, labels_y, counts_xy / len(x)


def conditional_entropy(x, y):
    labels_x, labels_y, p_xy = compute_joint_probability(x, y)
    p_y = np.sum(p_xy, axis=0)
    h_x_y = 0
    for i_x in range(len(labels_x)):
        for i_y in range(len(labels_y)):
            if p_xy[i_x, i_y] > 0:
                h_x_y -= p_xy[i_x, i_y] * log2(p_xy[i_x, i_y] / p_y[i_y])
    return h_x_y


def count_repetitions(array):
    return [len(list(v)) for _, v in groupby(array)]

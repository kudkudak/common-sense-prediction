import logging

import numpy as np


logger = logging.getLogger(__name__)

def featurize(term_batch, embedding):
    return np.array([[embedding[word] for word in term] for term in term_batch]).mean(axis=1)


def argsim_score(data_stream, embedding):
    targets = []
    argsims = []
    for data, target in data_stream.get_epoch_iterator():
        head = np.mean(embedding[data['head']], axis=1)
        tail = np.mean(embedding[data['tail']], axis=1)
        sim = np.einsum('ij,ik->i', head, tail)
        argsims.append(sim)
        targets.append(target)

    return np.concatenate(argsims), np.concatenate(targets)


def argsim_threshold(data_stream, embedding):
    scores, targets = argsim_score(data_stream, embedding)
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    threshold_acc = [np.mean((scores > t) == targets) for t in thresholds]
    threshold_argsim = thresholds[np.argmax(threshold_acc)]

    return threshold_argsim

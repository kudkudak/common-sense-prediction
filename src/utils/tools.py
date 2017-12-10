import logging

import numpy as np


logger = logging.getLogger(__name__)

def featurize(term_batch, embedding):
    return np.array([[embedding[word] for word in term] for term in term_batch]).mean(axis=1)

# TODO(kudkudak): Change mean to sum and divide by number of nonzeros
def argsim_score(data_stream, embedding):
    targets = []
    argsims = []
    for data, target in data_stream.get_epoch_iterator():
        head_len = data['head_mask'].sum(axis=1, keepdims=True)
        tail_len = data['tail_mask'].sum(axis=1, keepdims=True)
        # Warning: indexing works only for np.arrays
        head = np.sum(embedding[data['head']], axis=1)/head_len
        tail = np.sum(embedding[data['tail']], axis=1)/tail_len
        sim = np.einsum('ij,ji->i', head, tail.T)
        argsims.append(sim)
        targets.append(target)
    return np.concatenate(argsims), np.concatenate(targets)


def argsim_threshold(data_stream, embedding, N=1000):
    scores, targets = argsim_score(data_stream, embedding)
    if N == "all":
        thresholds = np.array(sorted(scores))
    else:
        thresholds = np.linspace(scores.min(), scores.max(), N)
    print("Testing {} thresholds".format(N))
    threshold_acc = [np.mean((scores > t) == targets) for t in thresholds]
    print("Max threshold acc " + str(np.max(threshold_acc)))

    threshold_argsim = thresholds[np.argmax(threshold_acc)]
    print("Best threshold {} in [{}, {}]".format(str(threshold_argsim), min(thresholds), max(thresholds)))


    return threshold_argsim

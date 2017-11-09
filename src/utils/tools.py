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
        sim = np.einsum('ij,ji->i', head, tail.T)
        argsims.append(sim)
        targets.append(target)

    """
    Bug proof, TODO: remove later:
    (Pdb) sim[0]
    -0.078352564280480014
    (Pdb) np.dot(head[0], tail[0])
    *** ValueError: shapes (1,200) and (1,200) not aligned: 200 (dim 1) != 1 (dim 0)
    (Pdb) ^[[A^[[D^[[D
    *** SyntaxError: invalid syntax (<stdin>, line 1)
    (Pdb) np.dot(head[0], tail[0].T)
    matrix([[ 0.08965615]])
    (Pdb) np.diag(head.dot(tail.T))[0]
    0.08965614676903999
    """

    return np.concatenate(argsims), np.concatenate(targets)


def argsim_threshold(data_stream, embedding, N=1000):
    scores, targets = argsim_score(data_stream, embedding)
    # TODO(kudkudak): If on dev, we can reuse the trick from callback to check all thresholds
    thresholds = np.linspace(scores.min(), scores.max(), N)
    print("Testing {} thresholds".format(N))
    threshold_acc = [np.mean((scores > t) == targets) for t in thresholds]
    print("Max threshold acc " + str(np.max(threshold_acc)))
    threshold_argsim = thresholds[np.argmax(threshold_acc)]

    return threshold_argsim

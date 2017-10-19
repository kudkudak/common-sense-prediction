import numpy as np
import tqdm


def _predict_scores(model, data_stream):
    scores = []
    targets = []
    for x_batch, y_batch in data_stream.get_epoch_iterator():
        scores.append(model.predict_on_batch(x_batch))
        targets.append(y_batch)

    return np.concatenate(scores), np.concatenate(targets)


def evaluate_fit_threshold(model, dev1_stream, dev2_stream, test_stream):
    scores_dev1, targets_dev1 = _predict_scores(model, dev1_stream)
    scores_dev2, targets_dev2 = _predict_scores(model, dev2_stream)
    scores_test, targets_test = _predict_scores(model, test_stream)

    thresholds = sorted(scores_dev1)
    threshold_accs = [np.mean((scores_dev1 > thr) == targets_dev1)
                      for thr in tqdm.tqdm(thresholds, total=len(thresholds))]
    optimal_threshold = thresholds[np.argmax(threshold_accs)]

    results = {
        "scores_dev": scores_dev1.flatten().tolist(),
        "scores_dev2": scores_dev2.flatten().tolist(),
        "scores_test": scores_test.flatten().tolist(),
        "threshold": optimal_threshold.item(),
    }
    return results



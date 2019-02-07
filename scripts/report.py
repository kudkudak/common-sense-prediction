"""Generate Results Table 1

given a list of directories for each model's run
generate a table of f1 scores bucketed by the novelty metric
"""
import os

import argh
from argh import arg
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA_DIR = os.environ['DATA_DIR']
TEST_PATH = os.path.join(DATA_DIR, "LiACL/conceptnet_my/test.txt")
TEST_DIST_PATH = os.path.join(DATA_DIR, "LiACL/conceptnet_my/test.txt.dists")
BUCKETS = ['entire', '(0%, 33%]', '(33%, 66$]', '(66%, 100%]']

def bucketed_scores(result_dir, test, test_dist_id):
    eval_results_path = os.path.join(result_dir, 'eval_results.json')
    with open(eval_results_path, 'r') as eval_results_file:
        eval_results = json.load(eval_results_file)

    length = len(test_dist_id) // 3
    bucket_scores = ["{0:.3f}".format(eval_results['test_thr_acc'])]
    for bucket_id in range(3):
        bucket_tests = test_dist_id[bucket_id*length:(bucket_id+1)*length]
        preds = np.array(eval_results['scores_test'])[bucket_tests] >= eval_results['threshold']
        bucket_f1 = f1_score(test.score.values[bucket_tests], preds)
        bucket_scores.append("{0:.3f}".format(bucket_f1)) # scores['test_thr_acc']

    return bucket_scores


@arg('result_dirs', type=str, nargs='+')
def table1(result_dirs):
    df = pd.DataFrame(index=BUCKETS)

    test = pd.read_csv(TEST_PATH, sep="\t", header=None)
    test.columns = ['rel', 'head', 'tail', 'score']

    test_dist = np.loadtxt(TEST_DIST_PATH)
    test_dist_id = np.argsort(test_dist)

    for result_dir in result_dirs:
        name = os.path.basename(result_dir)
        scores = bucketed_scores(result_dir, test, test_dist_id)

        df[name] = scores

    print(df.T)


if __name__ == '__main__':
    argh.dispatch_command(table1)


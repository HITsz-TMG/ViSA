import datasets
from scipy.stats import pearsonr
import itertools
import numpy as np
import pandas as pd
import math
import argparse

def shapley_weight(X):

    def v_consistency(S):
        if len(S) == 0:
            return 0
        X_S_mean = np.mean(X[:, list(S)], axis=1)
        correlation, _ = pearsonr(X_S_mean, X_mean)
        return correlation if not np.isnan(correlation) else 0

    num_models = X.shape[1]
    models = list(range(num_models))
    X_mean = np.mean(X, axis=1)

    shapley_values_consistency = np.zeros(num_models)

    for j in models:
        shapley_value = 0
        for S in itertools.chain.from_iterable(itertools.combinations(models, r) for r in range(num_models)):
            if j not in S:
                S_with_j = set(S) | {j}
                marginal_contribution = v_consistency(S_with_j) - v_consistency(set(S))
                shapley_value += marginal_contribution * (
                    math.factorial(len(S)) * math.factorial(num_models - len(S) - 1)
                ) / math.factorial(num_models)
        shapley_values_consistency[j] = shapley_value

    shapley_weights_consistency = shapley_values_consistency / np.sum(shapley_values_consistency)

    return shapley_weights_consistency

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs="+", type=str)
    parser.add_argument('--num_example', type=int, default=1000)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    score = []
    for data_path in args.data_path:
        data = datasets.load_from_disk(data_path)
        data = data.sort('id')
        score.append(data['agent score'])
    score = np.array(score).transpose((0,1))
    weight = shapley_weight(score[: args.num_example])
    print(weight)

    score = score * np.tile(weight, (len(score), 1)).sum(axis=1)

    data = data.map(lambda instance, index: {"agent score": score[index]}, with_indices=True, num_proc=16)
    data.save_to_disk(args.save_path)




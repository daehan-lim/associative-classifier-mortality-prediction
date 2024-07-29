import math
import random
import time
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import csv
import numpy as np
from tabulate import tabulate
import model
import warnings


# noinspection PyBroadException
if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    start_time = time.time()
    with open('data/dataset.csv', 'r') as file:
        dataset = [list(filter(None, row)) for row in csv.reader(file)]

    min_support = 0.2
    transactions_df = model.convert_trans_to_df(dataset)
    auc_sum = 0
    f1_sum = 0
    precision_sum = 0
    recall_sum = 0
    accuracy_sum = 0
    rules_count_sum = 0
    not_classified_sum = 0
    freq_itemsets_count_sum = 0

    for seed in range(10):
        print(f"\n\nseed: {seed}")
        print(f"min_supp: {min_support}")
        random.seed(seed)
        transactions_0 = pd.DataFrame(
            transactions_df[transactions_df['0']].reset_index(drop=True))
        transactions_1 = pd.DataFrame(
            transactions_df[transactions_df['1']].reset_index(drop=True))

        indices = list(range(0, len(transactions_0)))
        random.shuffle(indices)
        test_set_0 = transactions_0.iloc[indices[:417], :]
        training_set_0 = transactions_0.iloc[indices[417:], :]

        indices = list(range(0, len(transactions_1)))
        random.shuffle(indices)
        test_set_1 = transactions_1.iloc[indices[:43], :]
        training_set_1 = transactions_1.iloc[indices[43:], :]

        training_set = pd.concat([training_set_0, training_set_1])
        test_set = pd.concat([test_set_0, test_set_1])

        rules, freq_itemsets = model.train(transactions=training_set, m_min_support=min_support)
        rules_0 = [rule for rule in rules if rule['consequent'] == '0']
        rules_1 = [rule for rule in rules if rule['consequent'] == '1']
        sorted_rules = sorted(rules, key=lambda d: d['confidence'], reverse=True)

        start_time_pred = time.time()
        freq_itemsets_count = sum([len(sublist) for sublist in freq_itemsets[:-1]])
        freq_itemsets_count_sum += freq_itemsets_count
        sorted_rules = sorted_rules[:math.floor(freq_itemsets_count * min_support) + 1]
        test_transactions = test_set.drop(['1', '0'], axis=1).apply(lambda row: frozenset(row.index[row]),
                                                                    axis=1).tolist()
        y_test = test_set.apply(lambda row: 0 if row['0'] else 1, axis=1).tolist()
        y_test = np.array(y_test, dtype=np.uint8)
        scores = [model.predict_proba(object_o, sorted_rules) for object_o in test_transactions]
        scores = np.array(scores)
        y_pred = np.where(scores[:, 0] >= scores[:, 1], 0, 1)
        y_pred[(scores[:, 0] == 0) & (scores[:, 1] == 0)] = -1
        not_classified = np.sum(y_pred == -1)
        y_pred[y_pred == -1] = 0
        time_sec = time.time() - start_time_pred
        time_min = time_sec / 60
        print("\nProcessing time of %s: %.2f seconds (%.2f minutes)."
              % ("Predict", time.time() - start_time_pred, time_min))

        TP = np.sum(np.logical_and(y_pred == 1, y_test == 1))
        TN = np.sum(np.logical_and(y_pred == 0, y_test == 0))
        FP = np.sum(np.logical_and(y_pred == 1, y_test == 0))
        FN = np.sum(np.logical_and(y_pred == 0, y_test == 1))

        confusion_matrix = [
            ["1=died  0=alive", "Pred class = '1'", "Pred class = '0'", 'Total actual c'],
            ["Actual class = '1'", str(TP) + "\n(TP)", str(FN) + "\n(FN)",
             str(TP + FN) + "\n(Total actual c= '1')"],
            ["Actual class = '0'", str(FP) + "\n(FP)", str(TN) + "\n(TN)",
             str(FP + TN) + "\n(Total actual c= '0')"],
            ["Total pred c", str(TP + FP) + "\n(Total pred as '1')", str(FN + TN) + "\n(Total pred as '0')",
             str(len(test_set))],
        ]
        print(tabulate(confusion_matrix, headers='firstrow', tablefmt='fancy_grid'))

        precision = TP / (TP + FP)
        precision_sum += precision
        recall = TP / (TP + FN)
        recall_sum += recall
        F1 = 2 * recall * precision / (recall + precision)
        f1_sum += F1
        roc_auc = roc_auc_score(y_test, scores[:, 1])
        auc_sum += roc_auc
        accuracy = 100 * np.sum(y_test == y_pred) / len(y_test)
        accuracy_sum += accuracy
        not_classified_sum += not_classified
        rules_count_sum += len(sorted_rules)
        print(f"Pred as -1: {not_classified}")
        print(f"Precision: {round(precision, 3)}")
        print(f"Recall: {round(recall, 3)}")
        print(f"F1: {round(F1, 6)}")
        print(f"roc auc: {roc_auc}")
        print(f"Accuracy: {round(accuracy, 3)}%")
        print(f"Number of freq itemsets: {freq_itemsets_count}")
        print(f"Total Rules: {len(sorted_rules)}")
        print(f"Rules with class 0: {len(rules_0)}")
        print(f"Rules with class 1: {len(rules_1)}")
        print(f"Avg rule conf: {round(sum(rule['confidence'] for rule in sorted_rules) / len(sorted_rules), 3)}")
        print(f"Max rule conf: {round(sorted_rules[0]['confidence'], 3)}")
        print(f"Min rule conf: {round(sorted_rules[-1]['confidence'], 3)}\n")
        sorted_0 = [rule for rule in sorted_rules if rule['consequent'] == '0']
        sorted_1 = [rule for rule in sorted_rules if rule['consequent'] == '1']
        try:
            print(f"Avg conf for c0 rules: {round(sum(rule['confidence'] for rule in sorted_0) / len(sorted_0), 3)}")
            print(f"Max conf for c0 rules: {round(sorted_0[0]['confidence'], 3)}")
            print(f"Min conf for c0 rules: {round(sorted_0[-1]['confidence'], 3)}")
            print(f"Avg conf for c1 rules: {round(sum(rule['confidence'] for rule in sorted_1) / len(sorted_1), 3)}")
            print(f"Max conf for c1 rules: {round(sorted_1[0]['confidence'], 3)}")
            print(f"Min conf for c1 rules: {round(sorted_1[-1]['confidence'], 3)}")
        except Exception:
            pass
        print(classification_report(y_test, y_pred, zero_division=0))
        print()

    print("\n\nAvg")
    print(f"Roc auc (class 1): {auc_sum / 10}")
    print(f"f1: {f1_sum / 10}")
    print(f"Precision: {precision_sum / 10}")
    print(f"Recall: {recall_sum / 10}")
    print(f"Accuracy: {accuracy_sum / 10}")
    print(f"Total rules: {freq_itemsets_count_sum / 10}")
    print(f"Selected rules: {rules_count_sum / 10}")
    print(f"Not classified: {not_classified_sum / 10}")

    time_sec = time.time() - start_time
    time_min = time_sec / 60
    print("\nProcessing time of %s: %.2f seconds (%.2f minutes)."
          % ("whole code", time.time() - start_time, time_min))

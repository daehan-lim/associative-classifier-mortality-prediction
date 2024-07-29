import multiprocessing
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import apriori_mlx
from daehan_mlutil import utilities

classes = None
min_support = -1
transactions_df = None
class_support_count_dict = None


@utilities.timeit
def train(transactions, m_min_support):
    global min_support
    global transactions_df
    global class_support_count_dict
    global classes
    min_support = m_min_support

    rules = []
    transactions_df = transactions
    X_df = pd.DataFrame(transactions_df.drop(['1', '0'], axis=1))
    classes = [frozenset(['0']), frozenset(['1'])]
    class_support_count_dict = get_support_count_dict_df(classes, transactions_df)

    f1, previous_itemset_arr = apriori_mlx.apriori_of_size_1(X_df, min_support=min_support)
    f1 = f1.tolist()
    freq_itemsets = [f1]
    for item in f1:
        rules.extend(get_rules_per_item(item, classes, class_support_count_dict, transactions_df))

    k = 0
    while freq_itemsets[k] is not None and len(freq_itemsets[k]) > 0:
        k_freq_itemsets, previous_itemset_arr = apriori_mlx.apriori_of_size_k(
            X_df, previous_itemset_arr, min_support=min_support, k=k + 2, low_memory=True)
        if not k_freq_itemsets.empty:
            k_freq_itemsets = k_freq_itemsets.tolist()

            # Uncomment the following lines for parallel processing (works only on Linux):
            # with multiprocessing.Pool() as pool:
            #     result = pool.map(get_rules_per_item_parallel, k_freq_itemsets)
            # rules_to_extend = [x[0] for x in result if x != []]
            # rules.extend(rules_to_extend)

            # Sequential processing of rules
            for item in k_freq_itemsets:
                rules.extend(get_rules_per_item(item, classes, class_support_count_dict, transactions_df))

            freq_itemsets.append(k_freq_itemsets)
        else:
            freq_itemsets.append(None)
        k += 1

    return rules, freq_itemsets


# noinspection PyTypeChecker,PyUnresolvedReferences
def get_rules_per_item_parallel(item):
    return get_rules_per_item(item, classes, class_support_count_dict, transactions_df)


def get_rules_per_item(itemset, classes, class_supp_count_dict, transactions_df):
    rules = []
    cls0 = get_item_support_count_df(itemset | classes[0], transactions_df) / class_supp_count_dict[classes[0]]
    cls1 = get_item_support_count_df(itemset | classes[1], transactions_df) / class_supp_count_dict[classes[1]]
    if cls0 >= cls1:
        rules.append({'antecedent': itemset, 'consequent': '0', 'confidence': 1 - cls1 / cls0})
    else:
        rules.append({'antecedent': itemset, 'consequent': '1', 'confidence': 1 - cls0 / cls1})
    return rules


def convert_trans_to_df(transaction):
    te = TransactionEncoder()
    te_ary = te.fit_transform(transaction)
    data_df = pd.DataFrame(te_ary, columns=te.columns_)
    return data_df


def get_item_support_count_df(itemset: frozenset, df, negated=False):
    """
    Efficient support calculation
    :param negated: Whether it should find the support of positive or negated items
    :param itemset: Items need to be in transaction
    :param df: DataFrame of transactions
    :return: support of itemset
    """
    subset = df[list(itemset)] if negated is False else ~df[list(itemset)]
    # subset['support'] = subset.all(axis=1) # returns column
    support = subset.all(axis=1).sum()
    return support


def get_support_count_dict_df(ck, transactions_df):
    item_support_count = {}
    for item in ck:
        item_support_count[item] = get_item_support_count_df(item, transactions_df)
    return item_support_count


def predict_proba(object_o, rules_set):
    best_k = 3
    scores = [0, 0]
    count_0 = 0
    count_1 = 0
    for rule in rules_set:
        if (rule['antecedent']).issubset(object_o):
            if count_0 < best_k and rule['consequent'] == '0':
                count_0 += 1
                scores[0] += rule['confidence']
            elif count_1 < best_k and rule['consequent'] == '1':
                count_1 += 1
                scores[1] += rule['confidence']
        elif count_0 >= best_k and count_1 >= best_k:
            break
    return scores

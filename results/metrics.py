import argparse
import nltk.translate.gleu_score as gleu
import nltk.translate.bleu_score as bleu

from rouge import Rouge
import pandas as pd
import nlgeval
from nlgeval import compute_metrics
import numpy as np
from tqdm import trange


def sentence_score(reference, hypothesis):
    '''
    input : reference - list of reference sentence
    hypothesis - list of hypothesis sentence
    '''
    tp, fp, fn = 0, 0, 0

    reference_set = set(reference)
    hypothesis_set = set(hypothesis)

    for token in hypothesis:
        if token in reference_set:
            tp += 1
        else:
            fp += 1

    for token in reference:
        if token not in hypothesis_set:
            fn += 1
    return tp, fp, fn


def corpus_macro_score(ref_line, hypo_line):
    tp, fp, fn = 0, 0, 0

    for ref, hypo in zip(ref_line, hypo_line):
        sam_tp, sam_fp, sam_fn = sentence_score(ref.split(), hypo.split())
        tp += sam_tp
        fp += sam_fp
        fn += sam_fn

    return tp, fp, fn


def precision(tp, fp):
    return float(tp) / (tp + fp) if (tp + fp) > 0 else 0.


def recall(tp, fn):
    if (tp + fn) == 0:
        return 0.
    return float(tp) / (tp + fn)


def f_measure(tp, fp, fn, beta=1):
    f_percision = precision(tp, fp)
    f_recall = recall(tp, fn)
    if f_percision == 0 or f_recall == 0:
        return 0.
    else:
        return (1 + beta ** 2) * (f_percision * f_recall) / ((beta ** 2 * f_percision) + f_recall)


'''
def accuracy_macro_score(ref_line, hypo_line):
    correct = 0
    w = open('./different_result.txt','w')    
    for ref, hypo in zip(ref_line, hypo_line):
        if ref.strip() == hypo.strip():
            correct += 1
        else:
            w.write('---------------------\n'+ref.strip()+'\n'+hypo.strip()+'\n--------------------------\n')
    w.close()
    return correct, len(ref_line)    
'''


def accuracy_macro_score_final(ref_line, hypo_line):
    correct = 0
    total = 0
    w = open('./different_result.txt', 'w')
    for ref, hypo in zip(ref_line, hypo_line):
        ref = ref.strip()
        hypo = hypo.strip()
        flag = 0
        for index in range(len(ref)):
            try:
                if ref[index] != hypo[index]:
                    flag = 1
                else:
                    correct += 1
            except IndexError:
                flag = 1
                continue
        total += len(ref)
        if flag == 1:
            w.write('---------------------\n' + ref.strip() + '\n' + hypo.strip() + '\n--------------------------\n')
    w.close()
    return correct, total


def accuracy(length, correct):
    return float(correct) / length


def score_gleu(reference, hypothesis):
    score = 0
    for ref, hyp in zip(reference, hypothesis):
        score += gleu.sentence_gleu([ref.split()], hyp.split())
    return float(score) / len(reference)


def gleu_list(reference, hypothesis):
    scoreList = []
    for ref, hyp in zip(reference, hypothesis):
        score = gleu.sentence_gleu([ref.split()], hyp.split())
        scoreList.append(float(score))
    return scoreList


def bleu_list(reference, hypothesis):
    scoreList = []
    for ref, hyp in zip(reference, hypothesis):
        score = bleu.sentence_bleu([ref.split()], hyp.split())
        scoreList.append(float(score))
    return scoreList


def rouge_list(reference, hypothesis):
    rouge = Rouge()

    scoreList = []
    for ref, hyp in zip(reference, hypothesis):
        scores = rouge.get_scores(hyp, ref, avg=True)
        rouge_l_score = scores["rouge-l"]["r"]

        scoreList.append(rouge_l_score)
    return scoreList


def mrr_N_sentence(ground, pred, n):
    score = 0.0
    for rank, item in enumerate(pred[:n]):
        if str(item) in ground:
            score = 1.0 / (rank + 1.0)
            break

    return score


def mrr_N_List(preds, gloden, n):
    if "txt" in preds:
        preds = txt2DataFrame(preds, n)
    else:
        preds = pd.read_csv(preds, header=None)
    if "txt" in gloden:
        gloden = txt2DataFrame(gloden, 1).astype(str)
    else:
        gloden = pd.read_csv(gloden, header=None).astype(str)
    # ls_gold = gloden.values.tolist()
    # ls_gold = [''.join(x).lower() for x in ls_gold]
    # gloden = pd.DataFrame(ls_gold)
    score = 0
    total = 0
    for i in range(gloden.shape[0]):
        score += mrr_N_sentence(gloden.iloc[i, 0], preds.iloc[i], n)
        total = total + 1

    return score / total


def score_gleu(reference, hypothesis):
    score = 0
    for ref, hyp in zip(reference, hypothesis):
        score += gleu.sentence_gleu([ref.split()], hyp.split())
    return float(score) / len(reference)


def txt2DataFrame(file, n):
    file = open(file, 'r', encoding='utf8')
    txt = file.readlines()
    data = []
    one = []
    for index in range(len(txt)):
        one.append(txt[index])
        if (index + 1) % n == 0:
            data.append(one)
            one = []
    data = pd.DataFrame(data)
    return data


# exact accuracy goes through each integer in each array in the actual and prediction arrays.
# good for multilabel classification problems (this is the same as the Exact Match metric)
def EM(preds, gloden, n):
    if "txt" in preds:
        preds = txt2DataFrame(preds, n)
    else:
        preds = pd.read_csv(preds, header=None)
    if "txt" in gloden:
        gloden = txt2DataFrame(gloden, 1)
    else:
        gloden = pd.read_csv(gloden, header=None)
    correct = 0
    total = 0
    for i in range(gloden.shape[0]):
        for j in range(n):
            if gloden.iloc[i, 0] == preds.iloc[i, j]:
                correct = correct + 1
                break
        total = total + 1

    return correct / total


def compute(preds, gloden):
    t = open(gloden, 'r', encoding='utf8')
    p = open(preds, 'r', encoding='utf8')
    tline = t.readlines()
    pline = p.readlines()
    gleu_result = score_gleu(tline, pline)  # gleu_result = score_gleu(tline, pline)
    # chrf_result = score_chrf(tline, pline)
    print('GLEU : ', gleu_result)
    # print('CHRF : ', chrf_result)

    metrics_dict = compute_metrics(hypothesis=preds,
                                   references=[gloden], no_skipthoughts=True, no_glove=True)

    return gleu_result, metrics_dict


def compute_map(results_dict):
    ap_sum = 0.0
    num_queries = len(results_dict)
    for query, results in results_dict.items():
        relevant_docs = [i for i, (doc_id, rel) in enumerate(results) if rel == 1]
        if not relevant_docs:
            continue
        precision_sum = 0.0
        for i, (doc_id, rel) in enumerate(results):
            if i in relevant_docs:
                precision = (relevant_docs.index(i) + 1) / (i + 1)
                precision_sum += precision
        ap = precision_sum / len(relevant_docs)
        ap_sum += ap
    map_score = ap_sum / num_queries
    return map_score


def com_map(res):
    ap_sum = 0.0
    num = len(res)
    for ls in res:
        if ls.count(1) == 0:
            continue
        precision_sum = 0.0
        for item in ls:
            if item == 1:
                precision = 1
                precision_sum += precision
                break
        ap = precision_sum
        ap_sum += ap
    map_score = ap_sum / num
    return map_score


if __name__ == "__main__":
    frompath = r".\mrr_results"
    pred = frompath + r'\pred.csv'
    gold = frompath + r'\gold.csv'
    mrr = mrr_N_List(pred, gold, 10)

    print("MRR:{:.3f}".format(mrr))

    path = r'.\mrr_results'
    df_pred = pd.read_csv(path + r'\pred.csv', header=None).astype(str)
    df_gold = pd.read_csv(path + r'\gold.csv', header=None).astype(str)
    results_dict = {}
    for i in range(len(df_gold)):
        for j in range(0, 3):
            if df_pred.loc[i, j] == df_gold.loc[i, 0]:
                df_pred.loc[i, j] = 1
            else:
                df_pred.loc[i, j] = 0
    df_pred.to_csv(path + r'\score.csv', header=False, index=False)

    df_score = pd.read_csv(path + r'\score.csv', header=None)

    ls = np.array(df_score).tolist()

    map_score = com_map(ls)

    print('MAP:{:.3f}'.format(map_score))

    # EM@n
    for index in range(1, 6):
        pred_list = pd.read_csv(rf'.\beam_search_{index}\pred.csv',
                                header=None).values.tolist()
        gold_list = pd.read_csv(fr'.\beam_search_{index}\gold.csv',
                                header=None).values.tolist()

        pred_list_index = [pred_list[i:i + index] for i in range(0, len(pred_list), index)]

        em_list = [int(np.any([str(pred) == str(gold).lower() for pred in pred_index])) for gold, pred_index in

                   zip(gold_list, pred_list_index)]

        em_at_index = np.mean(em_list)
        print("EM@{}:{:.2f}%".format(index, em_at_index * 100))

import logging
import os
import shutil
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import sklearn
from algo.transformers.evaluation import pearson_corr, spearman_corr
from algo.transformers.run_model import QuestModel
from examples.common.util.draw import draw_scatterplot
from examples.common.util.normalizer import fit, un_fit
from examples.en_zh.transformer_config import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, transformer_config, SEED, \
    RESULT_FILE, RESULT_IMAGE, EVALUATION_FILE, NUM_LABELS

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

# TRAIN_FILE = "data/en-zh/train.enzh.df.short.tsv"
# TEST_FILE = "data/en-zh/dev.enzh.df.short.tsv"

# train = pd.read_csv(TRAIN_FILE, sep='\t', error_bad_lines=False)
# test = pd.read_csv(TEST_FILE, sep='\t', error_bad_lines=False)

with open("data/en-zh/train_majority.pickle", 'rb') as f:
    train = pickle.loads(f.read())

with open("data/en-zh/test_majority.pickle", 'rb') as f:
    test = pickle.loads(f.read())
test.reset_index(drop=True, inplace=True)

with open("data/en-zh/new_train_vis.pickle", 'rb') as f:
    vis1 = pickle.loads(f.read())

with open("data/en-zh/new_test_vis.pickle", 'rb') as f:
    vis2 = pickle.loads(f.read())

# train
col_name = train.columns.tolist()
col_name.insert(3,'vis')
train_vis = train.reindex(columns=col_name)
train_vis['vis'] = train_vis['vis'].astype(object)
for idx, v in enumerate(vis1):
    train_vis['vis'][idx] = v
    train_vis['score'][idx] = int(train_vis['score'][idx])

# test
col_name = test.columns.tolist()
col_name.insert(3,'vis')
test_vis = test.reindex(columns=col_name)
test_vis['vis'] = test_vis['vis'].astype(object)
for idx, v in enumerate(vis2):
    test_vis['vis'][idx] = v
    test_vis['score'][idx] = int(test_vis['score'][idx])

train = train_vis[['original', 'translation', 'score', 'vis']]

test = test_vis[['original', 'translation', 'score', 'vis']]

train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'score': 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b', 'score': 'labels'}).dropna()

# train = train[:20]
# test = test[:10]

if NUM_LABELS == 1:
    train = fit(train, 'labels')
    test = fit(test, 'labels')
# print(train)
if transformer_config["evaluate_during_training"]:
    if transformer_config["n_fold"] > 1:
        test_preds = np.zeros((len(test), transformer_config["n_fold"]))
        for i in range(transformer_config["n_fold"]):

            if os.path.exists(transformer_config['output_dir']) and os.path.isdir(transformer_config['output_dir']):
                shutil.rmtree(transformer_config['output_dir'])

            model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=NUM_LABELS, use_cuda=torch.cuda.is_available(),
                               args=transformer_config)
            train, eval_df = train_test_split(train, test_size=0.11, random_state=SEED*i)
            # model.train_model(train, eval_df=eval_df)
            if NUM_LABELS == 1:
                model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                                  mae=mean_absolute_error)
            else:
                model.train_model(train, eval_df=eval_df, multi_label=False, acc=sklearn.metrics.accuracy_score, f1=sklearn.metrics.f1_score,
                                  precision=sklearn.metrics.precision_score, recall=sklearn.metrics.recall_score)
            model = QuestModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=NUM_LABELS, use_cuda=torch.cuda.is_available(), args=transformer_config)
            # result, model_outputs, wrong_predictions = model.eval_model(test, acc=sklearn.metrics.accuracy_score)
            if NUM_LABELS == 1:
                result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                                            spearman_corr=spearman_corr,
                                                                            mae=mean_absolute_error)
            else:
                result, model_outputs, wrong_predictions = model.eval_model(test, acc=sklearn.metrics.accuracy_score, f1=sklearn.metrics.f1_score,
                precision=sklearn.metrics.precision_score, recall=sklearn.metrics.recall_score)
            test_preds[:, i] = model_outputs

        test['predictions'] = test_preds.mean(axis=1)

    else:
        model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=NUM_LABELS, use_cuda=torch.cuda.is_available(),
                           args=transformer_config)
        train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
        # model.train_model(train, eval_df=eval_df, acc=sklearn.metrics.accuracy_score)
        if NUM_LABELS == 1:
            model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                              mae=mean_absolute_error)
        else:
            model.train_model(train, eval_df=eval_df, multi_label=False, acc=sklearn.metrics.accuracy_score, f1=sklearn.metrics.f1_score,
                              precision=sklearn.metrics.precision_score, recall=sklearn.metrics.recall_score)
        model = QuestModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=NUM_LABELS,
                           use_cuda=torch.cuda.is_available(), args=transformer_config)
        # result, model_outputs, wrong_predictions = model.eval_model(test,acc=sklearn.metrics.accuracy_score)
        if NUM_LABELS == 1:
            result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                                        spearman_corr=spearman_corr,
                                                                        mae=mean_absolute_error)
        else:
            result, model_outputs, wrong_predictions = model.eval_model(test, multi_label= False,
                                                                        acc=sklearn.metrics.accuracy_score, f1=sklearn.metrics.f1_score,
                                                                        precision=sklearn.metrics.precision_score, recall=sklearn.metrics.recall_score)
        # print("Final result", result, "model_outputs", model_outputs, "wrong_predictions", wrong_predictions)
        test['predictions'] = model_outputs


else:
    model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=NUM_LABELS, use_cuda=torch.cuda.is_available(),
                       args=transformer_config)
    model.train_model(train)
    # result, model_outputs, wrong_predictions = model.eval_model(test,acc=sklearn.metrics.accuracy_score)
    if NUM_LABELS == 1:
        model.train_model(train, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
        result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr, mae=mean_absolute_error)
    else:
        model.train_model(train, multi_label=True, acc=sklearn.metrics.accuracy_score, f1=sklearn.metrics.f1_score,
                          precision=sklearn.metrics.precision_score, recall=sklearn.metrics.recall_score)
        result, model_outputs, wrong_predictions = model.eval_model(test, multi_label= False,
                                                                    acc=sklearn.metrics.accuracy_score, f1=sklearn.metrics.f1_score,
                                                                    precision=sklearn.metrics.precision_score, recall=sklearn.metrics.recall_score)
    test['predictions'] = model_outputs

if NUM_LABELS == 1:
    test = un_fit(test, 'labels')
    test = un_fit(test, 'predictions')
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
# draw_scatterplot(test, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), MODEL_TYPE + " " + MODEL_NAME, NUM_LABELS)
draw_scatterplot(test, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), os.path.join(TEMP_DIRECTORY, EVALUATION_FILE), MODEL_TYPE + " " + MODEL_NAME, NUM_LABELS, result)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from algo.transformers.evaluation import pearson_corr, spearman_corr, rmse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sns.set()


def draw_scatterplot(data_frame, real_column, prediction_column, img_path, eval_path, topic, num_labels, result):
    data_frame = data_frame.sort_values(real_column)
    sort_id = list(range(0, len(data_frame.index)))
    data_frame['id'] = pd.Series(sort_id).values

    if num_labels == 1:
        pearson = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
        spearman = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
        rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
        textstr = 'RMSE=%.3f\nPearson Correlation=%.3f\nSpearman Correlation=%.3f' % (rmse_value, pearson, spearman)
        metrics = {'RMSE': rmse_value, 'Pearson Correlation': pearson, 'Spearman Correlation': spearman}
        ax = data_frame.plot(kind='scatter', x='id', y=real_column, color='DarkBlue', label='z_mean', title=topic)
        ax = data_frame.plot(kind='scatter', x='id', y=prediction_column, color='DarkGreen', label='predicted z_mean',
                        ax=ax)
    else:
        acc = accuracy_score(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
        f1 = f1_score(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
        precision = precision_score(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
        recall = recall_score(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
        textstr = 'accuracy=%.3f\nf1 score=%.3f\nprecision=%.3f\nrecall=%.3f' % (acc, f1, precision, recall)
        metrics = {'Accuracy': acc, 'F1 Score': f1, 'Precision': precision, 'Recall': recall}
        ax = data_frame.plot(kind='scatter', x='id', y=real_column, color='DarkBlue', label='label', title=topic)
        ax = data_frame.plot(kind='scatter', x='id', y=prediction_column, color='DarkGreen', label='predicted label',
                        ax=ax)

    print(textstr)
    # file = open(eval_path, 'wb')
    # pickle.dump(metrics, file)
    # file.close()
    metrics.update(result)
    # fw = open(eval_path,'w+')
    # fw.write(str(metrics))
    # fw.close()
    with open(eval_path,'w+') as writer:
        for key in sorted(metrics.keys()):
            writer.write("{} = {}\n".format(key, str(metrics[key])))

    # ax = data_frame.plot(kind='scatter', x='id', y=real_column, color='DarkBlue', label='z_mean', title=topic)
    # ax = data_frame.plot(kind='scatter', x='id', y=prediction_column, color='DarkGreen', label='predicted z_mean',
    #                 ax=ax)
    ax.text(1500, 0.05, textstr, fontsize=12)

    fig = ax.get_figure()
    fig.savefig(img_path)

# -*-coding:GBK -*-
import os
import numpy as np
import torch

import logging
from collections import Counter


# config for training
class Config():

    def __init__(self):
        self.learning_rate = 1e-3
        self.dropout = 0.1
        self.epoch = 5
        self.embedding_dim = 300
        self.hidden_dim = 256
        self.batch_size = 16

        self.max_seq_length = 100
        self.logging_steps = 50
        self.warmup_steps = 0
        self.model_name_or_path = os.path.join('./ckpts', 'bert-base-chinese')
        self.trained_model_path = 'ckpts'
        self.data_dir = 'data/'
        self.output_path = './output'
        self.save_model = 'NERmodel.pth'
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.PAD_TAG = 'PAD'
        self.model = 'LSTM'

        if self.model == 'bert_bilstm_crf':
            self.base_path = os.path.abspath('./')
            self._init_train_config()

    def _init_train_config(self):
        self.label_list = get_labels()
        self.use_gpu = True
        self.device = "cuda"
        self.checkpoints = False  # 使用预训练模型时设置为False
        self.model = 'bert_bilstm_crf'  # 可选['bert_bilstm_crf','bilstm_crf','bilstm','crf','hmm']

        # 输入数据集、日志、输出目录
        self.train_file = os.path.join(self.base_path, 'data/train.txt')
        self.test_file = os.path.join(self.base_path, 'data/test.txt')
        self.log_path = os.path.join(self.base_path, 'logs')
        # self.output_path = os.path.join(self.base_path, 'output', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        self.output_path = os.path.join(self.base_path, 'output', self.model)
        self.trained_model_path = os.path.join(self.base_path, 'ckpts', self.model)
        self.model_name_or_path = os.path.join(self.base_path, 'ckpts', 'bert-base-chinese') if not self.checkpoints \
            else self.trained_model_path

        # 以下是模型训练参数
        self.embedding_dim = 768
        self.do_train = True
        self.do_eval = False
        self.need_birnn = True
        self.do_lower_case = True
        self.rnn_dim = 128
        self.max_seq_length = 128
        self.batch_size = 16
        self.num_train_epochs = 2
        self.ckpts_epoch = 1
        self.gradient_accumulation_steps = 1
        self.learning_rate = 3e-5
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.logging_steps = 50
        self.remove_O = False

def get_labels():
    """读取训练数据获取标签"""
    labels = ['I-PER', 'I-ORG', 'I-LOC', 'B-LOC', 'B-PER', 'O', 'B-ORG']
    labels.extend(['<START>', '<STOP>', 'PAD'])

    return labels


def build_vocab(data_dir):
    """
    :param data_dir: the dir of train_corpus.txt
    :return: the word dict for training
    """

    if (os.path.isfile('word_dict.npy')):
        word_dict = np.load('word_dict.npy', allow_pickle=True).item()
        return word_dict
    else:
        word_dict = {}
        train_corpus = data_dir + 'train' + '_corpus.txt'
        lines = open(train_corpus, encoding='utf-8').readlines()
        for line in lines:
            word_list = line.split()
            for word in word_list:
                if (word not in word_dict):
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        word_dict = dict(sorted(word_dict.items(), key=lambda x: x[1], reverse=True))
        np.save('word_dict.npy', word_dict)
        word_dict = np.load('word_dict.npy', allow_pickle=True).item()
        return word_dict


def build_dict(word_dict):
    """
    :param word_dict:
    :return: word2id and tag2id
    """

    # 7 is the label of pad
    tag2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'PAD': 7}
    word2id = {}
    for key in word_dict:
        word2id[key] = len(word2id)
    word2id['unk'] = len(word2id)
    word2id['pad'] = len(word2id)
    return word2id, tag2id


def cal_max_length(data_dir):
    file = data_dir + 'train' + '_corpus.txt'
    lines = open(file, encoding='utf-8').readlines()
    max_len = 0
    for line in lines:
        if (len(line.split()) > max_len):
            max_len = len(line.split())

    return max_len


class Metrics(object):
    """用于评价模型，计算每个标签的精确率，召回率，F1分数"""
    def __init__(self, golden_tags, predict_tags, remove_O=False):
        # 所有句子tags的拼接[[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = self.flatten_lists(golden_tags)
        self.predict_tags = self.flatten_lists(predict_tags)
        if remove_O:  # 将O标记移除，只关心实体标记
            self._remove_Otags()

        # 辅助计算的变量
        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)

        self.precision_scores = self.cal_precision()
        self.recall_scores = self.cal_recall()
        self.f1_scores = self.cal_f1()

    def flatten_lists(self, lists):
        flatten_list = []
        for l in lists:
            if type(l) == list:
                flatten_list += l
            else:
                flatten_list.append(l)
        return flatten_list

    def cal_precision(self):
        precision_scores = {}
        for tag in self.tagset:
            precision_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                self.predict_tags_counter[tag]

        return precision_scores

    def cal_recall(self):
        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                self.golden_tags_counter[tag]
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2*p*r / (p+r+1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores

    def report_scores(self):
        """将结果用表格的形式打印出来，像这个样子：
                      precision    recall  f1-score   support
              B-LOC      0.775     0.757     0.766      1084
              I-LOC      0.601     0.631     0.616       325
             B-MISC      0.698     0.499     0.582       339
             I-MISC      0.644     0.567     0.603       557
              B-ORG      0.795     0.801     0.798      1400
              I-ORG      0.831     0.773     0.801      1104
              B-PER      0.812     0.876     0.843       735
              I-PER      0.873     0.931     0.901       634
          avg/total      0.779     0.764     0.770      6178
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        logging.info(header_format.format('', *header))

        # 打印每个标签的 精确率、召回率、f1分数
        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        for tag in self.tagset:
            logging.info(row_format.format(
                tag,
                self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1_scores[tag],
                self.golden_tags_counter[tag]
            ))

        # 计算并打印平均值
        avg_metrics = self.cal_avg_metrics()
        logging.info(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            len(self.golden_tags)
        ))


    def count_correct_tags(self):
        """计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算"""
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict

    def cal_avg_metrics(self):
        avg_metrics = {}
        total = len(self.golden_tags)

        avg_metrics['precision'] = 0.
        avg_metrics['recall'] = 0.
        avg_metrics['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            avg_metrics['precision'] += self.precision_scores[tag] * size
            avg_metrics['recall'] += self.recall_scores[tag] * size
            avg_metrics['f1_score'] += self.f1_scores[tag] * size

        for metric in avg_metrics.keys():
            avg_metrics[metric] /= total

        return avg_metrics

    def _remove_Otags(self):
        length = len(self.golden_tags)
        O_tag_indices = [i for i in range(length) if self.golden_tags[i] == 'O']
        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags) if i not in O_tag_indices]
        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags) if i not in O_tag_indices]
        logging.info("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))

    def report_confusion_matrix(self):
        """计算混淆矩阵"""
        logging.info("Confusion Matrix:")
        tag_list = list(self.tagset)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 有极少数标记没有出现在golden_tags，但出现在predict_tags，跳过这些标记
                continue

        row_format_ = '{:>7} ' * (tags_size+1)
        logging.info(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            logging.info(row_format_.format(tag_list[i], *row))


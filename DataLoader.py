#-*-coding:GBK -*-
import os.path

from torch.utils.data import Dataset


class InputData(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, token_type_ids, attention_mask, label_id):
        """
        :param input_ids:       单词在词典中的编码
        :param attention_mask:  指定 对哪些词 进行self-Attention操作
        :param token_type_ids:  区分两个句子的编码（上句全为0，下句全为1）
        :param label_id:        标签的id
        """
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_id = label_id


class NERDataset(Dataset):
    def __init__(self, config, tokenizer, mode="train"):
        # text: a list of words, all text from the training dataset
        super(NERDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        if mode == "train":
            self.file_path = os.path.join(config.data_dir, 'train.txt')
        elif mode == "test":
            self.file_path = os.path.join(config.data_dir, 'test.txt')
        elif mode == "eval":
            self.file_path = os.path.join(config.data_dir, 'test.txt')
        else:
            raise ValueError("mode must be one of train, or test")

        self.tdt_data = self.get_data()
        self.len = len(self.tdt_data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx, tag2id=None):
        """
        对指定数据集进行预处理，进一步封装数据，包括:
        tdt_data：[InputData(guid=index, text=text, label=label)]
        feature：BatchEncoding( input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                label_id=label_ids)
        data_f： 处理完成的数据集, TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)
        """
        label_map = {label: i for i, label in enumerate(self.config.label_list)}
        max_seq_length = self.config.max_seq_length

        data = self.tdt_data[idx]
        data_text_list = data.text.split(" ")
        data_label_list = data.label.split(" ")
        assert len(data_text_list) == len(data_label_list)

        features = self.tokenizer(''.join(data_text_list), padding='max_length', max_length=max_seq_length, truncation=True)
        label_ids = [label_map[label] for label in data_label_list]
        label_ids = [label_map['<START>']] + label_ids + [label_map['<STOP>']]
        if len(label_ids) > max_seq_length:
            label_ids = label_ids[:max_seq_length]
        else:
            while len(label_ids) < max_seq_length:
                label_ids.append(label_map['PAD'])

        features.data['label_ids'] = label_ids

        return features

    def read_file(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines, words, labels = [], [], []
            for line in f.readlines():
                contends = line.strip()
                tokens = line.strip().split()
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label, word = [], []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words, labels = [], []
        return lines

    def get_data(self):
        '''数据预处理并返回相关数据'''
        lines = self.read_file()
        tdt_data = []
        for i, line in enumerate(lines):
            guid = str(i)
            text = line[1]
            # word_piece = self.word_piece_bool(text)
            # if word_piece:
            #     continue
            label = line[0]
            tdt_data.append(InputData(guid=guid, text=text, label=label))

        return tdt_data


    def word_piece_bool(self, text):
        word_piece = False
        data_text_list = text.split(' ')
        for i, word in enumerate(data_text_list):
            # 防止wordPiece情况出现，不过貌似不会
            token = self.tokenizer.tokenize(word)
            # 单个字符表示不会出现wordPiece
            if len(token) != 1:
                word_piece = True

        return word_piece
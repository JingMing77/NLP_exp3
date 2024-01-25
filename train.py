# pytorch code for sequence tagging
import numpy as np
# 此版本为简单的NER代码，没有使用CRF和训练好的词向量，仅做参考使用。

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertConfig

from DataLoader import NERDataset
from utils import build_vocab, build_dict, cal_max_length, Config, Metrics
from model import NERLSTM, BiLSTM_CRF, BERT_BiLSTM_CRF
from torch.optim import Adam, SGD
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NERdataset(Dataset):

    def __init__(self, data_dir, split, word2id, tag2id, max_length):
        file_dir = data_dir + split
        corpus_file = file_dir + '_corpus.txt'
        label_file = file_dir + '_label.txt'
        corpus = open(corpus_file, encoding='utf-8').readlines()
        label = open(label_file, encoding='utf-8').readlines()
        self.corpus = []
        self.label = []
        self.length = []
        self.word2id = word2id
        self.tag2id = tag2id
        for corpus_, label_ in zip(corpus, label):
            assert len(corpus_.split()) == len(label_.split())
            self.corpus.append([word2id[temp_word] if temp_word in word2id else word2id['unk']
                                for temp_word in corpus_.split()])
            self.label.append([tag2id[temp_label] for temp_label in label_.split()])
            self.length.append(len(corpus_.split()))
            if (len(self.corpus[-1]) > max_length):
                # 字数超出 max_length
                self.corpus[-1] = self.corpus[-1][:max_length]
                self.label[-1] = self.label[-1][:max_length]
                self.length[-1] = max_length
            else:
                # 字数补全至 max_length
                while (len(self.corpus[-1]) < max_length):
                    self.corpus[-1].append(word2id['pad'])
                    self.label[-1].append(tag2id['PAD'])

        self.corpus = torch.Tensor(self.corpus).long()
        self.label = torch.Tensor(self.label).long()
        self.length = torch.Tensor(self.length).long()

    def __getitem__(self, item):
        return self.corpus[item], self.label[item], self.length[item]

    def __len__(self):
        return len(self.label)


def val(config, model):
    # ignore the pad label
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=7)
    testset = NERdataset(config.data_dir, 'test', word2id, tag2id, max_length)
    dataloader = DataLoader(testset, batch_size=config.batch_size)
    preds, labels = [], []
    for index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        corpus, label, length = data
        corpus, label, length = corpus.to(device), label.to(device), length.to(device)
        if config.model == 'LSTM':
            output = model(corpus)
            predict = torch.argmax(output, dim=-1)
            loss = loss_function(output.view(-1, output.size(-1)), label.view(-1))
        else:
            score, output = model(corpus)
            predict = torch.tensor(output)

        leng = []
        for i in label.cpu():
            tmp = []
            for j in i:
                if j.item() < 7:
                    tmp.append(j.item())
            leng.append(tmp)

        for ind, i in enumerate(predict.tolist()):
            preds.extend(i[:len(leng[ind])])

        for ind, i in enumerate(label.tolist()):
            labels.extend(i[:len(leng[ind])])

    preds = np.array(preds)
    labels = np.array(labels)
    # preds[preds > 6] = labels[preds > 6]
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds)
    print()
    print(report)
    model.train()
    return precision, recall, f1


def train(config, model, dataloader, optimizer):
    # ignore the pad label
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=7)
    best_f1 = 0.0
    # prec, rec, f1 = val(config, model)
    for epoch in range(config.epoch):
        for index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            optimizer.zero_grad()

            corpus, label, length = data
            corpus, label, length = corpus.to(device), label.to(device), length.to(device)
            output = model(corpus)
            loss = loss_function(output.view(-1, output.size(-1)), label.view(-1))

            loss.backward()
            optimizer.step()
            if index % 200 == 0:
                print()
                print('epoch: ', epoch, ' step:%04d,------------loss:%f' % (index, loss.item()))

        prec, rec, f1 = val(config, model)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model, config.save_model)


def train_CRF(config, model, dataloader, optimizer, load_ckpt=False, BiLSTM_model_path=None):
    start = 0
    if load_ckpt:
        model = torch.load(config.save_model)
        # start = model['global_step']
        # optimizer.load_state_dict(model['optimizer_state_dict'])
    BiLSTM = torch.load(BiLSTM_model_path) if BiLSTM_model_path is not None else None

    best_f1 = 0.0
    # prec, rec, f1 = val(config, model)

    for epoch in range(start, config.epoch):
        for index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            model.zero_grad()
            corpus, label, length = data
            corpus, label, length = corpus.to(device), label.to(device), length.to(device)

            feats = None
            if BiLSTM_model_path is not None:
                feats0 = BiLSTM(corpus)  # [bs, max_len, target_size-2]
                zeros2 = torch.zeros((feats0.size(0), feats0.size(1), 2), dtype=feats0.dtype).to(device)
                feats = torch.cat((feats0, zeros2), dim=-1)
            loss = model.neg_log_likelihood(corpus, label, feats)

            loss.backward()
            optimizer.step()
            if index % 60 == 0:
                print()
                print('epoch: ', epoch, ' step:%04d,------------loss:%f' % (index, loss.item()))

        if epoch == config.epoch - 1:
            prec, rec, f1 = val(config, model)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model, config.save_model)


def evaluate(config, tokenizer, dataset, model, id2label, device, tqdm_desc):
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    id2label[-1] = 'NULL'  # 解码临时添加
    ori_tokens = [tokenizer.decode(tdt['input_ids']).split(" ") for tdt in dataset]
    ori_labels = [[id2label[idx] for idx in tdt['label_ids']] for tdt in dataset]
    pred_labels = []

    for b_i, batch_data in enumerate(tqdm(data_loader, desc=tqdm_desc)):
        batch_data = tuple(torch.stack(batch_data[k]).T.to(device) for k in batch_data.keys())
        input_ids, token_type_ids, attention_mask, label_ids = batch_data

        with torch.no_grad():
            logits = model.predict(input_ids, token_type_ids, attention_mask)

        for logit in logits:
            pred_labels.append([id2label[idx] for idx in logit])

    assert len(pred_labels) == len(ori_tokens) == len(ori_labels)
    eval_sens = []
    for ori_token, ori_label, pred_label in zip(ori_tokens, ori_labels, pred_labels):
        sen_tll = []
        for ot, ol, pl in zip(ori_token, ori_label, pred_label):
            if ot in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            sen_tll.append((ot, ol, pl))
        eval_sens.append(sen_tll)

    golden_tags = [[ttl[1] for ttl in sen] for sen in eval_sens]
    predict_tags = [[ttl[2] for ttl in sen] for sen in eval_sens]
    cal_indicators = Metrics(golden_tags, predict_tags, remove_O=config.remove_O)
    avg_metrics = cal_indicators.cal_avg_metrics()  # avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1_score']

    return avg_metrics, cal_indicators, eval_sens

def train_BERT_LSTM_CRF(config, model, dataloader, optimizer, scheduler, tokenizer, do_eval=True):
    model.train()
    label2id = {label: i for i, label in enumerate(config.label_list)}
    id2label = {i: label for label, i in label2id.items()}
    global_step, tr_loss, logging_loss, best_f1 = 0, 0.0, 0.0, 0.0
    writer = SummaryWriter(log_dir=os.path.join(config.output_path, "visual"), comment="ner")

    for epoch in range(int(config.epoch)):
        for batch, batch_data in enumerate(tqdm(dataloader, desc="Train_DataLoader")):

            batch_data = tuple(torch.stack(batch_data[k]).T.to(device) for k in batch_data.keys())
            input_ids, token_type_ids, attention_mask, label_ids = batch_data
            # outputs = model(input_ids, label_ids, token_type_ids, attention_mask)

            loss = model.neg_log_likelihood(input_ids, label_ids, token_type_ids, attention_mask)

            loss.backward()
            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()

            model.zero_grad()
            global_step += 1

            if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                tr_loss_avg = (tr_loss - logging_loss) / config.logging_steps
                writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                logging_loss = tr_loss
                print("loss: ", loss)

        if do_eval:
            eval_data = NERDataset(config, tokenizer, mode="eval")
            avg_metrics, cal_indicators, eval_sens = evaluate(
                config, tokenizer, eval_data, model, id2label, device, tqdm_desc="Eval_DataLoader")
            f1_score = avg_metrics['f1_score']
            writer.add_scalar("Eval/precision", avg_metrics['precision'], epoch)
            writer.add_scalar("Eval/recall", avg_metrics['recall'], epoch)
            writer.add_scalar("Eval/f1_score", avg_metrics['f1_score'], epoch)

            # save the best performs model
            if f1_score > best_f1:
                best_f1 = f1_score
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(config.trained_model_path)
                tokenizer.save_pretrained(config.trained_model_path)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(os.path.join(config.trained_model_path, 'checkpoints'))
                tokenizer.save_pretrained(os.path.join(config.trained_model_path, 'checkpoints'))

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(os.path.join(config.trained_model_path, 'checkpoints'))
    tokenizer.save_pretrained(os.path.join(config.trained_model_path, 'checkpoints'))


if __name__ == '__main__':
    config = Config()
    word_dict = build_vocab(config.data_dir)
    word2id, tag2id = build_dict(word_dict)
    max_length = cal_max_length(config.data_dir)
    config.max_seq_length = max_length

    if config.model == 'LSTM':
        trainset = NERdataset(config.data_dir, 'train', word2id, tag2id, max_length)
        dataloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
        nerlstm = NERLSTM(config.embedding_dim, config.hidden_dim, config.dropout, len(word2id), tag2id).to(device)
        optimizer = Adam(nerlstm.parameters(), config.learning_rate)
        train(config, nerlstm, dataloader, optimizer)
        exit()

    if config.model == 'bert_bilstm_crf':
        # trainset = NERdataset(config.data_dir, 'train', word2id, tag2id, max_length)
        # dataloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
        tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path, do_lower_case=True)
        dataset = NERDataset(config, tokenizer, mode='train')
        dataloader = DataLoader(dataset, config.batch_size, shuffle=True)

        bert_config = BertConfig.from_pretrained(config.model_name_or_path, num_labels=len(tag2id))

        model = BiLSTM_CRF.from_pretrained(config.model_name_or_path, config=bert_config,
                                           vocab_size=len(word2id), tag_to_ix=tag2id,
                                           embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
                                           dropout=config.dropout).to(device)

        print("loading tokenizer、bert_config and bert_bilstm_crf model successful!")
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-5)
        t_total = len(dataloader) // config.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                    num_training_steps=t_total)
        train_BERT_LSTM_CRF(config, model, dataloader, optimizer, scheduler, tokenizer)

        # optimizer = SGD(nercrf.parameters(), lr=0.001, weight_decay=1e-4)
        # # optimizer = Adam(nercrf.parameters(), lr=0.1)
        # train_CRF(config, nercrf, dataloader, optimizer, load_ckpt=False)
        exit()

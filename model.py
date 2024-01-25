#-*-coding:GBK -*-
import torch
import torch.nn as nn
import os
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NERLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, vocab_size, tag2id):
        super(NERLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag2id
        self.tagset_size = len(tag2id)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True,
                            batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

    def forward(self, x):  # x: [bs, max_len]
        embedding = self.word_embeds(x)  # [bs, max_len_sent, embedding_dim]
        outputs, hidden = self.lstm(embedding)  # [bs, max_len, hidden_dim]
        outputs = self.dropout(outputs)  # [bs, max_len, hidden_dim]
        outputs = self.hidden2tag(outputs)  # [bs, max_len, len_target]
        return outputs


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(mat):
    max_score = torch.max(mat, dim=-1, keepdim=True)[0]
    return max_score + torch.log(torch.sum(
        torch.exp(torch.sub(mat, max_score)), dim=-1, keepdim=True))



class BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout,
                 START_TAG="<START>", STOP_TAG="<STOP>", PAD_TAG='PAD'):
        super(BiLSTM_CRF, self).__init__(config)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.START_TAG = START_TAG
        self.STOP_TAG = STOP_TAG

        self.tag_to_ix[START_TAG] = len(self.tag_to_ix) if START_TAG not in self.tag_to_ix else self.tag_to_ix[START_TAG]
        self.tag_to_ix[STOP_TAG] = len(self.tag_to_ix) if STOP_TAG not in self.tag_to_ix else self.tag_to_ix[STOP_TAG]
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.bert = BertModel(config)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that
        # we never transfer to the start tag and we never transfer from the stop tag
        # self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        # self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000
        # self.transitions.data[:, self.tag_to_ix[PAD_TAG]] = -10000
        # self.transitions.data[self.tag_to_ix[PAD_TAG], self.tag_to_ix[PAD_TAG]] = torch.randn(1)
        # self.transitions.data[self.tag_to_ix[STOP_TAG], self.tag_to_ix[PAD_TAG]] = torch.randn(1)
        self.transitions.data[self.tag_to_ix['I-PER'], :] = -10000
        self.transitions.data[self.tag_to_ix['I-PER'], self.tag_to_ix['B-PER']] = torch.randn(1)
        self.transitions.data[self.tag_to_ix['I-PER'], self.tag_to_ix['I-PER']] = torch.randn(1)
        self.transitions.data[self.tag_to_ix['I-ORG'], :] = -10000
        self.transitions.data[self.tag_to_ix['I-ORG'], self.tag_to_ix['B-ORG']] = torch.randn(1)
        self.transitions.data[self.tag_to_ix['I-ORG'], self.tag_to_ix['I-ORG']] = torch.randn(1)
        self.transitions.data[self.tag_to_ix['I-LOC'], :] = -10000
        self.transitions.data[self.tag_to_ix['I-LOC'], self.tag_to_ix['B-LOC']] = torch.randn(1)
        self.transitions.data[self.tag_to_ix['I-LOC'], self.tag_to_ix['I-LOC']] = torch.randn(1)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -10000.).to(device)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            # t_r1_k = feats[:,feat_index,:].repeat(feats.shape[0],1,1).transpose(1, 2)
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[self.STOP_TAG]].repeat(
            [feats.shape[0], 1])
        # terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _get_lstm_features(self, sentence, token_type_ids=None, attention_mask=None):
        # embedding = self.word_embeds(sentence)  # [bs, max_len_sent, embedding_dim]
        outputs = self.bert(sentence, token_type_ids=token_type_ids, attention_mask=attention_mask)
        embedding = outputs[0]  # torch.Size([batch_size,seq_len,hidden_size])

        outputs, hidden = self.lstm(embedding)  # [bs, max_len, hidden_dim]
        outputs = self.dropout(outputs)  # [bs, max_len, hidden_dim]
        outputs = self.hidden2tag(outputs)  # [bs, max_len, len_target]
        return outputs

    def _score_sentence(self, feats, tags):
        # Gives the score of provided tag sequences
        # feats = feats.transpose(0,1)

        score = torch.zeros(tags.shape[0]).to(device)
        tags = torch.cat(
            [torch.full([tags.shape[0], 1], self.tag_to_ix[self.START_TAG], dtype=torch.long).to(device), tags], dim=1)
        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            score = score + \
                    self.transitions[tags[:, i + 1], tags[:, i]] + feat[range(feat.shape[0]), tags[:, i + 1]]
            # if self.transitions[tags[:, i + 1], tags[:, i]].sum() < -5000:
            #     print(tags[:, i + 1], ',', tags[:, i])
            #     print(self.transitions[tags[:, i + 1], tags[:, i]])
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[:, -1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            add_up = forward_var + self.transitions
            viterbivars_t, t = torch.max(add_up, dim=-1)
            bptrs_t = t.tolist()
            forward_var = (viterbivars_t + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, token_type_ids=None, attention_mask=None, feats=None):
        if feats is None:
            feats = self._get_lstm_features(sentence, token_type_ids, attention_mask)  # 1 sent: [n, 5]  ->  bs sents: [bs, n, 5]
        # feats = feats.view(-1, self.tagset_size)        # [bs * n, 5]
        # tags = tags.view(tags.size(0) * tags.size(1))   # [bs * n, ]
        # forward_score = 0
        # gold_score = 0
        # for feat, tag in zip(feats, tags):
        #     forward_score += self._forward_alg(feat)
        #     gold_score += self._score_sentence(feat, tag)
        # return forward_score - gold_score
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.mean(forward_score - gold_score)

    def forward(self, sentence, token_type_ids=None, attention_mask=None, lstm_feats=None):  # dont confuse this with _forward_alg above.
        if lstm_feats is None:
            # Get the emission scores from the BiLSTM
            lstm_feats = self._get_lstm_features(sentence, token_type_ids, attention_mask)
        # lstm_feats = lstm_feats.view(-1, self.tagset_size)
        score = torch.zeros(lstm_feats.size(0)).to(device)
        tag_seq = []
        for idx, feat in enumerate(lstm_feats):
            # Find the best path, given the features.
            score_cur, tag_seq_cur = self._viterbi_decode(feat)
            score[idx] = score_cur
            tag_seq.append(tag_seq_cur)

        return score, tag_seq


class BERT_BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config, need_birnn=True, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.need_birnn = need_birnn
        self.birnn = nn.LSTM(input_size=config.hidden_size, hidden_size=rnn_dim//2, num_layers=1, bidirectional=True,
                             batch_first=True)

        self.hidden2tag = nn.Linear(in_features=rnn_dim, out_features=config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, tags, token_type_ids=None, attention_mask=None):
        """
        :param input_ids:      torch.Size([batch_size,seq_len]), 代表输入实例的tensor张量
        :param token_type_ids: torch.Size([batch_size,seq_len]), 一个实例可以含有两个句子,相当于标记
        :param attention_mask:     torch.Size([batch_size,seq_len]), 指定对哪些词进行self-Attention操作
        :param tags:
        :return:
        """
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # torch.Size([batch_size,seq_len,hidden_size])

        sequence_output, _ = self.birnn(sequence_output)  # (seq_length,batch_size,num_directions*hidden_size)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)  # [seq_length, batch_size, num_labels]
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return self.crf.decode(emissions, attention_mask.byte())

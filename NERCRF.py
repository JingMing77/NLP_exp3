import time
import torch
import torch.nn as nn
import torch.optim as optim

from model import BiLSTM_CRF

START_TAG = "<START>"
STOP_TAG = "<STOP>"


# torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def prepare_sequence_batch(data, word_to_ix, tag_to_ix):
    seqs = [i[0] for i in data]
    tags = [i[1] for i in data]
    max_len = max([len(seq) for seq in seqs])
    seqs_pad = []
    tags_pad = []
    for seq, tag in zip(seqs, tags):
        seq_pad = seq + ['<PAD>'] * (max_len - len(seq))
        tag_pad = tag + ['<PAD>'] * (max_len - len(tag))
        seqs_pad.append(seq_pad)
        tags_pad.append(tag_pad)
    idxs_pad = torch.tensor([[word_to_ix[w] for w in seq] for seq in seqs_pad], dtype=torch.long)
    tags_pad = torch.tensor([[tag_to_ix[t] for t in tag] for tag in tags_pad], dtype=torch.long)
    return idxs_pad, tags_pad

if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    PAD_TAG = "<PAD>"
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {}
    word_to_ix['<PAD>'] = 0
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4, PAD_TAG: 5}

    # model = BiLSTM_CRF_MODIFY_PARALLEL(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, dropout=0.4,
                       START_TAG=START_TAG, STOP_TAG=STOP_TAG, PAD_TAG=PAD_TAG).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        # precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        # precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        precheck_sent, targets_pad = prepare_sequence_batch(training_data, word_to_ix, tag_to_ix)
        precheck_sent = precheck_sent.to(device)
        targets_pad = targets_pad.to(device)
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            300):  # again, normally you would NOT do 300 epochs, it is toy data
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # Step 2. Get our batch inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        # If training_data can't be included in one batch, you need to sample them to build a batch
        sentence_in_pad, targets_pad = prepare_sequence_batch(training_data, word_to_ix, tag_to_ix)
        sentence_in_pad = sentence_in_pad.to(device)
        targets_pad = targets_pad.to(device)
        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in_pad, targets_pad)
        if epoch % 10 == 0:
            print(loss)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        # precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        # precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        precheck_sent, targets_pad = prepare_sequence_batch(training_data, word_to_ix, tag_to_ix)
        precheck_sent = precheck_sent.to(device)
        targets_pad = targets_pad.to(device)
        print(model(precheck_sent))
        # We got it!

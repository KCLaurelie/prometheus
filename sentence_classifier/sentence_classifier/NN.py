# -*- coding: utf-8 -*-
from collections import Counter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from sentence_classifier.sentence_classifier.NLP_utils import *

"""# FUNCTIONS TO BUILD NN MODELS"""


class RNN(nn.Module):
    def __init__(self, embeddings, padding_idx=0, hidden_size=300, num_layers=2, dropout=0.5, bid=True,
                 rnn_type='default', debug_mode=False):
        super(RNN, self).__init__()
        self.debug_mode = debug_mode
        self.dropout = dropout
        # Initialize embeddings
        vocab_size = len(embeddings)
        embedding_size = len(embeddings[0])
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)  # Create embeddings
        self.embeddings.load_state_dict({'weight': embeddings})  # load existing weights
        self.embeddings.weight.requires_grad = False  # Disable training for the embeddings - IMPORTANT

        # Create the RNN cell
        if 'gru' in rnn_type.lower():
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              dropout=dropout)
        elif 'lstm' in rnn_type.lower():
            self.rnn = nn.LSTM(input_size=embedding_size, num_layers=num_layers, dropout=dropout, bidirectional=bid,
                               hidden_size=hidden_size // (2 if bid else 1))
        else:
            self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 2)
        if dropout is not None: self.d1 = nn.Dropout(dropout)

    def forward(self, x, lns, mask):
        if self.debug_mode: print('start forward function - x: ', x.shape)
        x = self.embeddings(x)  # x.shape = batch_size x sequence_length x emb_size
        if self.debug_mode: print('after embedding - x: ', x.shape)
        # Tell RNN to ignore padding and set the batch_first to True (sequence length 1st for historical reasons)
        lengths = to_cpu(mask.sum(1).int())
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, hidden = self.rnn(x)  # run 'x' through the RNN
        x, hidden = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)  # Add the padding again
        # select the value at the length of that sentence (we are only interested in last output) or middle if bidirectional
        row_indices = torch.arange(0, x.size(0)).long()
        x = x[row_indices, lns - 1, :]
        if self.debug_mode: print('before d1/fc1: ', x.shape)
        if self.dropout is not None: x = self.d1(x)
        x = self.fc1(x)
        return x


class ANN(nn.Module):
    def __init__(self, embeddings, padding_idx=0, final_layer_neurons=1, dropout=0.5, debug_mode=False):
        super(ANN, self).__init__()
        self.debug_mode = debug_mode
        self.dropout = dropout
        # Initialize embeddings
        vocab_size = embeddings.shape[0]
        embedding_size = embeddings.shape[1]
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.embeddings.load_state_dict({'weight': embeddings})
        self.embeddings.weight.requires_grad = False  # disable training for the embeddings
        first_layer_neurons = embedding_size

        self.fc1 = nn.Linear(first_layer_neurons, 100)
        self.fc2 = nn.Linear(100, final_layer_neurons)  # 3 categories/classes = 3 final neurons
        if dropout is not None: self.d1 = nn.Dropout(dropout)

    def forward(self, x, lns=0, mask=None):
        if self.debug_mode: print('start forward function - x: ', x.shape)
        x = self.embeddings(x)
        if self.debug_mode: print('after embedding - x: ', x.shape)
        # this does not work to check
        x = torch.mean(x, dim=1)  # to sum over all columns
        if self.debug_mode: print('x after average', x.size())
        x = torch.relu(self.fc1(x))  # torch.sigmoid(self.fc1(x))
        if self.dropout is not None: x = self.d1(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class CNN(nn.Module):
    def __init__(self, embeddings, dropout=0.5, debug_mode=False):
        super(CNN, self).__init__()
        self.debug_mode = debug_mode
        self.dropout = dropout
        # Initialize embeddings
        vocab_size = embeddings.shape[0]
        embedding_size = embeddings.shape[1]
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embeddings.load_state_dict({'weight': embeddings})
        self.embeddings.weight.requires_grad = False  # disable training for the embeddings
        n_filters = 128  # Set the number of filters
        #  3 different kernel sizes (to have patterns of 3, 4 and 2 words)
        k1 = (3, embedding_size)
        k2 = (4, embedding_size)
        k3 = (2, embedding_size)
        # convolutional layers
        self.conv1 = nn.Conv2d(1, n_filters, k1)  # nb of channels always 1 in text
        self.conv2 = nn.Conv2d(1, n_filters, k2)
        self.conv3 = nn.Conv2d(1, n_filters, k3)
        # fully connected network: concatenate the 3 conv layers and put through linear layer (size = 3*n_filters)
        self.fc1 = nn.Linear(3 * n_filters, 2)
        if dropout is not None: self.d1 = nn.Dropout(dropout)

    def conv_block(self, input, conv):
        out = conv(input)  # conv function
        out = F.relu(out.squeeze(3))  # activation function
        out = F.max_pool1d(out, out.size()[2]).squeeze(2)  # max pooling
        return out

    def forward(self, x, lns=0, mask=None):
        x = self.embeddings(x)  # x.shape = batch_size x sequence_length x emb_size
        x = x.unsqueeze(1)  # Because the expected shape = batch_size x channels x sequence_length x emb_size
        x1 = self.conv_block(x, self.conv1)  # Get the output from conv layer 1
        x2 = self.conv_block(x, self.conv2)  # Get the output from conv layer 2
        x3 = self.conv_block(x, self.conv3)  # Get the output from conv layer 3
        x_all = torch.cat((x1, x2, x3), 1)  # concatenate 3 outputs for the 3 conv blocks
        if self.dropout is not None: x_all = self.d1(x_all)  # dropout
        logits = self.fc1(x_all)  # run through fc1
        return logits


"""# FUNCTIONS TO PREP DATASET / TRAIN THE MODEL"""


def prep_NN_dataset(emb_model, sentences, labels=None, tokenization_type='clean'):
    # The dataset is prepared in the following way
    # texts tokenized
    # embedding model loaded and converted to format: word ID <-> embedding vector, word <-> word ID (to save time and memory)
    # tokenized texts converted to word ID
    # texts padded to ensure all have same lengths
    sentences = pd.Series(sentences)
    if labels is not None:
        labels = convert_to_cat(labels, binary=False)['labels']
    else:
        labels = pd.Series([1] * len(sentences))
    print('tokenizing input text')
    tokenized_text = tokenize_spacy(sentences, text_col=None, tokenization_type=tokenization_type)
    print('loading embedding model data')
    # check if embedding was already converted to be usable in torch, otherwise convert to correct format
    emb_model_torch = embedding2torch(emb_model) if type(emb_model) is not dict else emb_model
    word2id = emb_model_torch['word2id']
    embeddings = emb_model_torch['embeddings']
    input_ids = []
    attention_masks = []
    lens = []
    print('converting sentences to tokens and creating attention mask')
    for snt in sentences:
        ind_snt = [word2id[tkn] if tkn in word2id else word2id['<UNK>'] for tkn in snt]
        mask_snt = [1] * len(snt)
        len_snt = len(snt)
        input_ids.append(ind_snt)
        attention_masks.append(mask_snt)
        lens.append(len_snt)
    print('padding')
    # pad input sentences and attention mask
    input_ids_padded = pad_sequence([torch.tensor(i) for i in input_ids]).transpose(0, 1)
    attention_masks_padded = pad_sequence([torch.tensor(i) for i in attention_masks]).transpose(0, 1)
    print('creating dataset')
    # create dataset
    dataset = TensorDataset(input_ids_padded, torch.tensor(labels), torch.tensor(attention_masks_padded),
                            torch.tensor(lens))
    num_labels = labels.nunique()
    return {'embeddings': embeddings, 'word2id': word2id, 'dataset': dataset, 'num_labels': num_labels}


def run_NN(nn_model, train_dataloader, validation_dataloader, output_dir=None, n_epochs=50):
    device = get_device()
    try:
        nn_model.cuda()
    except:
        device = None
        print('using CPU, this will be slow!')
    parameters = filter(lambda p: p.requires_grad,
                        nn_model.parameters())  # We don't want parameters that don't require a grad in the optimizer
    optimizer = optim.Adam(parameters, lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    best_f1, best_epoch = [0, 0]

    for epoch in range(n_epochs):
        ###################################################################################
        ## TRAINING
        nn_model.train()
        # Tracking variables
        tr_loss, tr_perf, tr_perf_classes = 0, Counter({}), pd.DataFrame()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # Unpack the inputs from our dataloader
            batch = batch_to_gpu(batch, device)
            x_train, y_train, mask_train, l_train = batch
            train_preds = nn_model(x_train, l_train, mask_train)
            # train_preds_flat = torch.max(train_preds, 1)[1]
            loss = criterion(train_preds, y_train)
            loss.backward()
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            tmp_tr_perf = perf_metrics(to_cpu(train_preds), to_cpu(y_train), average='weighted')
            tr_perf = tr_perf + Counter(tmp_tr_perf)
            tmp_tr_perf_classes = perf_metrics_classes(to_cpu(train_preds), to_cpu(y_train))
            tr_perf_classes = pd.concat((tr_perf_classes, tmp_tr_perf_classes))
            # print('step:',step, 'perf:', tmp_tr_perf, 'perf tot:', tr_perf)

        tr_perf = {k: v / (1 + step) for k, v in tr_perf.items()}
        tr_perf_classes = tr_perf_classes.replace(0, np.NaN).groupby(tr_perf_classes.index).agg(
            {'f1-score': 'mean', 'precision': 'mean', 'recall': 'mean', 'support': 'sum'})
        # tr_perf_classes = tr_perf_classes.groupby(tr_perf_classes.index).mean()
        # tr_perf_classes['support'] = np.round((1+step) * tr_perf_classes['support'])
        print("TRAIN - Loss: {:.3f} - F1: {:.3f} Acc: {:.3f} P: {:.3f} R: {:.3f}".format(tr_loss / (1 + step),
                                                                                         tr_perf['f1'], tr_perf['acc'],
                                                                                         tr_perf['p'], tr_perf['r']))

        if (epoch % 5 == 0) or (epoch >= n_epochs - 1):
            ###################################################################################
            ## TESTING
            nn_model.eval()
            # Tracking variables
            eval_perf, eval_perf_classes = Counter({}), pd.DataFrame()
            for step, batch in enumerate(validation_dataloader):
                # Unpack the inputs from our dataloader
                batch = batch_to_gpu(batch, device)
                x_test, y_test, mask_test, l_test = batch
                test_preds = nn_model(x_test, l_test, mask_test)
                # test_preds_flat = torch.max(test_preds, 1)[1]
                # Update tracking variables
                tmp_eval_perf = perf_metrics(to_cpu(test_preds), to_cpu(y_test), average='weighted')
                eval_perf = eval_perf + Counter(tmp_eval_perf)
                tmp_eval_perf_classes = perf_metrics_classes(to_cpu(test_preds), to_cpu(y_test))
                eval_perf_classes = pd.concat((eval_perf_classes, tmp_eval_perf_classes))

            eval_perf = {k: v / (1 + step) for k, v in eval_perf.items()}
            eval_perf_classes = eval_perf_classes.replace(0, np.NaN).groupby(eval_perf_classes.index).agg(
                {'f1-score': 'mean', 'precision': 'mean', 'recall': 'mean', 'support': 'sum'})
            # eval_perf_classes = eval_perf_classes.groupby(eval_perf_classes.index).mean()
            # eval_perf_classes['support'] = np.round((1+step) * eval_perf_classes['support'])
            print("TEST -- F1: {:.3f} Acc: {:.3f} P: {:.3f} R: {:.3f}".format(eval_perf['f1'], eval_perf['acc'],
                                                                              eval_perf['p'], eval_perf['r']))

            # store perf metrics and model
            if eval_perf['f1'] >= best_f1:
                best_f1 = eval_perf['f1']
                best_epoch = epoch + 1
                stats_to_save = eval_perf
                tr_perf_classes['dataset'] = 'train'
                eval_perf_classes['dataset'] = 'test'
                stats_classes_to_save = pd.concat([tr_perf_classes, eval_perf_classes])
                model_to_save = nn_model
            print('best F1 score obtained: {:.3f} at epoch {}'.format(best_f1, best_epoch))

    print('Finished Training, best F1 obtained on test set:', best_f1, 'at epoch', best_epoch)
    # save model with best f1
    if output_dir is not None:
        try:
            print('saving model in:', output_dir)
            torch.save(model_to_save, output_dir)
            stats_classes_to_save.to_csv(output_dir + '_stats.csv', header=True)
        except:
            print('model not saved, please enter valid path')

    return {'stats': stats_to_save, 'stats_classes': stats_classes_to_save, 'model': model_to_save}


"""# Functions to train/evaluate model"""


def train_NN(sentences, labels, nn_model, emb_model, tokenization_type='clean', SEED=0, test_size=0.2, n_epochs=20,
             output_dir=None):
    emb_model_torch = embedding2torch(emb_model, SEED=SEED) if type(emb_model) is not dict else emb_model
    res = prep_NN_dataset(emb_model=emb_model_torch, sentences=sentences, labels=labels,
                          tokenization_type=tokenization_type)
    dataset = res['dataset']

    print('splitting in train/test sets')
    test_len = int(len(dataset) * test_size)
    train_len = len(dataset) - test_len
    print('test set:', test_len, 'train set:', train_len)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_dataloader, validation_dataloader = create_dataloader(train_dataset, val_dataset, batch_size=32)
    res = run_NN(nn_model, train_dataloader, validation_dataloader, n_epochs=n_epochs, output_dir=output_dir)

    return res


def NN_KFOLD(sentences, labels, nn_model, emb_model, tokenization_type='clean',
             n_splits=10, random_state=42, n_epochs=5, output_dir=None):
    emb_model_torch = embedding2torch(emb_model) if type(emb_model) is not dict else emb_model
    prep_data = prep_NN_dataset(emb_model=emb_model_torch, sentences=sentences, labels=labels,
                                tokenization_type=tokenization_type)
    dataset = prep_data['dataset']

    res = run_KFOLD(dataset=dataset, base_model=nn_model, model_trainer=run_NN,
                    n_splits=n_splits, random_state=random_state, n_epochs=n_epochs)
    if output_dir is not None:
        try:
            print('saving model in:', output_dir)
            torch.save(res['model'], output_dir)
            res['stats_classes'].to_csv(output_dir + '_stats.csv', header=True)
        except:
            print('model not saved, please enter valid path')
    return res


"""# Function to load saved model and classify new data"""


# load pre-trained model and classify a new sentence
def load_and_run_NN(sentences, trained_nn_model, emb_model, tokenization_type='clean'):
    # if user passed input model as file path
    if isinstance(trained_nn_model, str):
        print('loading model from disk...', trained_nn_model)
        trained_nn_model = torch.load(trained_nn_model)

    trained_nn_model.eval()
    model_to_cpu(trained_nn_model)

    print('formating dataset')
    res = prep_NN_dataset(emb_model=emb_model, sentences=sentences, labels=None, tokenization_type=tokenization_type)
    sentences_dataset = res['dataset']
    dataloader = DataLoader(sentences_dataset)
    x, y, mask, l = sentences_dataset.tensors
    preds = trained_nn_model(x, l, mask)

    # put results in nice format
    res = pd.DataFrame()
    res['preds'] = np.argmax(preds.detach().numpy(), axis=1).flatten()
    res['sentences'] = sentences
    return res

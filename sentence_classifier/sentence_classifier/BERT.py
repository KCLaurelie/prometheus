from collections import Counter

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
from tqdm import trange
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from sentence_classifier.sentence_classifier.NLP_utils import *

"""# FUNCTIONS TO PREP DATASET / TRAIN BERT"""


def prep_BERT_dataset(sentences, labels=None, BERT_tokenizer='bert-base-uncased', MAX_TKN_LEN=511, debug=False):
    """
    prepares data for BERT fine-tuning
    :param sentences: list or array-like. list of texts to classify
    :param labels: list or array-like, default None. list of labels corresponding to sentences
    :param BERT_tokenizer: string, default 'bert-base-uncased'. BERT base model used
    :param MAX_TKN_LEN: integer, default 511. see https://github.com/huggingface/transformers/issues/2446
    :param debug: bool, default False. set to True to display inetrmediate results
    :return: prepared dataset (torch Dataset) and nmber of dictinct labels
    """
    sentences = pd.Series(sentences)
    # load relevant data and add special tokens for BERT to work properly
    sentences = ["[CLS] " + query + " [SEP]" for query in sentences]
    if labels is not None:
        labels = convert_to_cat(labels, binary=False)['labels']
    else:
        labels = pd.Series([1] * len(sentences))
    if debug: print(sentences[0])

    # Tokenize with BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_tokenizer, do_lower_case=True)
    if MAX_TKN_LEN is not None:
        print('cutting the length of tokens to', MAX_TKN_LEN)
        tokenized_texts = [tokenizer.tokenize(sent)[0:MAX_TKN_LEN] for sent in sentences]
    else:
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]
    if debug:
        print("Tokenize the first sentence:")
        print(len(tokenized_texts), len(input_ids), len(input_ids[0]))
        print(tokenized_texts[0], input_ids[0])

    # add paddding to input_ids
    input_ids_padded = pad_sequence([torch.tensor(i) for i in input_ids]).transpose(0, 1)
    if debug: print(input_ids_padded.size(), len(input_ids_padded))

    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids_padded:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # create dataset
    dataset = TensorDataset(input_ids_padded, torch.tensor(attention_masks), torch.tensor(labels))
    num_labels = labels.nunique()
    return {'dataset': dataset, 'num_labels': num_labels}


def run_BERT(model, train_dataloader, validation_dataloader, n_epochs=5, output_dir=None):
    """
    fine-tunes Bert for text classification
    :param model: base Bert model
    :param train_dataloader: Tensor Dataset. training data
    :param validation_dataloader: Tensor Dataset. testing data
    :param n_epochs: integer, default 5. number of epochs
    :param output_dir: string, default None. directory where to save the fine-tuned model
    :return:    model: fine-tuned Bert model with highest performance over k folds
                stats: high level performance statistics
                stats_classes: detailed performance statistics
    """
    ###################################################################################
    # BERT fine-tuning parameters
    device = get_device()
    try:
        model.cuda()
    except:
        device = None
        print('using CPU, this will be slow!')
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

    # BERT training loop
    train_loss_set = []
    best_f1, best_epoch = 0, 0
    for _ in trange(n_epochs, desc="Epoch"):
        ###################################################################################
        ## TRAINING

        # Set our model to training mode
        model.train()
        # Tracking variables
        tr_loss, tr_perf, tr_perf_classes = 0, Counter({}), pd.DataFrame()
        running_len = 0
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Unpack the inputs from our dataloader (and move to GPU if using)
            batch = batch_to_gpu(batch, device)
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            tmp_tr_perf = perf_metrics(to_cpu(logits), to_cpu(b_labels), average='weighted')
            tmp_tr_perf.update((k, v * len(b_input_ids)) for k, v in tmp_tr_perf.items())
            running_len += len(b_input_ids)
            tr_perf = tr_perf + Counter(tmp_tr_perf)
            tmp_tr_perf_classes = perf_metrics_classes(to_cpu(logits), to_cpu(b_labels))
            tr_perf_classes = pd.concat((tr_perf_classes, tmp_tr_perf_classes))

        # print('classes detail \n\n', tr_perf_classes)
        tr_perf = {k: v / running_len for k, v in tr_perf.items()}
        tr_perf_classes[['f1-score', 'precision', 'recall']] = tr_perf_classes[
            ['f1-score', 'precision', 'recall']].multiply(tr_perf_classes['support'], axis="index")
        tr_perf_classes = tr_perf_classes.groupby(tr_perf_classes.index).sum()
        tr_perf_classes[['f1-score', 'precision', 'recall']] = tr_perf_classes[['f1-score', 'precision', 'recall']].div(
            tr_perf_classes['support'], axis="index")
        print('TRAIN - Loss: {:.3f} - F1: {:.3f} Acc: {:.3f} P: {:.3f} R: {:.3f}'.format(tr_loss / (1 + step),
                                                                                         tr_perf['f1'], tr_perf['acc'],
                                                                                         tr_perf['p'], tr_perf['r']))

        ###################################################################################
        ## VALIDATION

        # Put model in evaluation mode
        model.eval()
        # Tracking variables
        eval_perf, eval_perf_classes = Counter({}), pd.DataFrame()
        running_len = 0
        # Evaluate data for one epoch
        for step, batch in enumerate(validation_dataloader):
            # Unpack the inputs from our dataloader (and move to GPU if using)
            batch = batch_to_gpu(batch, device)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():  # Telling the model not to compute or store gradients, saving memory and speeding up validation
                # Forward pass, calculate logit predictions
                (loss, logits) = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            # Update tracking variables
            tmp_eval_perf = perf_metrics(to_cpu(logits), to_cpu(b_labels), average='weighted')
            tmp_eval_perf.update((k, v * len(b_input_ids)) for k, v in tmp_eval_perf.items())
            # print('STEP:', step, 'LEN', len(b_input_ids), tmp_eval_perf)
            running_len += len(b_input_ids)
            eval_perf = eval_perf + Counter(tmp_eval_perf)
            tmp_eval_perf_classes = perf_metrics_classes(to_cpu(logits), to_cpu(b_labels))
            eval_perf_classes = pd.concat((eval_perf_classes, tmp_eval_perf_classes))

        eval_perf = {k: v / running_len for k, v in
                     eval_perf.items()}  # eval_perf = {k:v/(1+step) for k,v in eval_perf.items()}
        eval_perf_classes[['f1-score', 'precision', 'recall']] = eval_perf_classes[
            ['f1-score', 'precision', 'recall']].multiply(eval_perf_classes['support'], axis="index")
        eval_perf_classes = eval_perf_classes.groupby(eval_perf_classes.index).sum()
        eval_perf_classes[['f1-score', 'precision', 'recall']] = eval_perf_classes[
            ['f1-score', 'precision', 'recall']].div(eval_perf_classes['support'], axis="index")
        print('TEST -- F1: {:.3f} Acc: {:.3f} P: {:.3f} R: {:.3f}'.format(eval_perf['f1'], eval_perf['acc'],
                                                                          eval_perf['p'], eval_perf['r']))

        # store perf metrics and model
        if eval_perf['f1'] >= best_f1:
            best_f1 = eval_perf['f1']
            best_epoch = _ + 1
            stats_to_save = eval_perf
            tr_perf_classes['dataset'] = 'train'
            eval_perf_classes['dataset'] = 'test'
            stats_classes_to_save = pd.concat([tr_perf_classes, eval_perf_classes])
            model_to_save = model.copy()
        print('best F1 score obtained: {:.3f} at epoch {}'.format(best_f1, best_epoch))

    # save model with best f1
    if output_dir is not None:
        try:
            print('saving model...')
            model_to_save.save_pretrained(output_dir)
            stats_classes_to_save.to_csv(output_dir + '/stats.csv', header=True)
        except:
            print('model not saved, please enter valid path')

    return {'stats': stats_to_save, 'stats_classes': stats_classes_to_save, 'model': model_to_save}


def train_BERT(sentences, labels, BERT_tokenizer='bert-base-uncased', test_size=0.1,
               n_epochs=5, batch_size=32, output_dir=None, MAX_TKN_LEN=511):
    """
    formats dataset and fine-tunes Bert model on it
    :param sentences: list or array-like. list of texts to classify
    :param labels: list or array-like. list of labels corresponding to sentences
    :param BERT_tokenizer: string, default 'bert-base-uncased'. BERT base model used
    :param test_size: float, default 0.10. train/test split size
    :param n_epochs: integer, default 5. number of epochs used to fine-tune Bert
    :param batch_size: integer, default 32. how many samples pper batch to load
    :param output_dir: string, default None. directory where to save the fine-tuned model
    :param MAX_TKN_LEN: integer, default 511. see https://github.com/huggingface/transformers/issues/2446
    :return:
    """
    print('formating dataset')
    prep_data = prep_BERT_dataset(sentences=sentences, labels=labels, BERT_tokenizer=BERT_tokenizer,
                                  MAX_TKN_LEN=MAX_TKN_LEN)
    dataset = prep_data['dataset']
    num_labels = prep_data['num_labels']
    # split into train/test
    print('splitting in train/test sets')
    test_len = int(len(dataset) * test_size)
    train_len = len(dataset) - test_len
    print('test set:', test_len, 'train set:', train_len)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    # Create the DataLoaders for our training and validation sets.
    train_dataloader, validation_dataloader = create_dataloader(train_dataset, val_dataset, batch_size=batch_size)
    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
    print('loading pre-trained BERT')
    pretrained_model = BertForSequenceClassification.from_pretrained(BERT_tokenizer, num_labels=num_labels)
    # train and evaluate BERT
    print('training BERT')
    res = run_BERT(pretrained_model, train_dataloader, validation_dataloader, n_epochs=n_epochs, output_dir=output_dir)
    return res


def BERT_KFOLD(sentences, labels, BERT_tokenizer='bert-base-uncased',
               n_splits=10, random_state=42, n_epochs=5, output_dir=None, MAX_TKN_LEN=511):
    """
    fine-tunes Bert model using k-old cross validation
    :param sentences: list or array-like. list of texts to classify
    :param labels: list or array-like. list of labels corresponding to sentences
    :param BERT_tokenizer: string, default 'bert-base-uncased'. BERT base model used
    :param n_splits: integer, default None. number of folds to use
    :param random_state: integer, default 42. random seed to initialize folds generation
    :param n_epochs: integer, default 5. number of epochs used to fine-tune Bert
    :param output_dir: string, default None. directory where to save the fine-tuned model
    :param MAX_TKN_LEN: integer, default 511. see https://github.com/huggingface/transformers/issues/2446
    :return:    model: fine-tuned Bert model with highest performance over k folds
                stats: high level performance statistics
                stats_classes: detailed performance statistics
    """
    prep_data = prep_BERT_dataset(sentences=sentences, labels=labels, BERT_tokenizer=BERT_tokenizer,
                                  MAX_TKN_LEN=MAX_TKN_LEN)
    dataset = prep_data['dataset']
    res = run_KFOLD(dataset=dataset, base_model=BERT_tokenizer, model_trainer=run_BERT,
                    base_model_loader=BertForSequenceClassification.from_pretrained,
                    n_splits=n_splits, random_state=random_state, n_epochs=n_epochs, num_labels=prep_data['num_labels'])
    if output_dir is not None:
        try:
            print('saving model in:', output_dir)
            res['model'].save_pretrained(output_dir)
            res['stats_classes'].to_csv(output_dir + '/stats.csv', header=True)
        except:
            print('model not saved, please enter valid path')
    return res


"""# Function to load saved model and classify new data"""


def load_BERT_components(trained_bert_model, BERT_tokenizer='bert-base-uncased', do_lower_case=True):
    """
    loads in cache model components
    :param trained_bert_model: string or pytorch model. pre-trained model name or path
    :param BERT_tokenizer: string, default 'bert-base-uncased'. BERT base model used
    :param do_lower_case: bool, default True. Whether or not to lowercase the input when tokenizing
    :return:    tokenizer: tokenizer loaded in cache
                model: pytorch model loaded in cache
    """
    tokenizer = BertTokenizer.from_pretrained(BERT_tokenizer, do_lower_case=do_lower_case)
    trained_bert_model = BertForSequenceClassification.from_pretrained(trained_bert_model)
    return {'tokenizer': tokenizer, 'model': trained_bert_model}


# load pre-trained model and classify a new sentence
def load_and_run_BERT(sentences, trained_bert_model, BERT_tokenizer='bert-base-uncased', MAX_TKN_LEN=511,
                      batch_size=32):
    """
    loads pre-trained Bert model and runs it on new text to classify
    :param sentences: list or array-like. list of texts to classify
    :param trained_bert_model: string or pytorch model. pre-trained model name or path
    :param BERT_tokenizer: string, default 'bert-base-uncased'. BERT base model used
    :param MAX_TKN_LEN: integer, default 511. see https://github.com/huggingface/transformers/issues/2446
    :param batch_size: integer, default 32. how many samples pper batch to load
    :return: dataframe of sentences classified along with probaility scores
    """
    if isinstance(trained_bert_model, str):  # load trained BERT model if needed
        trained_bert_model = BertForSequenceClassification.from_pretrained(trained_bert_model)
    preds_class, probs, preds = [], [], pd.DataFrame()
    sentences = pd.Series(sentences)
    sentences_dataset = \
    prep_BERT_dataset(sentences, labels=None, BERT_tokenizer=BERT_tokenizer, MAX_TKN_LEN=MAX_TKN_LEN)['dataset']
    # b_input_ids, b_input_mask, b_labels = sentences_dataset.tensors
    validation_dataloader = DataLoader(sentences_dataset, sampler=SequentialSampler(sentences_dataset),
                                       batch_size=batch_size)
    for step, batch in enumerate(validation_dataloader):
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            (loss, logits) = trained_bert_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                                                labels=b_labels)
        preds_curr = logits.detach().numpy()
        preds = preds.append(pd.DataFrame(preds_curr), ignore_index=True)
        probs = np.append(probs, np.max(np.exp(preds_curr) / (1 + np.exp(preds_curr)), axis=1))
        preds_class = np.append(preds_class, np.argmax(preds_curr, axis=1).flatten())

    # put results in nice format
    preds['sentences'] = sentences
    preds['preds'] = preds_class
    preds['probs'] = probs
    return preds

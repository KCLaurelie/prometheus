import numpy as np
import pandas as pd
from nltk.stem.porter import *
# for tokenization
from tokenizers import ByteLevelBPETokenizer

stemmer = PorterStemmer()
# for modelling
from gensim.models import KeyedVectors
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold
import spacy

nlp = spacy.load('en')

"""# Functions to tokenize text"""


def center_text(sentences, pos, context=15):
    sentences = pd.Series(sentences)
    pos = pd.Series(pos)
    res = []
    for i, snt in enumerate(sentences):
        snt1 = snt[0:pos[i]].split()
        snt2 = snt[pos[i]:].split()
        snt_new = snt1[-context:] + snt2[0:(1 + context)]
        res.append(snt_new)
    return res


def load_and_clean_data(filename, text_col, label_col, strip=False, MAX_LEN=None, clean_labels=False, binary=False,
                        pos_col=None, context=15):
    df = pd.read_csv(filename) if 'csv' in filename else pd.read_excel(filename)
    if strip:
        df[text_col] = df[text_col].apply(lambda x: x.strip())
    if pos_col is not None:
        df[text_col] = center_text(sentences=df[text_col], pos=df[pos_col], context=context)
    if MAX_LEN is not None:
        df[text_col] = df[text_col].apply(lambda x: x.split()[0:MAX_LEN])
    if clean_labels:
        df[label_col] = convert_to_cat(df[label_col], binary=binary)['labels']

    return df


def tokenize_spacy(df, text_col=None, tokenization_type='clean', outfile=None):
    tok_snts = []
    if outfile is not None: f = open(outfile, 'w', encoding='utf8')
    data = df if text_col is None else df[text_col]
    for snt in data:
        tkns = nlp.tokenizer(snt)
        if ('low' in tokenization_type) and ('wos' in tokenization_type):
            _tkns = [str(x.text).lower() for x in tkns if not x.is_space]
        elif 'wos' in tokenization_type:
            _tkns = [str(x.text) for x in tkns if not x.is_space]
        elif 'lem' in tokenization_type:
            _tkns = [str(x.lemma_).lower() for x in tkns if not x.is_space and not x.is_punct]
        elif 'stem' in tokenization_type:
            _tkns = [stemmer.stem(str(x.text).lower()) for x in tkns if not x.is_space and not x.is_punct]
        else:  # clean by default
            _tkns = [str(x.text).lower() for x in tkns if not x.is_space and not x.is_punct]

        if outfile is not None:  # flush to file if option selected
            f.write("{}\n".format("\t".join(_tkns)))
        else:  # otherwise save in variable
            tok_snts.append(_tkns)

    return tok_snts


def tokenize_hf(df, text_col='text', outfile=None):
    tokenizer = ByteLevelBPETokenizer(merges_file="/home/ubuntu/data/mimic/bbpe_tokenizer/mimic-merges.txt",
                                      vocab_file="/home/ubuntu/data/mimic/bbpe_tokenizer/mimic-vocab.json")
    tok_snts = []
    if outfile is not None: f = open(outfile, 'w', encoding='utf8')
    data = df if text_col is None else df[text_col]
    for snt in data:
        tokenized_snt = tokenizer.encode(snt)
        if outfile is not None:
            f.write("{}\n".format("\t".join(_tkns)))
        else:
            tok_snts.append(tokenized_snt.tokens)
    return tok_snts


def tokenize_df(df, tokenization_type, text_col='text', outfile=None):
    if 'bpe' in tokenization_type:
        res = tokenize_hf(df, text_col=text_col, outfile=outfile)
    else:
        res = tokenize_spacy(df, text_col=text_col, tokenization_type=tokenization_type, outfile=outfile)
    return res


"""# Functions to format dataset for models"""


# Create the DataLoaders for our training and validation sets.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
def create_dataloader(train_dataset, val_dataset, batch_size=32):
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    return [train_dataloader, validation_dataloader]


def convert_to_cat(labels, binary=False):
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)
    if binary:
        main_val = labels.mode()[0]
        other_val = [x for x in labels.unique() if x != main_val]
        labels = np.where(labels != main_val, 1, 0)
        corresp = pd.DataFrame([main_val, 'other'], columns=['old_label'])
    elif np.issubdtype(labels.dtype, np.number) and (labels.min() >= 0):
        print('labels already in a good format')
        return {'labels': labels, 'categories': pd.DataFrame(labels.unique())}
    else:
        cats = labels.astype('category').cat
        labels = cats.codes.astype('long')  # convert annotations to integers
        corresp = pd.DataFrame(cats.categories, columns=['old_label'])
    corresp.index.rename('new_label', inplace=True)
    print('labels have been transformed for the model:\n\n', corresp)
    return {'labels': labels, 'categories': corresp}


def to_list(x):
    """
    converts variable to a list
    :param x: variable to convert
    :return: variable converted to list
    """
    if x is None:
        res = []
    elif isinstance(x, str):
        res = [x]
    elif isinstance(x, list):
        res = x
    else:
        res = list(x)
    return res


def extract_sentences(docs, keywords, num_chars=200, num_words=None):
    res = []
    keywords = [keywords] if isinstance(keywords, str) else list(keywords)
    for doc in docs:
        doc = doc.lower()
        kw_idx = [doc.find(kw) for kw in keywords]
        kw_idx = [value for value in kw_idx if value != -1]
        if len(kw_idx) > 0:
            for i in kw_idx:
                if num_words is not None:
                    snt1 = doc[0:i].split()
                    snt2 = doc[i:].split()
                    snt = snt1[-num_words:] + snt2[0:(1 + num_words)]
                    snt = ' '.join(snt)
                else:
                    snt = doc[np.maximum(0, i - num_chars):np.maximum(0, 10 + i + num_chars)]
                res.append(snt)
    return pd.Series(res)


"""# Embedding utils"""


def embedding2torch(w2v_model, SEED=0):
    if isinstance(w2v_model, str):
        print('loading embedding model from file')
        w2v_model = KeyedVectors.load(w2v_model)
    print('formatting w2v model for pytorch')
    np.random.seed(SEED)
    # Embeddings is a list, meaning we know that embeddings[1] is a vector for the word with ID=1, but we don't know what word this is.
    # That is why we need the id2word and word2id mappings.
    embeddings = []  # A list of embeddings for each word (represented by their idin the word2vec vocab
    id2word = {}  # gives coresspondence id <-> word
    word2id = {}  # gives coresspondence word <-> id
    len_vec = w2v_model.wv.vectors.shape[1]
    # First add <PAD> and <UNK>. we want PAD first so it corresponds to ID=0 and works out with pad_sequence function
    # For the word '<PAD>', embedding is set to all zeros
    id2word[len(embeddings)] = '<PAD>'
    word2id['<PAD>'] = len(embeddings)
    embeddings.append(np.zeros(len_vec))
    # For unknown words we use a random vector
    id2word[len(embeddings)] = '<UNK>'
    word2id['<UNK>'] = len(embeddings)
    embeddings.append(np.random.rand(len_vec))
    # load model vocabulary (depending what version was used, need to use .wv or not)
    try:
        keys = w2v_model.vocab.keys()
    except:
        keys = w2v_model.wv.vocab.keys()
    # assign ID to each word of the vocabulary
    for word in keys:
        id = len(embeddings)  # What is the position of this word in the embeddings list?
        id2word[id] = word  # Add mapping from ID to word
        word2id[word] = id  # From word to ID
        embeddings.append(w2v_model[word])  # Add the embedding for 'word', embeddings are available in the 'model'

    # Convert the embeddings list into a tensor of type float32
    # embeddings = torch.tensor(embeddings, dtype=torch.float32)
    return {'embeddings': torch.tensor(embeddings, dtype=torch.float32), 'id2word': id2word, 'word2id': word2id}


def get_wa(sentence, keywords, context=10, fixed_weights=False, debug=False):
    """
    generate vector of weighted averages given specific keywords in a tokenized sentence.
    the words closest to the keyword will get maximum weight etc...
    :param sentence: tokenized sentence
    :param keywords: keywords to look for in the sentence (can be either a list or string)
    :param context: number of words before/after keyword to use
    :param fixed_weights: set to true to use weight=1 for all words in context
    :param debug: printouts for debugging
    :return: array of weights
    """
    if debug:
        print('assigning word weights using:\n', ('fixed weights' if fixed_weights else 'decreasing weights'),
              '\nkeywords:', keywords, '\ncontext:', context)
    sentence = to_list(sentence)
    keywords = to_list(keywords)
    kw_idx = [sentence.index(s) for s in sentence if any(xs in s for xs in keywords)]
    weights = [0] * len(sentence)
    context_weights = [0] + [1] * context if fixed_weights else list(np.arange(0, 1 + 1 / context, 1 / context))
    for i in kw_idx:
        left = context_weights[-i:] if i > 0 else []
        right = context_weights[::-1][:len(sentence) - i - 1]
        lst = [0] * (i - context - 1) + left + [1] + right + [0] * (len(sentence) - i - context - 2)
        weights = np.maximum(weights, lst)
    if debug: print(sentence, weights)
    return weights


def embed_sentences(tkn_sentences, w2v_model, do_avg=True, use_weights=False, **kwargs):
    """
    convert sentences to embedded sentences using pre-trained Word2Vec model
    :param sentences: tokenized sentences
    :param w2v_model: Word2Vec model
    :param do_avg: (True or False) if set to True, compute average of embedding vectors (instead of storing them for each word)
    :param use_weights: (True or False) use weighted average instead of simple average, based on distance from specific keyword(s)
    :return: embedded sentences
    """

    # initialize array to store embedded sentences
    if isinstance(w2v_model, str):  # load model from disk if needed
        print('loading embedding model from disk')
        w2v_model = KeyedVectors.load(w2v_model)
    emb_size = w2v_model.wv.vector_size
    size = emb_size if do_avg else max([len(snt) for snt in tkn_sentences]) * emb_size
    sentences_emb = np.zeros((len(tkn_sentences), size))

    # convert each sentence into the average sum of the vector representations of its tokens
    not_in_model = []
    for i_snt, snt in enumerate(tkn_sentences):  # Loop over sentences
        cnt = 0
        if use_weights:
            if i_snt == 0: print('embedding using Word2Vec model (using average sum of tokens)')
            debug = True if i_snt < 2 else False  # print first 2 sentences to check
            weights = gutils.get_wa(sentence=snt, debug=debug, **kwargs)
        for i_word, word in enumerate(snt):  # Loop over the words of a sentence
            if word in w2v_model.wv:
                word_emb = w2v_model.wv.get_vector(word) * (weights[i_word] if use_weights else 1)
                if do_avg:
                    sentences_emb[i_snt] += word_emb
                    cnt += 1
                else:
                    sentences_emb[i_snt, (i_word * emb_size):((i_word + 1) * emb_size)] = word_emb
            else:
                not_in_model.append(word)
        if cnt > 0 and not use_weights:
            sentences_emb[i_snt] = sentences_emb[i_snt] / cnt

    excluded_words = list(dict.fromkeys(not_in_model))
    print(len(excluded_words), 'words not in model')  # , excluded_words)
    return sentences_emb


"""# Functions to evaluate models"""


# Function to calculate performance of our predictions vs labels
def perf_metrics(preds, labels, average='weighted', debug=False):
    try:
        pred_flat = np.argmax(preds, axis=1).flatten()
        # pred_flat = torch.max(pred_vec, 1)[1]
    except:
        print('only 1 dimension in labels prediction, no need for argmax')
        pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    acc = accuracy_score(labels_flat, pred_flat)
    f1 = f1_score(labels_flat, pred_flat, average=average)
    p = precision_score(labels_flat, pred_flat, average=average)
    r = recall_score(labels_flat, pred_flat, average=average)
    if debug:
        print("PERF -- Acc: {:.3f} F1: {:.3f} Precision: {:.3f} Recall: {:.3f}".format(acc, f1, p, r))
    return {'f1': f1, 'acc': acc, 'p': p, 'r': r}


def perf_metrics_classes(preds, labels):
    try:
        pred_flat = np.argmax(preds, axis=1).flatten()
    except:
        print('only 1 dimension in labels prediction, no need for argmax')
        pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    report = classification_report(labels_flat, pred_flat, output_dict=True)
    df = pd.DataFrame(report).sort_index().transpose()
    # df['accuracy'] = accuracy_score(labels, preds)

    return df


# TO RUN K-FOLD VALIDATION
def run_KFOLD(dataset, base_model, model_trainer, base_model_loader=None,
              n_splits=10, random_state=42, n_epochs=5, **kwargs):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    labels = pd.Series([1] * len(dataset))
    # tracking variables
    best_f1, fold_nb = 0, 0
    stats, stats_classes = pd.DataFrame(), pd.DataFrame()
    # run k fold
    for train_ix, test_ix in kfold.split(labels, labels):
        fold_nb += 1
        # need to load each time otherwise remembers training from previous fold
        if base_model_loader is not None:
            base_model_tmp = base_model_loader(base_model, **kwargs)
        else:
            base_model_tmp = base_model
        print('####################### RUNNING FOLD:', fold_nb)
        train_dataset = torch.utils.data.Subset(dataset, train_ix)
        val_dataset = torch.utils.data.Subset(dataset, test_ix)
        print(type(train_dataset), ' train set:', len(train_dataset), ' test set:', len(val_dataset))
        train_dataloader, validation_dataloader = create_dataloader(train_dataset, val_dataset)
        print('training and evaluating model')
        res = model_trainer(base_model_tmp, train_dataloader, validation_dataloader, n_epochs=n_epochs, output_dir=None)
        del base_model_tmp
        # store perf metrics and model
        stats_tmp = pd.DataFrame.from_dict(res['stats'], orient='index', columns=['value'])
        stats_tmp['fold'] = fold_nb
        stats = pd.concat([stats, stats_tmp])
        res['stats_classes']['fold'] = fold_nb
        stats_classes = pd.concat([stats_classes, res['stats_classes']])
        if res['stats']['f1'] >= best_f1:
            best_f1 = res['stats']['f1']
            res_to_save = res

    print('best F1 score obtained across splits: {:.3f}'.format(best_f1))
    return {'stats': stats, 'stats_classes': stats_classes, 'model': res_to_save['model']}


"""# Functions to handle devices"""


def to_cpu(vec, detach=True):
    try:
        vec = vec.detach().cpu().numpy() if detach else vec.to('cpu').numpy()
    except:
        vec = vec.detach().numpy() if detach else vec.numpy()
    return vec


def model_to_cpu(model):
    try:
        model.to("cpu")
    except:
        pass


def batch_to_gpu(batch, device=None):
    if device is None:
        return batch
    try:  # we're using a GPU
        batch = tuple(t.to(device) for t in batch)
    except:
        batch = tuple(t for t in batch)
    return batch


def get_device():
    # specify GPU device
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device != 'cpu':
            print('number GPUs used:', torch.cuda.device_count())
            print('device name:', torch.cuda.get_device_name(0))
    except:
        device = None
        print('no CUDA capable device detected')
    return device


# specify GPU device
device = get_device()

import pickle

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sentence_classifier.sentence_classifier.NLP_utils import *

"""# Prepare dataset"""


def prep_classifier_dataset(sentences, labels, emb_model='tfidf', max_features_idf=1000, test_size=0.2, output_dir=None,
                            **kwargs):
    """

    :param sentences:
    :param labels:
    :param emb_model:
    :param max_features_idf:
    :param test_size:
    :param output_dir:
    :param kwargs: 
    :return:
    """
    # classification using tfidf
    if 'tfidf' in str(emb_model).lower():
        print('classification using tfidf')
        X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=test_size)
        tfidf_vect = TfidfVectorizer(max_features=max_features_idf, strip_accents='ascii')
        tfidf_vect_fit = tfidf_vect.fit(X_train)
        tfidf_train = tfidf_vect_fit.transform(X_train)
        X_train_vect = pd.DataFrame(tfidf_train.toarray(), columns=tfidf_vect.get_feature_names())
        tfidf_test = tfidf_vect_fit.transform(X_test)
        X_test_vect = pd.DataFrame(tfidf_test.toarray(), columns=tfidf_vect.get_feature_names())
        if output_dir is not None:
            try:
                pickle.dump(tfidf_vect, open(output_dir + '_tfidf.pickle', 'wb'))
            except:
                print('tfidf model not saved, please enter valid path')
    # classification using word2vec
    else:
        print('classification using word2vec, glove or fasttext')
        tkn_sentences = tokenize_spacy(sentences)
        if isinstance(emb_model, str): emb_model = KeyedVectors.load(emb_model)
        emb_sentences = embed_sentences(tkn_sentences, w2v_model=emb_model, **kwargs)
        emb_sentences = pd.DataFrame(emb_sentences)
        X_train_vect, X_test_vect, y_train, y_test = train_test_split(emb_sentences, labels, test_size=test_size)
        print('size x_train:', X_train_vect.shape, 'size x_test:', X_test_vect.shape)
    return [X_train_vect, X_test_vect, y_train, y_test]


"""# Functions to train/evaluate model"""


def train_classifier(sentences, labels, classifier, emb_model='tfidf', SEED=0, test_size=0.2, output_dir=None,
                     **kwargs):
    # split data and prepare model
    X_train_vect, X_test_vect, y_train, y_test = prep_classifier_dataset(sentences=sentences, labels=labels,
                                                                         emb_model=emb_model, test_size=test_size,
                                                                         output_dir=output_dir, **kwargs)
    model = classifier.fit(X_train_vect, y_train)
    y_train_pred = model.predict(X_train_vect)
    y_test_pred = model.predict(X_test_vect)
    # performance metrics
    tr_perf = perf_metrics(y_train_pred, np.array(y_train), average='weighted')
    tr_perf_classes = perf_metrics_classes(y_train_pred, np.array(y_train))
    eval_perf = perf_metrics(y_test_pred, np.array(y_test), average='weighted')
    tr_perf_classes = perf_metrics_classes(y_train_pred, np.array(y_train))
    tr_perf_classes['dataset'] = 'train'
    eval_perf_classes = perf_metrics_classes(y_test_pred, np.array(y_test))
    eval_perf_classes['dataset'] = 'test'
    print("TRAIN -- F1: {:.3f} Acc: {:.3f} P: {:.3f} R: {:.3f}".format(tr_perf['f1'], tr_perf['acc'], tr_perf['p'],
                                                                       tr_perf['r']))
    print("TEST -- F1: {:.3f} Acc: {:.3f} P: {:.3f} R: {:.3f}".format(eval_perf['f1'], eval_perf['acc'], eval_perf['p'],
                                                                      eval_perf['r']))
    try:  # only works for some classifiers
        print('The most important features are:',
              sorted(zip(model.feature_importances_, X_train_vect.columns), reverse=True)[0:10])
    except:
        pass
    if output_dir is not None:
        try:
            print('saving model...', output_dir)
            joblib.dump(model, output_dir)
            pd.concat([tr_perf_classes, eval_perf_classes]).to_csv(output_dir + '_stats.csv', header=True)
        except:
            print('model not saved, please enter valid path')
    return {'stats': eval_perf, 'stats_classes': pd.concat([tr_perf_classes, eval_perf_classes]), 'model': model}


def classifier_KFOLD(sentences, labels, classifier, emb_model='tfidf', max_features_idf=1000,
                     n_splits=10, random_state=42, output_dir=None, **kwargs):
    # labels = convert_to_cat(labels, binary=False)['labels']
    if 'tfidf' not in str(emb_model).lower():
        tkn_sentences = tokenize_spacy(sentences)
        emb_sentences = embed_sentences(tkn_sentences, w2v_model=emb_model, **kwargs)
        emb_sentences = pd.DataFrame(emb_sentences)
        print('dimensions of embeddings', emb_sentences.shape)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # tracking variables
    best_f1, fold_nb = 0, 0
    stats, stats_classes = pd.DataFrame(), pd.DataFrame()
    # run k fold
    for train_ix, test_ix in kfold.split(labels, labels):
        fold_nb += 1
        print('####################### RUNNING FOLD:', fold_nb)
        y_train, y_test = labels[train_ix], labels[test_ix]
        print('size y_train:', y_train.shape, 'size y_test:', y_test.shape)
        if 'tfidf' in str(emb_model).lower():
            print('classification using tfidf')
            X_train, X_test = sentences[train_ix], sentences[test_ix]
            tfidf_vect = TfidfVectorizer(max_features=max_features_idf, strip_accents='ascii')
            tfidf_vect_fit = tfidf_vect.fit(X_train)
            tfidf_train = tfidf_vect_fit.transform(X_train)
            X_train_vect = pd.DataFrame(tfidf_train.toarray(), columns=tfidf_vect.get_feature_names())
            tfidf_test = tfidf_vect_fit.transform(X_test)
            X_test_vect = pd.DataFrame(tfidf_test.toarray(), columns=tfidf_vect.get_feature_names())
        else:
            print('classification using word2vec, glove or fasttext')
            X_train_vect = emb_sentences.iloc[train_ix, :]
            X_test_vect = emb_sentences.iloc[test_ix, :]
        print('size x_train:', X_train_vect.shape, 'size x_test:', X_test_vect.shape)

        # run classifier
        model = classifier.fit(X_train_vect, y_train)
        y_train_pred = model.predict(X_train_vect)
        y_test_pred = model.predict(X_test_vect)

        # store perf metrics and model
        tr_perf = perf_metrics(y_train_pred, np.array(y_train), average='weighted')
        eval_perf = perf_metrics(y_test_pred, np.array(y_test), average='weighted')
        tr_perf_classes = perf_metrics_classes(y_train_pred, np.array(y_train))
        tr_perf_classes['dataset'], tr_perf_classes['fold'] = 'train', fold_nb
        eval_perf_classes = perf_metrics_classes(y_test_pred, np.array(y_test))
        eval_perf_classes['dataset'], eval_perf_classes['fold'] = 'test', fold_nb
        print("TRAIN -- F1: {:.3f} Acc: {:.3f} P: {:.3f} R: {:.3f}".format(tr_perf['f1'], tr_perf['acc'], tr_perf['p'],
                                                                           tr_perf['r']))
        print("TEST -- F1: {:.3f} Acc: {:.3f} P: {:.3f} R: {:.3f}".format(eval_perf['f1'], eval_perf['acc'],
                                                                          eval_perf['p'], eval_perf['r']))
        stats = stats.append(pd.DataFrame([eval_perf]))
        stats_classes = stats_classes.append(pd.concat([tr_perf_classes, eval_perf_classes]))
        if eval_perf['f1'] >= best_f1:
            best_f1 = eval_perf['f1']
            model_to_save = model

    # save model with best f1
    if output_dir is not None:
        try:
            print('saving model in:', output_dir)
            joblib.dump(model, output_dir)
            stats_classes.to_csv(output_dir + '_stats.csv', header=True)
        except:
            print('model not saved, please enter valid path')

    print('best F1 score obtained across splits: {:.3f}'.format(best_f1))
    return {'stats': stats, 'stats_classes': stats_classes, 'model': model_to_save}


"""# Function to load saved model and classify new data"""


def load_and_run_classifier(sentences, trained_classifier, emb_model, **kwargs):
    sentences = pd.Series(sentences)
    if 'tfidf' in str(emb_model).lower():
        print('classification using tfidf')
        if isinstance(emb_model, str):
            print('loading model from disk...', emb_model)
            emb_model = pickle.load(open(emb_model, 'rb'))
        emb_sentences = emb_model.transform(sentences)
    else:
        print('classification using word2vec, glove or fasttext')
        tkn_sentences = tokenize_spacy(sentences)
        emb_sentences = embed_sentences(tkn_sentences, w2v_model=emb_model, **kwargs)
        emb_sentences = pd.DataFrame(emb_sentences)
    if isinstance(trained_classifier, str):
        print('loading classifier from disk...', trained_classifier)
        trained_classifier = joblib.load(trained_classifier)
    y_pred = trained_classifier.predict(emb_sentences)
    # put results in nice format
    res = pd.DataFrame()
    res['preds'] = y_pred
    res['sentences'] = sentences
    return res

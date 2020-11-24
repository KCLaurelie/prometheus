"""Main module."""
import gensim.downloader as api
from sklearn import svm
from sentence_classifier.sentence_classifier.BERT import *
from sentence_classifier.sentence_classifier.NN import *
from sentence_classifier.sentence_classifier.simple_classifiers import *


"""
Some examples on how to use the various functions
"""


"""# Load data"""

df = pd.read_csv('https://raw.githubusercontent.com/KCLaurelie/prometheus/master/sentence_classifier/sentence_classifier/imdb_5k_reviews.csv',
                 header=1)
text_col = 'review'
label_col = 'sentiment'
df = df[0:100]
df[text_col] = df[text_col].apply(lambda x: x.strip().lower()[0:1000])
print('num annotations:', len(df), '\n\n', df[label_col].value_counts(), '\n\n', df[[label_col, text_col]].head())

"""# Run BERT"""
BERT_tokenizer = 'bert-base-uncased'
n_epochs = 1  # 5
bert_model = train_BERT(sentences=df[text_col], labels=df[label_col], BERT_tokenizer=BERT_tokenizer,
                        test_size=0.2, n_epochs=n_epochs, output_dir=None)
bert_model['stats']

load_and_run_BERT(sentences=['hello my name is link i am in love with princess zelda', 'this is just a test sentence'],
                  trained_bert_model=bert_model['model'],
                  BERT_tokenizer=BERT_tokenizer)

kf = BERT_KFOLD(sentences=df[text_col], labels=df[label_col], n_splits=10, BERT_tokenizer=BERT_tokenizer, n_epochs=1,
                random_state=666)
print(kf['stats'], '\n\n', kf['stats_classes'])

"""# Run SVM"""
w2v_model = api.load("glove-wiki-gigaword-50")
emb_model_torch = embedding2torch(w2v_model, SEED=0)
svm_model = train_classifier(sentences=df[text_col], labels=df[label_col], emb_model=w2v_model,
                             classifier=svm.LinearSVC(multi_class='crammer_singer', class_weight='balanced'),
                             test_size=0.2, output_dir=None)

load_and_run_classifier(sentences=df[text_col][0:2],
                        trained_classifier=svm_model['model'],
                        emb_model=w2v_model)

"""# Run LSTM"""
n_epochs = 1
rnn_obj = RNN(emb_model_torch['embeddings'], padding_idx=emb_model_torch['word2id']['<PAD>']
              , rnn_type='lstm', bid=True, simulate_attn=True, debug_mode=False)
print(rnn_obj)
rnn_model = train_NN(rnn_obj, sentences=df[text_col], labels=df[label_col], emb_model=emb_model_torch,
                     tokenization_type='clean'
                     , SEED=0, test_size=0.2, n_epochs=n_epochs, output_dir=None)
print(rnn_model['stats'])

import os
import datetime
import json
import itertools
import copy
import shutil
import unidecode
import re
import string
import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
import contractions
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, \
    concatenate, Bidirectional, LSTM, SimpleRNN, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models.phrases import Phraser
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from tensorflow_core.python.keras import Input, Model
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

CHECK_ONLY = True


def check_preprocessor_arguments(kwargs):
    for k in ['num_unique_words', 'max_sequence_length']:
        assert isinstance(kwargs[k], int) and kwargs[k] > 0

    for k in ['trunc_type', 'pad_type']:
        assert kwargs[k] in {'pre', 'post'}

    assert kwargs['do_clean'] in {True, False}

    if kwargs['do_clean']:
        for k in ['ignore_urls', 'omit_stopwords', 'fix_contractions', 'stem', 'remove_foreign_characters', 'lower',
                  'remove_punctuation']:
            assert kwargs[k] in {True, False}
        if kwargs['lower']:
            assert kwargs['bigrams'] in {True, False}
        else:
            assert kwargs['bigrams'] is None
    else:
        for k in ['ignore_urls', 'omit_stopwords', 'fix_contractions', 'stem', 'remove_foreign_characters', 'lower',
                  'remove_punctuation', 'bigrams']:
            assert kwargs[k] is None


class Preprocessor:
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    bigram = Phraser.load('bigrams.pkl')
    word_tokenizer = RegexpTokenizer(r"\w+|[^\w\s]")
    re_url = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)'

    # add hyperparameters for preprocessor here
    def __init__(self, num_unique_words=None, max_sequence_length=None, trunc_type=None, pad_type=None,
                 do_clean=None, ignore_urls=None, omit_stopwords=None, fix_contractions=None, stem=None,
                 remove_foreign_characters=None, lower=None, remove_punctuation=None, bigrams=None):
        """
        Create a new Preprocessor object.

        :param num_unique_words: number of words in the vocabulary for the embedding.
        :param max_sequence_length: the number of dimensions for input data (num_messages, max_sequence_length)
        :param trunc_type: truncation type for pad_sequences
        :param pad_type: padding type for pad_sequences
        :param do_clean: whether or not to clean the messages
        :param ignore_urls: whether or not to remove URLs from messages
        :param omit_stopwords: whether or not to remove stop words
        :param fix_contractions: whether or not to replace contractions with their full phrases
        :param stem: whether or not to replace words with their stems
        :param remove_foreign_characters: whether or not to remove foreign characters
        :param lower: whether or not to convert to lower case
        :param remove_punctuation: whether or not to remove punctuation
        :param bigrams: whether or not to combine words into bigrams
        """
        check_preprocessor_arguments(locals())
        self.num_unique_words = num_unique_words
        self.do_clean = do_clean
        self.max_sequence_length = max_sequence_length
        self.trunc_type = trunc_type
        self.pad_type = pad_type
        self.tokenizer = None
        self.omit_stopwords = omit_stopwords
        self.ignore_urls = ignore_urls
        self.fix_contractions = fix_contractions
        self.stem = stem
        self.remove_foreign_characters = remove_foreign_characters
        self.lower = lower
        self.remove_punctuation = remove_punctuation
        self.bigrams = bigrams
        self.stemmer = None
        if stem:
            self.stemmer = PorterStemmer()

    def clean(self, x):  # should not contain any other arguments (use fields set in constructor instead).
        """
        Clean the strings in 'x' inspired by
        https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/natural_language_preprocessing.ipynb

        :param x: A list of strings, one for each message
        :return: A cleaned list of strings
        """

        def repl(m):
            return chr(int('0x' + m.group(1), 16))

        # replace double escaped "\\" unicode strings with their unicode characters
        x = [re.sub(r'\\n', '\n', message) for message in x]
        x = [re.sub(r'\\x([a-f0-9]{2})', repl, message) for message in x]
        x = [re.sub(r'\\u([a-f0-9]{4})', repl, message) for message in x]
        if self.ignore_urls:
            x = [re.sub(self.re_url, '', message) for message in x]

        if self.fix_contractions:
            x = [contractions.fix(message) for message in x]

        if self.remove_foreign_characters:
            # replace accented characters with unaccented
            x = [unidecode.unidecode(message) for message in x]

            # replace nonascii characters with space
            x = [''.join(character if ord(character) < 128 else ' ' for character in message) for message in x]

        # Create sentence structure like nltk gutenberg.sents()
        # list of sentences for each message:
        x = [self.sent_detector.tokenize(message.strip()) for message in x]
        # list of list of words for each message/sentence:
        x = [[self.word_tokenizer.tokenize(sentence) for sentence in message] for message in x]

        if self.lower:
            # lower_sents: lowercase words ignoring punctuation
            x = [[[
                word.lower() for word in sentence] for sentence in message
            ] for message in x]

        if self.remove_punctuation:
            x = [[[
                word for word in sentence if word not in list(string.punctuation)] for sentence in message
            ] for message in x]

        if self.stem:
            x = [[[self.stemmer.stem(word) for word in sentence] for sentence in message] for message in x]

        if self.lower and self.bigrams:
            # clean_sents: replace common adjacent words with bigrams
            x = [[self.bigram[sentence] for sentence in message] for message in x]

        if self.omit_stopwords:
            x = [[[word for word in sentence if word not in stopwords.words('english')] for sentence in message] for
                 message in x]

        # convert back to one string per message (join words into sentences and sentences into messages)
        x = ['\n'.join(' '.join(sentence) for sentence in message) for message in x]
        return x

    def fit(self, x):  # takes no other parameters (use fields initialized in constructor instead).
        """
        Use training data 'x' to learn the parameters for preprocessing.

        :param x: list of messages.
        :return: self
        """
        if self.do_clean:
            x = self.clean(x)
        self.tokenizer = Tokenizer(num_words=self.num_unique_words)
        self.tokenizer.fit_on_texts(x)
        # other fitting?
        return self

    def transform(self, x):  # takes no other parameters (use fields initialized in constructor instead).
        """
        Return transformed list of strings into an ndarray ready for input to Keras model.

        :param x: list of messages (strings)
        :return: ndarray of data (num_messages, max_sequence_length)
        """
        if self.do_clean:
            x = self.clean(x)
        if self.tokenizer is None:
            raise ValueError('Tokenizer has not been initialized.')
        # other transforming to produce tensor for input layer of model
        x = self.tokenizer.texts_to_sequences(x)
        return pad_sequences(x, maxlen=self.max_sequence_length, padding=self.pad_type, truncating=self.trunc_type,
                             value=0)


def get_performance(model, data_sets, set_names):
    """
    Helper function to compute the performance (balanced accuracy and balanced log_loss).

    :param model: A Keras model.
    :param data_sets: A list of datasets: [(x1, y1), (x2, y2), ...]
    :param set_names: A list of names: ['train', 'valid', ...]
    :return: results dict: {'name': {'accuracy': accuracy, 'loss': loss}, ...}
    """
    results = {}
    for (x, y), name in zip(data_sets, set_names):
        y_hat = model.predict(x)
        sample_weight = get_sample_weight(y)
        # should be same as balanced accuracy
        # acc = accuracy_score(y_true=y, y_pred=y_hat.argmax(axis=1), sample_weight=sample_weight)
        acc = balanced_accuracy_score(y_true=y, y_pred=y_hat.argmax(axis=1))
        loss = log_loss(y_true=y, y_pred=y_hat.astype(np.float64), sample_weight=sample_weight)
        results[name] = {
            'accuracy': acc,
            'loss': loss,
        }
    return results


def check_model(output_dir):
    """
    Load and check model from its output directory.

    :param output_dir: The directory that stores model.h5, params.json, and roatan.py.
    :return: None
    """
    import importlib.util

    model_file = os.path.join(output_dir, 'model.h5')
    model = load_model(model_file, compile=True)

    with open(os.path.join(output_dir, 'params.json'), 'r') as fp:
        params = json.load(fp)

    spec = importlib.util.spec_from_file_location('Preprocessor', os.path.join(output_dir, 'Preprocessor.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    this_preprocessor = getattr(mod, 'Preprocessor')
    classes, data_sets, set_names = get_xy(this_preprocessor(**params['preprocessor'],
                                                             **params['model_preprocessor']))
    results = get_performance(model, data_sets, set_names)

    # Check that matching data set names report same performance as in params.json
    for name in results.keys():
        if name in params['results']:
            assert params['results'][name] == results[name], \
                f'Loaded model reports different results than stored: {params["results"][name]} != {results[name]}'

    print(output_dir)
    print(pd.DataFrame(data=results).T)


def get_xy(preprocessor, target='Coding:Level1'):
    """
    Get the data from CSV files depending on target.

    :param preprocessor: A Preprocessor object ready to be fit to training data.
    :param target: The target 'Coding:Level1', 'Coding:Level2', other column
    :return: list of class names: ['label1', ...],
            list of datasets: [(x_train, y_train), (x_valid, y_valid)],
            list of names: ['train', 'valid']
    """
    set_names = ['train', 'valid']
    dfs = [pd.read_csv(f'roatan_{s}.csv') for s in set_names]

    # fit preprocessor with training set
    preprocessor.fit(dfs[0]['message'])
    # transform all data sets
    xs = [preprocessor.transform(df['message']) for df in dfs]

    # encode labels as integers 0 ... n-1 using training set
    le = LabelEncoder().fit(dfs[0][target])
    # transform labels for all data sets
    ys = [le.transform(df[target]) for df in dfs]

    classes = le.classes_
    data_sets = list(zip(xs, ys))
    return classes, data_sets, set_names


def get_sample_weight(y):
    """
    Return sample weights so that each class counts equally. For unbalanced data sets,
    say 90% positive and 10% negative, this ensures that both classes are equally important.

    :param y: The list of class labels (integers)
    :return: A weight for each sample that corresponds to its class label
    """
    class_counts = np.bincount(y)
    class_weight = 1 / class_counts
    sample_weight = class_weight[y]
    sample_weight = sample_weight / sample_weight.sum() * len(y)
    return sample_weight


# noinspection PyTypeChecker
def assemble_results(output_root):
    """
    Helper function to traverse output root directory to assemble and save a CSV file with results.

    :param output_root: The directory that contains all the output model directories
    :return: A list of parameter sets (dicts), the dict for the best validation loss parameter set.
    """
    all_params = []
    for run in sorted(os.listdir(output_root)):
        run_dir = os.path.join(output_root, run)
        if os.path.isdir(run_dir):
            r = {'dir': run}
            json_file = os.path.join(run_dir, f'params.json')
            try:
                with open(json_file, 'r') as fp:
                    d = json.load(fp)
                    r.update(d)
            except (FileNotFoundError, KeyError) as e:
                print(str(e))
                print(f'removing {run_dir}')
                shutil.rmtree(run_dir)
            all_params.append(r)

    data = [pd.json_normalize(d, sep='__').to_dict(orient='records')[0] for d in all_params]

    # save CSV file of all results
    csv_file = os.path.join(output_root, 'results.csv')
    pd.DataFrame(data).to_csv(csv_file, index=False)

    # assemble list of params to check what's been done
    best_val_loss = float('inf')
    best_params = None
    all_params2 = []
    for d in all_params:
        if 'results' in d:
            if d['results']['valid']['loss'] < best_val_loss:
                best_val_loss = d['results']['valid']['loss']
                best_params = copy.deepcopy(d)
            del d['results']
            del d['dir']
            all_params2.append(d)

    if best_params is not None:
        print(f'best params: {best_params}')
        print(f'best val loss: {best_params["results"]["valid"]["loss"]:.6f}')
        print(f'best val acc: {best_params["results"]["valid"]["accuracy"]:.4%}')
    return all_params2, best_params


def check_model_arguments(kwargs):
    for k in ['name', 'optimizer']:
        assert isinstance(kwargs[k], str)

    assert kwargs['activation'] is None or isinstance(kwargs['activation'], str)
    assert kwargs['num_classes'] in [3, 12]
    assert kwargs['learning_rate'] is None or kwargs['learning_rate'] > 0

    for k in ['num_unique_words', 'embedded_dims', 'max_sequence_length']:
        assert isinstance(kwargs[k], int) and kwargs[k] > 0

    drops = ['drop_embed', 'drop_h1', 'drop_h2', 'drop_h3']
    for k in drops:
        assert kwargs[k] is None or (isinstance(kwargs[k], float) and 0 < kwargs[k] < 1)

    neurons = ['num_units_h1', 'num_units_h2', 'num_units_h3']
    for k in neurons:
        assert kwargs[k] is None or (isinstance(kwargs[k], int) and kwargs[k] > 0)

    kernels = ['k_conv_h1', 'k_conv_h2', 'k_conv_h3']
    for k in kernels:
        assert kwargs[k] is None or (isinstance(kwargs[k], int) and kwargs[k] > 0)

    params = set(drops) | set(neurons) | set(kernels) | {'activation'}

    optional = set()
    if kwargs['name'] == 'dense_h1':
        required = {'activation', 'num_units_h1'}
        optional = {'drop_h1'}
    elif kwargs['name'] == 'dense_h2':
        required = {'num_units_h1', 'activation', 'num_units_h2'}
        optional = {'drop_h1', 'drop_h2'}
    elif kwargs['name'] == 'conv_h1':
        required = {'drop_embed', 'num_units_h1', 'k_conv_h1', 'activation'}
    elif kwargs['name'] == 'conv_h2':
        required = {'drop_embed', 'num_units_h1', 'k_conv_h1', 'activation', 'num_units_h2'}
        optional = {'drop_h2'}
    elif kwargs['name'] == 'conv_h3':
        required = {'drop_embed', 'num_units_h1', 'k_conv_h1', 'activation', 'num_units_h2'}
        optional = {'drop_h2'}
    elif kwargs['name'] == 'conv_h2.1':
        required = {'drop_embed', 'num_units_h1', 'k_conv_h1', 'activation'}
    elif kwargs['name'] == 'rnn_h1':
        required = {'drop_embed', 'num_units_h1'}
    elif kwargs['name'] == 'lstm_h1':
        required = {'drop_embed', 'num_units_h1', 'drop_h1'}
    elif kwargs['name'] == 'lstm_h2':
        required = {'drop_embed', 'num_units_h1', 'drop_h1', 'num_units_h2', 'activation'}
    elif kwargs['name'] == 'bi_lstm_h1':
        required = {'drop_embed', 'num_units_h1'}
        optional = {'drop_h1'}
    elif kwargs['name'] == 'bi_lstm_h2':
        required = {'num_units_h1', 'num_units_h2'}
        optional = {'drop_embed', 'drop_h1', 'drop_h2'}
    elif kwargs['name'] == 'multi_conv_h3_s2':
        required = {'drop_embed', 'num_units_h1', 'k_conv_h1', 'activation', 'num_units_h2', 'k_conv_h2',
                    'num_units_h3', 'num_units_h4', 'drop_h3'}
    elif kwargs['name'] == 'multi_conv_h3_s3':
        required = {'drop_embed', 'num_units_h1', 'k_conv_h1', 'activation', 'num_units_h2', 'k_conv_h2',
                    'num_units_h3', 'k_conv_h3', 'num_units_h4', 'drop_h4'}
    else:
        assert False, f'Unknown model name {kwargs["name"]}'

    unacceptable = params - (required | optional)
    for r in required:
        assert kwargs[r] is not None, f'For {kwargs["name"]}, "{r}" should not be None'
    for u in unacceptable:
        assert kwargs[u] is None, f'For {kwargs["name"]}, "{u}" = {kwargs[u]} should be None'


def build_fn(name=None, num_classes=None,
             optimizer=None, learning_rate=None, activation=None,
             num_unique_words=None, embedded_dims=None, max_sequence_length=None, drop_embed=None,
             num_units_h1=None, num_units_h2=None, num_units_h3=None, num_units_h4=None, num_units_h5=None,
             drop_h1=None, drop_h2=None, drop_h3=None, drop_h4=None,
             k_conv_h1=None, k_conv_h2=None, k_conv_h3=None, k_conv_h4=None
             ):
    """
    Build and compile a model based on the input parameters.

    :param name: name of the model.
    :param num_classes: number of outputs for the model.
    :param optimizer: string representation of the optimizer.
    :param learning_rate: learning rate for optimizer (None = default).
    :param activation: name of activation function to be used with hidden layers
    :param num_unique_words: number of words in the vocabulary for the embedding.
    :param embedded_dims: the number of dimensions to embed the tokens (words) in.
    :param max_sequence_length: the number of dimensions for input data (num_messages, max_sequence_length)
    :param drop_embed: dropout for the embedding layer
    :param num_units_h1: Number of neurons in first hidden layer
    :param num_units_h2: Number of neurons in second hidden layer (if applicable)
    :param num_units_h3: Number of neurons in third hidden layer (if applicable)
    :param num_units_h4: Number of neurons in fourth hidden layer (if applicable)
    :param num_units_h5: Number of neurons in fifth hidden layer (if applicable)
    :param drop_h1: Dropout for first hidden layer (if applicable)
    :param drop_h2: Dropout for second hidden layer (if applicable)
    :param drop_h3: Dropout for third hidden layer (if applicable)
    :param drop_h4: Dropout for fourth and following hidden layer (if applicable)
    :param k_conv_h1: Kernel size for first convolutional layer (if applicable)
    :param k_conv_h2: Kernel size for second convolutional layer (if applicable)
    :param k_conv_h3: Kernel size for third convolutional layer (if applicable)
    :param k_conv_h4: Kernel size for fourth convolutional layer (if applicable)
    :return: a Keras model ready to be fit on data with these dimensions.
    """
    check_model_arguments(locals())
    if CHECK_ONLY:
        return None
    clear_session()
    if 'multi_conv' not in name:
        model = Sequential(name=name)
        model.add(Embedding(num_unique_words, embedded_dims, input_length=max_sequence_length))
    else:
        model = None

    # get the optimizer with this name and learning rate (None = default)
    config = {} if learning_rate is None else dict(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.deserialize({'class_name': optimizer, 'config': config})

    if name == 'dense_h1':
        model.add(Flatten())
        model.add(Dense(num_units_h1, activation=activation))
        if drop_h1:
            model.add(Dropout(drop_h1))
    elif name == 'dense_h2':
        model.add(Flatten())
        model.add(Dense(num_units_h1, activation=activation))
        if drop_h1:
            model.add(Dropout(drop_h1))
        model.add(Dense(num_units_h2, activation=activation))
        if drop_h2:
            model.add(Dropout(drop_h2))
    elif name == 'conv_h1':
        model.add(SpatialDropout1D(drop_embed))
        model.add(Conv1D(num_units_h1, kernel_size=k_conv_h1, activation=activation))
        model.add(GlobalMaxPooling1D())
    elif name == 'conv_h2':
        model.add(SpatialDropout1D(drop_embed))
        model.add(Conv1D(num_units_h1, kernel_size=k_conv_h1, activation=activation))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(num_units_h2, activation=activation))
        if drop_h2:
            model.add(Dropout(drop_h2))
    elif name == 'conv_h3':
        model.add(SpatialDropout1D(drop_embed))
        model.add(Conv1D(num_units_h1, kernel_size=k_conv_h1, activation=activation))
        model.add(Conv1D(num_units_h1 * 2, kernel_size=k_conv_h1, activation=activation))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(num_units_h2, activation=activation))
        if drop_h2:
            model.add(Dropout(drop_h2))
    elif name == 'conv_h2.1':
        model.add(SpatialDropout1D(drop_embed))
        model.add(Conv1D(num_units_h1, kernel_size=k_conv_h1, activation=activation))
        model.add(Conv1D(num_units_h1, kernel_size=k_conv_h1, activation=activation))
        model.add(GlobalMaxPooling1D())
    elif name == 'rnn_h1':
        model.add(SpatialDropout1D(drop_embed))
        model.add(SimpleRNN(num_units_h1))
    elif name == 'lstm_h1':
        model.add(SpatialDropout1D(drop_embed))
        model.add(LSTM(num_units_h1, dropout=drop_h1))
    elif name == 'lstm_h2':
        model.add(SpatialDropout1D(drop_embed))
        model.add(LSTM(num_units_h1, dropout=drop_h1))
        model.add(Dense(num_units_h2, activation=activation))
    elif name == 'bi_lstm_h1':
        model.add(SpatialDropout1D(drop_embed))
        if drop_h1:
            model.add(Bidirectional(LSTM(num_units_h1, dropout=drop_h1)))
        else:
            model.add(Bidirectional(LSTM(num_units_h1)))
    elif name == 'bi_lstm_h2':
        if drop_embed:
            model.add(SpatialDropout1D(drop_embed))
        if drop_h1:
            model.add(Bidirectional(LSTM(num_units_h1, dropout=drop_h1, return_sequences=True)))
        else:
            model.add(Bidirectional(LSTM(num_units_h1, return_sequences=True)))
        if drop_h2:
            model.add(Bidirectional(LSTM(num_units_h2, dropout=drop_h2)))
        else:
            model.add(Bidirectional(LSTM(num_units_h2)))
    elif name == 'multi_conv_h3_s2':
        input_layer = Input(shape=(max_sequence_length,), dtype='int16', name='input')
        embedding_layer = Embedding(num_unique_words, embedded_dims, name='embedding')(input_layer)

        drop_embed_layer = SpatialDropout1D(drop_embed, name='drop_embed')(embedding_layer)

        # three parallel convolutional streams:
        conv_h1 = Conv1D(num_units_h1, kernel_size=k_conv_h1, activation=activation, name='conv_h1')(drop_embed_layer)
        max_pool_h1 = GlobalMaxPooling1D(name='max_pool_h1')(conv_h1)

        conv_h2 = Conv1D(num_units_h2, kernel_size=k_conv_h2, activation=activation, name='conv_h2')(drop_embed_layer)
        max_pool_h2 = GlobalMaxPooling1D(name='max_pool_h2')(conv_h2)

        # concatenate the activations from the three streams:
        concat = concatenate([max_pool_h1, max_pool_h2])

        # dense hidden layers:
        dense_layer = Dense(num_units_h3, activation=activation, name='dense')(concat)
        drop_dense_layer = Dropout(drop_h3, name='drop_dense')(dense_layer)
        dense_2 = Dense(num_units_h4, activation=activation, name='dense_2')(drop_dense_layer)
        dropout_2 = Dropout(drop_h3, name='drop_dense_2')(dense_2)

        # softmax output layer:
        predictions = Dense(num_classes, activation='softmax', name='output')(dropout_2)

        # create model:
        model = Model(input_layer, predictions)
    elif name == 'multi_conv_h3_s3':
        input_layer = Input(shape=(max_sequence_length,), dtype='int16', name='input')
        embedding_layer = Embedding(num_unique_words, embedded_dims, name='embedding')(input_layer)

        drop_embed_layer = SpatialDropout1D(drop_embed, name='drop_embed')(embedding_layer)

        # three parallel convolutional streams:
        conv_h1 = Conv1D(num_units_h1, kernel_size=k_conv_h1, activation=activation, name='conv_h1')(drop_embed_layer)
        max_pool_h1 = GlobalMaxPooling1D(name='max_pool_h1')(conv_h1)

        conv_h2 = Conv1D(num_units_h2, kernel_size=k_conv_h2, activation=activation, name='conv_h2')(drop_embed_layer)
        max_pool_h2 = GlobalMaxPooling1D(name='max_pool_h2')(conv_h2)

        conv_h3 = Conv1D(num_units_h3, kernel_size=k_conv_h3, activation=activation, name='conv_h3')(drop_embed_layer)
        max_pool_h3 = GlobalMaxPooling1D(name='max_pool_h3')(conv_h3)

        # concatenate the activations from the three streams:
        concat = concatenate([max_pool_h1, max_pool_h2, max_pool_h3])

        # dense hidden layers:
        dense_layer = Dense(num_units_h4, activation=activation, name='dense')(concat)
        drop_dense_layer = Dropout(drop_h4, name='drop_dense')(dense_layer)
        dense_2 = Dense(int(num_units_h5), activation=activation, name='dense_2')(drop_dense_layer)
        dropout_2 = Dropout(drop_h4, name='drop_dense_2')(dense_2)

        # softmax output layer:
        predictions = Dense(num_classes, activation='softmax', name='output')(dropout_2)

        # create model:
        model = Model(input_layer, predictions)

    else:
        raise ValueError(f'Unknown model name: {name}')

    if 'multi_conv' not in name:
        model.add(Dense(num_classes, activation='softmax', name='output'))

    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, weighted_metrics=['accuracy'])
    return model


# noinspection PyTypeChecker
def main():
    """
    Example of how to set up a grid search.

    :return: None
    """
    target = 'Coding:Level1'
    output_root = f'problem_5_output/{target.replace(":", "_")}'
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    # dictionary of parameter grids, one for each process
    param_grids = {
        'early_stopping': ParameterGrid([
            {
                'patience': [5],  # , 20, 40]
            },
        ]),
        'fit': ParameterGrid([
            {
                'batch_size': [32, 64, 128, 256],
                'epochs': [5, 20, 50],
            },
        ]),
        'model_preprocessor': ParameterGrid([
            {
                'num_unique_words': [1000, 4000, 5000, 6000, 10000],
                'max_sequence_length': [5, 50, 75, 100, 125, 150, 200],
            },
        ]),
        'model': ParameterGrid([
            {
                # Dense single hidden layer model hyperparameters:
                'name': ['dense_h1'],
                'embedded_dims': [8],  # , 16, 32, 64, 128, 256],
                'num_units_h1': [8],  # , 16, 32, 64, 128, 256],
                'drop_h1': [None],  # , 0.1, 0.2, 0.25, 0.5, 0.75],
                'optimizer': ['nadam', 'adam'],
                'learning_rate': [None],  # , 0.01, 0.001],
                'activation': ['relu', 'tanh'],
            },
            {
                # Dense double hidden layer model hyperparameters:
                'name': ['dense_h2'],
                'embedded_dims': [64],
                'num_units_h1': [128],
                'num_units_h2': [128],
                'drop_h1': [None],
                'drop_h2': [0.5],
                'optimizer': ['nadam'],
                'activation': ['relu'],
                'learning_rate': [0.01],
            },
            {
                # CNN single hidden layer model hyperparameters
                'name': ['conv_h1'],
                'embedded_dims': [64],
                'num_units_h1': [32],  # , 64, 256],
                'k_conv_h1': [2],  # , 3, 4],
                'drop_embed': [0.2],  # , 0.5],
                'activation': ['relu', 'tanh'],
                'optimizer': ['adam', 'nadam']
            },
            {
                # CNN double hidden layer model hyperparameters
                'name': ['conv_h2'],
                'embedded_dims': [128],  # , 64, 32, 16, 8],
                'num_units_h1': [32],  # , 64, 128],
                'drop_h2': [0.5],  # , 0.75, 0.25, 0.1],
                'k_conv_h1': [2],  # , 3, 4],
                'num_units_h2': [128],  # , 64, 32, 16, 8],
                'drop_embed': [0.2],  # , 0.50],
                'activation': ['relu'],
                'optimizer': ['adam'],  # , 'nadam'],
            },
            {
                # CNN double hidden layer model hyperparameters
                'name': ['conv_h2.1'],
                'embedded_dims': [64],
                'num_units_h1': [32],  # , 64, 128],
                'k_conv_h1': [2],  # , 3, 4],
                'drop_embed': [0.2],  # , 0.5],
                'activation': ['relu'],
                'optimizer': ['adam'],  # , 'nadam']
            },
            {
                # RNN single hidden layer model hyperparameters
                'name': ['rnn_h1'],
                'embedded_dims': [64],
                'drop_embed': [0.2],
                'num_units_h1': [128],
                'optimizer': ['nadam'],
                'learning_rate': [0.01]
            },
            {
                # LSTM double hidden layer (second layer dense FC) model hyperparameters
                'name': ['lstm_h1'],
                'embedded_dims': [64],
                'drop_embed': [0.2],
                'drop_h1': [0.5],
                'num_units_h1': [128],
                'optimizer': ['nadam'],
            },
            {
                # LSTM double hidden layer (second layer dense FC) model hyperparameters
                'name': ['lstm_h2'],
                'embedded_dims': [64],
                'drop_embed': [0.2],
                'num_units_h1': [128],
                'drop_h1': [0.5],
                'num_units_h2': [128],
                'optimizer': ['nadam'],
                'activation': ['relu']
            },
            {
                # Bi-directional LSTM single hidden layer model hyperparameters
                'name': ['bi_lstm_h1'],
                'embedded_dims': [32],  # , 64, 128],
                'drop_embed': [0.2],  # , 0.25, 0.5],
                'num_units_h1': [32],  # , 64, 128],
                'drop_h1': [0.2],  # , 0.25, 0.5],
                'optimizer': ['nadam', 'adam']
            },
            {
                # Bi-directional LSTM double hidden layer (second layer Bi-LSTM) model hyperparameters
                'name': ['bi_lstm_h2'],
                'embedded_dims': [32],  # , 64, 128],
                'num_units_h1': [32],  # , 64, 128],
                'num_units_h2': [32],  # , 64, 128],
                'drop_h1': [0.25, 0.5],
                'drop_h2': [0.25, 0.5],
                'optimizer': ['nadam', 'adam']
            },
            {
                # Multi Convolutional model hyperparameters
                'name': ['multi_conv_h3_s2'],
                'drop_embed': [0.5],  # , 0.3],
                'embedded_dims': [32],  # , 64, 128, 256],
                'num_units_h1': [32],  # , 64, 128, 256],
                'num_units_h2': [32],  # , 64, 128, 256],
                'num_units_h3': [32],  # , 64, 128, 256],
                'num_units_h4': [32],  # , 64, 128, 256],
                'k_conv_h1': [3],
                'k_conv_h2': [2],
                'activation': ['relu', 'tanh'],
                'drop_h3': [0.1],  # , 0.2, 0.25, 0.5, 0.75],
                'optimizer': ['adam', 'nadam']
            },
            {
                # Multi Convolutional model hyperparameters
                'name': ['multi_conv_h3_s3'],
                'drop_embed': [0.5],  # , 0.3],
                'embedded_dims': [32],  # , 64, 128, 256],
                'num_units_h1': [32],  # , 64, 128, 256],
                'num_units_h2': [32],  # , 64, 128, 256],
                'num_units_h3': [32],  # , 64, 128, 256],
                'num_units_h4': [32],  # , 64, 128, 256],
                'k_conv_h1': [3],
                'k_conv_h2': [2],
                'k_conv_h3': [4],
                'k_conv_h4': [4],
                'activation': ['relu', 'tanh'],
                'drop_h4': [0.1],  # , 0.2, 0.25, 0.5, 0.75],
                'optimizer': ['adam', 'nadam']
            },
        ]),
        'preprocessor': ParameterGrid([
            # {
            #     'do_clean': [False],
            #     'pad_type': ['pre', 'post'],
            #     'trunc_type': ['pre', 'post'],
            # },
            {
                'do_clean': [True],
                'pad_type': ['pre', 'post'],
                'trunc_type': ['pre', 'post'],
                'omit_stopwords': [True, False],
                'ignore_urls': [True, False],
                'fix_contractions': [True, False],
                'stem': [True, False],
                'remove_foreign_characters': [True],  # , False],
                'lower': [True],  # , False],
                'remove_punctuation': [True],  # , False],
                'bigrams': [True],  # , False]
            },
        ])
    }

    def prod(a):
        if len(a) == 0:
            return 1
        return a[0] * prod(a[1:])

    num_models = prod([len(pg) for pg in param_grids.values()])

    param_grid_names = sorted(list(param_grids.keys()))
    param_grid_list = [param_grids[k] for k in param_grid_names]

    all_params, best_params = assemble_results(output_root)

    if CHECK_ONLY:
        for i, params in enumerate(itertools.product(*param_grid_list[3:5])):
            params = {k: v for k, v in zip(param_grid_names[3:5], params)}
            print(i, params)
            Preprocessor(**params['preprocessor'], **params['model_preprocessor'])

        for i, params in enumerate(itertools.product(*param_grid_list[2:4])):
            params = {k: v for k, v in zip(param_grid_names[2:4], params)}
            print(i, params)
            build_fn(num_classes=3, **params['model'], **params['model_preprocessor'])
        return

    for i, params in enumerate(itertools.product(*param_grid_list)):
        params = {k: v for k, v in zip(param_grid_names, params)}
        print(f'\n{i + 1}/{num_models}: {params}\n')

        if params in all_params:
            # skip this one because we already ran it.
            continue

        if best_params is not None:
            # print best performance so far
            print(f'best params: {best_params}')
            print(f'best val loss: {best_params["results"]["valid"]["loss"]:.6f}')
            print(f'best val acc: {best_params["results"]["valid"]["accuracy"]:.4%}')

        # create a new output directory with path to model file.
        date = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H.%M.%S.%f")
        output_dir = os.path.join(output_root, date)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_file = os.path.join(output_dir, 'model.h5')

        # get the preprocessed training and validation data
        classes, data_sets, set_names = get_xy(Preprocessor(**params['preprocessor'], **params['model_preprocessor']),
                                               target=target)
        ((x_train, y_train), (x_valid, y_valid)) = data_sets

        # build and compile model
        model = build_fn(num_classes=len(classes), **params['model'], **params['model_preprocessor'])

        # setup callbacks
        early_stopping = EarlyStopping(monitor='val_loss', verbose=1, **params['early_stopping'])
        model_checkpoint = ModelCheckpoint(
            filepath=model_file,
            save_weights_only=False, save_freq='epoch',
            save_best_only=True, monitor='val_loss', verbose=1)
        callbacks = [early_stopping, model_checkpoint]

        # Use sample weights to treat classes equally in loss and accuracy.
        sample_weight = get_sample_weight(y_train)
        sample_weight_valid = get_sample_weight(y_valid)

        # fit the model
        model.fit(x=x_train, y=y_train, sample_weight=sample_weight, verbose=1,
                  validation_data=(x_valid, y_valid, sample_weight_valid), callbacks=callbacks, **params['fit'])

        # load the best model (last one saved)
        model = load_model(model_file, compile=True)

        # compute results
        results = get_performance(model, data_sets, set_names)
        print(pd.DataFrame(data=results).T)
        params['results'] = results

        # save params and results
        with open(os.path.join(output_dir, 'params.json'), 'w') as fp:
            json.dump(params, fp)

        # save a copy of *this* Python file.
        shutil.copyfile(__file__, os.path.join(output_dir, 'roatan.py'))

        # for convenience, show the validation loss and accuracy in a file name in the same directory.
        result_file_name = f'{params["results"]["valid"]["loss"]:.6f}_{params["results"]["valid"]["accuracy"]:.4f}.out'
        with open(os.path.join(output_dir, result_file_name), 'w'):
            pass

        # check_model(output_dir)

        if best_params is None or (params['results']['valid']['loss'] < best_params['results']['valid']['loss']):
            best_params = params

    # assemble results from all runs into one CSV file in output root.
    assemble_results(output_root)


if __name__ == '__main__':
    main()

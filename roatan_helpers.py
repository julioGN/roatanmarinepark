import nltk
import os
import datetime
import json
import itertools
import copy
import shutil
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import log_loss, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models.phrases import Phraser
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model  # new!
from tensorflow.keras.layers import Input, concatenate  # new!
from tensorflow.keras.layers import Dense, Dropout, Embedding, SpatialDropout1D, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

nltk.download('punkt')


class Preprocessor:
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    bigram = Phraser.load('bigrams.pkl')
    word_tokenizer = RegexpTokenizer(r"\w+|[^\w\s]")

    # add hyperparameters for preprocessor here
    def __init__(self, num_unique_words=5000, max_sequence_length=100, trunc_type='pre', pad_type='pre',
                 do_clean=False):
        self.num_unique_words = num_unique_words
        self.do_clean = do_clean
        self.max_sequence_length = max_sequence_length
        self.trunc_type = trunc_type
        self.pad_type = pad_type
        self.tokenizer = None

    def clean(self, x):  # should not contain any other arguments (use fields set in constructor instead).
        """
        Clean the strings in 'x' inspired by
        https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/natural_language_preprocessing.ipynb

        :param x: A list of strings, one for each message
        :return: A cleaned list of strings
        """
        import unidecode
        import re
        import string

        def repl(m):
            return chr(int('0x' + m.group(1), 16))

        # replace double escaped "\\" unicode strings with their unicode characters
        x = [re.sub(r'\\n', '\n', message) for message in x]
        x = [re.sub(r'\\x([a-f0-9]{2})', repl, message) for message in x]
        x = [re.sub(r'\\u([a-f0-9]{4})', repl, message) for message in x]

        # replace accented characters with unaccented
        x = [unidecode.unidecode(message) for message in x]

        # replace nonascii characters with space
        x = [''.join(character if ord(character) < 128 else ' ' for character in message) for message in x]

        # Create sentence structure like nltk gutenberg.sents()
        # list of sentences for each message:
        x = [self.sent_detector.tokenize(message.strip()) for message in x]
        # list of list of words for each message/sentence:
        x = [[self.word_tokenizer.tokenize(sentence) for sentence in message] for message in x]

        # lower_sents: lowercase words ignoring punctuation
        x = [[[
            word.lower() for word in sentence if word.lower() not in list(string.punctuation)] for sentence in message]
            for message in x]

        # stopwrds: remove stopwords and punctuation
        # x = [[[word for word in sentence if not word.lower() in stopwords.words('english')] for sentence in message]
        #      for message in x]

        # clean_sents: replace common adjacent words with bigrams
        x = [[self.bigram[sentence] for sentence in message] for message in x]
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
    dfs = [pd.read_csv(f'data/roatan_{s}.csv') for s in set_names]

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
            # noinspection PyTypeChecker
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


def build_fn(name=None, num_classes=None, optimizer=None, num_unique_words=None, embedded_dims=None,
             max_sequence_length=None, num_dense=None, dropout=None, spatial_dropout=None, n_conv_1=None,
             n_conv_2=None, n_conv_3=None, k_conv_1=None, k_conv_2=None, k_conv_3=None):
    """
    Build and compile a model based on the input parameters.

    :param spatial_dropout: The percent dropout in the convolutional layer.
    :param k_conv_3: convolutional layer size.
    :param k_conv_2: convolutional layer size.
    :param k_conv_1: convolutional layer size.
    :param n_conv_3: kernel size.
    :param n_conv_2: kernel size.
    :param n_conv_1: kernel size.
    :param name: name of the model.
    :param num_classes: number of outputs for the model.
    :param optimizer: string representation of the optimizer.
    :param num_unique_words: number of words in the vocabulary for the embedding.
    :param max_sequence_length: the number of dimensions for input data (num_messages, max_sequence_length)
    :param embedded_dims: the number of dimensions to embed the tokens (words) in.
    :param num_dense: The number of neurons in the dense layer
    :param dropout: The percent dropout for dense layer.
    :return: a Keras model ready to be fit on data with these dimensions.
    """
    clear_session()


    if name == 'dense':
        model = Sequential()
        model.add(Embedding(num_unique_words, embedded_dims, input_length=max_sequence_length))
        model.add(Flatten())
        model.add(Dense(num_dense, activation='relu'))
        model.add(Dropout(dropout))
    elif name == 'mcn':
        # convolutional layer architecture:

        input_layer = Input(shape=(max_sequence_length,),
                            dtype='int16', name='input')

        # embedding:
        embedding_layer = Embedding(num_unique_words, embedded_dims,
                                    name='embedding')(input_layer)
        drop_embed_layer = SpatialDropout1D(spatial_dropout,
                                            name='drop_embed')(embedding_layer)

        # three parallel convolutional streams:
        conv_1 = Conv1D(n_conv_1, k_conv_1,
                        activation='relu', name='conv_1')(drop_embed_layer)
        maxp_1 = GlobalMaxPooling1D(name='maxp_1')(conv_1)

        conv_2 = Conv1D(n_conv_2, k_conv_2,
                        activation='relu', name='conv_2')(drop_embed_layer)
        maxp_2 = GlobalMaxPooling1D(name='maxp_2')(conv_2)

        #  conv_3 = Conv1D(n_conv_3, k_conv_3,
        #                  activation='relu', name='conv_3')(drop_embed_layer)
        #  maxp_3 = GlobalMaxPooling1D(name='maxp_3')(conv_3)

        # concatenate the activations from the three streams:
        concat = concatenate([maxp_1, maxp_2])

        # dense hidden layers:
        dense_layer = Dense(num_dense,
                            activation='relu', name='dense')(concat)
        drop_dense_layer = Dropout(dropout, name='drop_dense')(dense_layer)
        dense_2 = Dense(int(num_dense / 2),
                        activation='relu', name='dense_2')(drop_dense_layer)
        dropout_2 = Dropout(dropout, name='drop_dense_2')(dense_2)

        # sigmoid output layer:
        predictions = Dense(3, activation='softmax', name='output')(dropout_2)
        #  model.add(Dense(num_classes, activation='softmax'))

        # create model:
        model = Model(input_layer, predictions)
    elif name == 'mcn_complex':
        # convolutional layer architecture:

        input_layer = Input(shape=(max_sequence_length,),
                            dtype='int16', name='input')

        # embedding:
        embedding_layer = Embedding(num_unique_words, embedded_dims,
                                    name='embedding')(input_layer)
        drop_embed_layer = SpatialDropout1D(spatial_dropout,
                                            name='drop_embed')(embedding_layer)

        # three parallel convolutional streams:
        conv_1 = Conv1D(n_conv_1, k_conv_1,
                        activation='relu', name='conv_1')(drop_embed_layer)
        maxp_1 = GlobalMaxPooling1D(name='maxp_1')(conv_1)

        conv_2 = Conv1D(n_conv_2, k_conv_2,
                        activation='relu', name='conv_2')(drop_embed_layer)
        maxp_2 = GlobalMaxPooling1D(name='maxp_2')(conv_2)

        conv_3 = Conv1D(n_conv_3, k_conv_3,
                        activation='relu', name='conv_3')(drop_embed_layer)
        maxp_3 = GlobalMaxPooling1D(name='maxp_3')(conv_3)

        # concatenate the activations from the three streams:
        concat = concatenate([maxp_1, maxp_2, maxp_3])

        # dense hidden layers:
        dense_layer = Dense(num_dense,
                            activation='relu', name='dense')(concat)
        drop_dense_layer = Dropout(dropout, name='drop_dense')(dense_layer)
        dense_2 = Dense(int(num_dense / 2),
                        activation='relu', name='dense_2')(drop_dense_layer)
        dropout_2 = Dropout(dropout, name='drop_dense_2')(dense_2)

        # sigmoid output layer:
        predictions = Dense(3, activation='softmax', name='output')(dropout_2)

        # create model:
        model = Model(input_layer, predictions)
    else:
        raise ValueError(f'Unknown model name: {name}')

    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, weighted_metrics=['accuracy'])

    return model


# noinspection PyTypeChecker
def main():
    """
    Example of how to set up a grid search.

    :return: None
    """
    output_root = 'problem_5_output'
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    # dictionary of parameter grids, one for each process
    param_grids = {
        'early_stopping': ParameterGrid([
            {
                'patience': [15, 30]
            },
        ]),
        'fit': ParameterGrid([
            {
                'batch_size': [128, 256],
                'epochs': [20, 40],
            },
        ]),
        'model_preprocessor': ParameterGrid([
            {
                'num_unique_words': [5000, 6000],
                'max_sequence_length': [100, 150],
            },
        ]),
        'model': ParameterGrid([
            {
                'name': ['mcn', 'mcn_complex'],
                'embedded_dims': [64, 128],
                'num_dense': [64, 128],
                'dropout': [0.5, 0.2],
                'spatial_dropout': [0.2, 0.2],
                'n_conv_1': [64, 128],
                'n_conv_2': [64, 128],
                'n_conv_3': [64, 128],
                'k_conv_1': [3, 3],
                'k_conv_2': [2, 3],
                'k_conv_3': [4, 3],
                'optimizer': ['adam', 'nadam'],
            },
        ]),
        'preprocessor': ParameterGrid([
            {
                'pad_type': ['post'],
                'trunc_type': ['pre'],
                'do_clean': [True]
            },
        ])
    }

    def prod(a):
        if len(a) == 0:
            return 1
        return a[0] * prod(a[1:])

    num_models = prod([len(pg) for pg in param_grids.values()])

    param_grid_names = list(param_grids.keys())
    param_grid_list = [param_grids[k] for k in param_grid_names]

    all_params, best_params = assemble_results(output_root)

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
        date = datetime.datetime.utcnow().strftime("%Y-%m-%d %H.%M.%S.%f'")
        output_dir = os.path.join(output_root, date)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_file = os.path.join(output_dir, 'model.h5')

        # get the preprocessed training and validation data
        classes, data_sets, set_names = get_xy(Preprocessor(**params['preprocessor'], **params['model_preprocessor']))
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

__author__ = 'parryrm'
import unittest
import numpy as np
import math
import re
from copy import deepcopy
import os
import functools
from sklearn.datasets import load_iris
import scipy
import json
iris = load_iris()


def compose(*functions):
    def compose2(f, g):
        return lambda *args, **kwargs: f(g(*args, **kwargs))
    return functools.reduce(compose2, functions, lambda x: x)


def tolist(x):
    if isinstance(x, np.ndarray):
        if np.squeeze(x).ndim < 2:
            x = x.reshape(-1, 1).tolist()
        else:
            x = x.tolist()
    elif isinstance(x, tuple):
        x = list(x)
        for i in range(len(x)):
            x[i] = tolist(x[i])
        x = tuple(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = tolist(x[k])
    return x


def tonumpy(x):
    if isinstance(x, list):
        x = np.array(x)
    elif isinstance(x, tuple):
        x = list(x)
        for i in range(len(x)):
            x[i] = tonumpy(x[i])
        x = tuple(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = tonumpy(x[k])
    return x


def get_function(function_path):
    items = function_path.split('.')
    module = '.'.join(items[:-1])
    method = items[-1]
    f = getattr(__import__(module, fromlist=[method]), items[-1])
    return f


def get_class(class_path):
    items = class_path.spli('.')
    module = __import__(items[0])
    for item in items[1:-1]:
        module = getattr(module, item)
    cls = getattr(module, items[-1])
    return cls


def remove_all_images():
    for file in os.listdir('.'):
        if file.endswith('.png'):
            os.remove(file)


# noinspection PyUnresolvedReferences
class TestMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        remove_all_images()

    @classmethod
    def tearDownClass(cls):
        remove_all_images()

    def compare(self, student_answer, correct_answer, msg, places=7, equiv=None):
        if isinstance(student_answer, scipy.sparse.csr_matrix):
            student_answer = student_answer.todense()
        if isinstance(correct_answer, scipy.sparse.csr_matrix):
            correct_answer = correct_answer.todense()

        if isinstance(correct_answer, (list, tuple)):
            for s, a in zip(student_answer, correct_answer):
                self.compare(s, a, msg, places=places)
        elif isinstance(correct_answer, dict):
            for k in correct_answer.keys():
                self.compare(student_answer[k], correct_answer[k], msg, places=places)
        elif isinstance(correct_answer, np.ndarray):
            if equiv == 'rows_sign' or equiv == 'cols_sign':
                if len(correct_answer.shape) == 1:
                    correct_answer = correct_answer.reshape((1, -1))
                    student_answer = student_answer.reshape((1, -1))
                if equiv == 'cols_sign':
                    correct_answer = correct_answer.T
                    student_answer = student_answer.T

                # match sign of maximum absolute element
                for i in range(len(correct_answer)):
                    idx = np.argmax(np.abs(correct_answer[i, :]))
                    if correct_answer[i, idx] * student_answer[i, idx] < 0:
                        student_answer[i, :] *= -1

                if equiv == 'cols_sign':
                    correct_answer = correct_answer.T
                    student_answer = student_answer.T

            np.testing.assert_almost_equal(student_answer, correct_answer, err_msg=msg, decimal=places)
        elif isinstance(correct_answer, float) and math.isnan(correct_answer):
            self.assertTrue(math.isnan(student_answer), '%s\nstudent_answer != nan' % msg)
        elif isinstance(correct_answer, str):
            self.assertEqual(student_answer, correct_answer, msg)
        else:
            self.assertAlmostEqual(student_answer, correct_answer, msg=msg, places=places)

    def make_text_tests(self):
        import pandas as pd
        df = pd.read_csv('../data/messages.csv')
        messages = df['message'].tolist()
        tests = list()
        path_correct = 'sklearn.feature_extraction.text.CountVectorizer'
        test = {
            'unlock': [path_correct],
            'import': ['re', 'numpy'],
            'params': [[
                {
                    'ret': 'cv',
                    'func': path_correct,
                    'args': [],
                    'kwargs': {'input': 'content', 'encoding': 'utf-8', 'decode_error': 'strict', 'strip_accents': None,
                               'analyzer': 'word', 'stop_words': None, 'lowercase': True, 'max_df': 1.0, 'min_df': 1,
                               'max_features': None, 'binary': False}
                },
                {
                    'obj': 'cv',
                    'func': 'fit',
                    'args': [messages],
                    'kwargs': {}
                },
                {
                    'obj': 'cv',
                    'attr': 'vocabulary_',
                }
            ]] + [[
                {
                    'ret': 'cv',
                    'func': path_correct,
                    'args': [],
                    'kwargs': {'input': 'content', 'encoding': 'utf-8', 'decode_error': 'strict', 'strip_accents': None,
                               'analyzer': 'word', 'stop_words': None, 'lowercase': True, 'max_df': 1.0, 'min_df': 1,
                               'max_features': None, 'binary': False}
                },
                {
                    'obj': 'cv',
                    'func': 'fit',
                    'args': [messages],
                    'kwargs': {}
                },
                {
                    'obj': 'cv',
                    'func': 'transform',
                    'args': [messages],
                    'kwargs': {}
                }
            ]] + [[
                {
                    'ret': 'cv',
                    'func': path_correct,
                    'args': [],
                    'kwargs': {'input': 'content', 'encoding': 'utf-8', 'decode_error': 'strict', 'strip_accents': None,
                               'analyzer': 'word', 'stop_words': None, 'lowercase': True, 'max_df': 1.0, 'min_df': 1,
                               'max_features': None, 'binary': False}
                },
                {
                    'obj': 'cv',
                    'func': 'fit_transform',
                    'args': [messages],
                    'kwargs': {}
                }
            ]] + [[
                {
                    'ret': 'cv',
                    'func': path_correct,
                    'args': [],
                    'kwargs': {'input': 'content', 'encoding': 'utf-8', 'decode_error': 'strict', 'strip_accents': None,
                               'analyzer': 'word', 'stop_words': None, 'lowercase': True, 'max_df': 1.0, 'min_df': 1,
                               'max_features': None, 'binary': False}
                },
                {
                    'obj': 'cv',
                    'func': 'fit',
                    'args': [messages],
                    'kwargs': {}
                },
                {
                    'obj': 'cv',
                    'func': 'get_feature_names',
                    'args': [],
                    'kwargs': {}
                }
            ]]
        }
        tests.append(test)
        return tests

    def make_naive_bayes_tests(self):
        def create_data(num_classes, num_samples, num_features, random_state):
            rng = np.random.RandomState(random_state)
            probs = rng.uniform(size=(num_classes, num_features))
            probs /= probs.sum(axis=1, keepdims=True)
            class_weights = np.concatenate(([0], rng.uniform(size=(num_classes,))))
            class_weights /= class_weights.sum()
            class_samples = np.diff(np.round(np.cumsum(class_weights) * num_samples, 0).astype(int))
            x = list()
            y = list()
            for i in range(num_classes):
                pvals = probs[i]
                for j in range(class_samples[i]):
                    n = rng.randint(1, high=10, size=1)
                    x.append(rng.multinomial(n, pvals, size=1).reshape((-1,)).tolist())
                    y.append(i)
            return x, y

        def sum_to_one(n_):
            x = np.random.uniform(size=(n_,)).reshape((-1,))
            x / sum(x)
            return x.tolist()

        tests = list()
        path_correct = 'sklearn.naive_bayes.MultinomialNB'
        test = {
            'unlock': [path_correct],
            'import': ['numpy', 'math.factorial'],
            'params': [[
                {
                    'ret': 'mnb',
                    'func': path_correct,
                    'args': [],
                    'kwargs': {'alpha': 1.0, 'fit_prior': fit_prior, 'class_prior': class_prior}
                },
                {
                    'obj': 'mnb',
                    'func': 'fit',
                    'args': create_data(num_classes, num_samples, num_features, random_state),
                    'kwargs': {}
                },
                {
                    'obj': 'mnb',
                    'func': 'predict_log_proba',
                    'args': [create_data(num_classes, num_samples, num_features, random_state)[0]],
                    'kwargs': {}
                }
            ] for num_classes in range(2, 4) for num_samples in [100, 1000] for num_features in range(5, 10, 2)
                for random_state in [1, 42]
                for fit_prior, class_prior in [(True, None), (False, sum_to_one(num_classes))]]
        }
        tests.append(test)
        return tests

    def make_cluster_tests(self):
        # weights = np.array([1/3, 1/3, 1/3])
        # means = np.array([
        #     [5, 15],
        #     [15, 15],
        #     [10, 5]
        # ])
        # covars = np.array([
        #     [[3, 0],
        #      [0, 3]],
        #     [[3, 0],
        #      [0, 3]],
        #     [[3, 0],
        #      [0, 3]]
        # ])
        # from sklearn.mixture import GMM
        # gmm = GMM(n_components=3, covariance_type='full')
        # gmm.weights_ = np.array(weights)
        # gmm.means_ = np.array(means)
        # gmm.covars_ = np.array(covars)
        # x = gmm.sample(n_samples=600)
        # np.savez('../data/cluster_k3.npz', x=x)
        with np.load('../data/cluster_k3.npz') as data:
            x = data['x']
        tests = list()
        path_correct = 'sklearn.cluster.KMeans'
        close = np.array([[4.0, 16.0], [16.0, 14.0], [9.0, 4.0]])
        test = {
            'unlock': [path_correct],
            'import': ['numpy', 'scipy.stats.multivariate_normal'],
            'params': [[
                {
                    'ret': 'km',
                    'func': path_correct,
                    'args': [],
                    'kwargs': {'n_clusters': 3, 'init': init, 'random_state': random_state, 'n_init': 1},
                },
                {
                    'obj': 'km',
                    'func': 'fit',
                    'args': [x],
                    'kwargs': {},
                },
                {
                    'obj': 'km',
                    'attr': 'cluster_centers_',
                }
            ] for init, random_state in [(close, 0), ('random', 0), ('random', 1), ('random', 2)]]
        }
        tests.append(test)

        path_correct = 'sklearn.mixture.GMM'
        close_weights = np.array([0.4, 0.3, 0.3])
        close_means = np.array([
            [4.0, 16.0],
            [16.0, 14.0],
            [9.0, 4.0]
        ])
        close_covars = np.array([
            [[2.0, 0.0],
             [0.0, 3.0]],
            [[3.0, 0.0],
             [0.0, 4.0]],
            [[3.0, 0.0],
             [0.0, 2.0]]
        ])
        test = {
            'unlock': [path_correct],
            'import': ['numpy', 'scipy.stats.multivariate_normal'],
            'places': 3,
            'params': [[
                {
                    'ret': 'gmm',
                    'func': path_correct,
                    'args': [],
                    'kwargs': {'n_components': 3, 'covariance_type': 'full', 'init_params': '', 'n_init': 1},
                },
                {
                    'obj': 'gmm',
                    'attr': 'weights_',
                    'value': close_weights,
                },
                {
                    'obj': 'gmm',
                    'attr': 'means_',
                    'value': close_means,
                },
                {
                    'obj': 'gmm',
                    'attr': 'covars_',
                    'value': close_covars
                },
                {
                    'obj': 'gmm',
                    'func': 'fit',
                    'args': [x],
                    'kwargs': {},
                },
                {
                    'obj': 'gmm',
                    'attr': attr,
                },
            ] for attr in ['weights_', 'means_', 'covars_']]
        }
        tests.append(test)

        return tests

    def make_cross_validation_tests(self):
        x = np.concatenate((np.arange(100).reshape((-1, 1)), np.random.uniform(size=(100, 1))), axis=1)
        y = np.concatenate((np.zeros((60,)), np.ones((40,))), axis=0)
        tests = list()
        path_correct = 'sklearn.cross_validation.train_test_split'
        test = {
            'unlock': [path_correct],
            'import': ['numpy'],
            'params': [[{
                'func': path_correct,
                'args': [x, y],
                'kwargs': {'test_size': test_size, 'random_state': random_state},
            }] for test_size in [0.1, 0.4, 0.7] for random_state in [0, 1, 42]]
        }
        tests.append(test)

        path_correct = 'sklearn.cross_validation.KFold'
        test = {
            'unlock': [path_correct],
            'import': ['numpy'],
            'params': [[
                {
                    'ret': 'kf',
                    'func': path_correct,
                    'args': [n],
                    'kwargs': {'n_folds': n_folds, 'shuffle': shuffle, 'random_state': random_state},
                },
                {
                    'ret': 'it',
                    'obj': 'kf',
                    'func': '__iter__',
                    'args': [],
                    'kwargs': {},
                },
                {
                    'obj': 'it',
                    'func': '__next__',
                    'args': [],
                    'kwargs': {},
                }
            ] for n in [10, 50, 100] for n_folds in [2, 3, 5, 10] for shuffle in [False, True]
                for random_state in [0, 1, 42]]
        }
        tests.append(test)

        path_correct = 'sklearn.cross_validation.cross_val_score'
        with np.load('../data/Fig4.23.npz') as data:
            x = data['x']
            y = data['y']
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.cross_validation import KFold
        test = {
            'unlock': [path_correct],
            'import': ['numpy'],
            'params': [[
                {
                    'func': path_correct,
                    'args': [KNeighborsClassifier(n_neighbors=k), x, y],
                    'kwargs': {'cv': KFold(len(x), n_folds=n_folds)},
                }
            ] for k in [1, 10] for n_folds in [3, 10]]
        }
        tests.append(test)

        return tests

    def make_metrics_tests(self):
        tests = list()
        path_correct = 'sklearn.metrics.accuracy_score'
        test = {
            'unlock': [path_correct],
            'import': ['numpy'],
            'params': [[{
                'func': path_correct,
                'args': [np.random.choice(2, size=(n,), p=[p0, 1 - p0]),
                         np.random.choice(2, size=(n,), p=[p0, 1 - p0])],
                'kwargs': {},
            }] for n in [100, 1000] for p0 in np.arange(0.1, 0.6, 0.1)]
        }
        tests.append(test)
        return tests

    def make_classification_tests(self):
        fig4_3 = {
            'x_train': [
                [1, 3, 125000],
                [0, 2, 100000],
                [0, 1, 70000],
                [1, 2, 120000],
                [0, 3, 95000],
                [0, 2, 60000],
                [1, 3, 220000],
                [0, 1, 85000],
                [0, 2, 75000],
                [0, 1, 90000]
            ],
            'y_train': [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            'x_test': [
                [0, 1, 55000],
                [1, 2, 80000],
                [1, 3, 110000],
                [0, 1, 95000],
                [0, 3, 67000]
            ]
        }
        iris_data = {
            'x_train': iris.data,
            'y_train': iris.target,
            'x_test': [
                [0, 0, 2.44, 0],
                [0, 0, 2.46, 1.64],
                [0, 0, 4.94, 1.64],
                [0, 0, 4.96, 1.54],
                [6.94, 0, 4.96, 1.56],
                [6.94, 0, 4.96, 1.74],
                [6.96, 0, 4.96, 1.56],
                [6.96, 0, 4.96, 1.74],
                [5.94, 0, 2.46, 1.76],
                [5.94, 0, 4.84, 1.76],
                [5.96, 0, 2.46, 1.76],
                [5.96, 0, 4.84, 1.76],
                [0, 0, 4.86, 1.76],
            ]
        }
        iris_data2 = {
            'x_train': iris.data[:, :2],
            'y_train': iris.target,
            'x_test': [[x/10, y/10] for x in range(40, 81) for y in range(20, 46)]
        }
        iris_data3 = {
            'x_train': iris.data[:, :2] + 0.1 * np.random.normal(size=(len(iris.data), 2)),
            'y_train': iris.target,
            'x_test': [[x/10, y/10] for x in range(40, 81) for y in range(20, 46)]
        }
        tests = list()
        path_correct = 'sklearn.tree.DecisionTreeClassifier'
        test = {
            'unlock': [path_correct],
            'import': ['numpy'],
            'params': [[
                {
                    'ret': 'tree',
                    'func': path_correct,
                    'args': [],
                    'kwargs': {'criterion': 'gini', 'min_samples_split': min_samples_split, 'random_state': 10},
                },
                {
                    'obj': 'tree',
                    'func': 'fit',
                    'args': [data['x_train'], data['y_train']],
                    'kwargs': {},
                },
                {
                    'obj': 'tree',
                    'func': f,
                    'args': [data['x_test']],
                    'kwargs': {},
                }
            ] for min_samples_split in [2, 5] for data in [fig4_3, iris_data, iris_data2]
                for f in ['predict', 'predict_proba']]
        }
        tests.append(test)

        path_correct = 'sklearn.neighbors.KNeighborsClassifier'
        test = {
            'unlock': [path_correct],
            'import': ['numpy'],
            'params': [[
                {
                    'ret': 'knn',
                    'func': path_correct,
                    'args': [],
                    'kwargs': {'n_neighbors': n_neighbors, 'p': p, 'algorithm': 'brute'},
                    'exception': False,
                },
                {
                    'obj': 'knn',
                    'func': 'fit',
                    'args': [data['x_train'], data['y_train']],
                    'kwargs': {},
                    'exception': False,
                },
                {
                    'obj': 'knn',
                    'func': f,
                    'args': [data['x_test']],
                    'kwargs': {},
                    'exception': False,
                }
            ] for n_neighbors in range(1, 6) for p in range(1, 3) for data in [fig4_3, iris_data, iris_data3]
                for f in ['predict', 'predict_proba']]
        }
        tests.append(test)
        return tests

    def make_prelim_tests(self):
        randi = lambda **kwargs: np.random.randint(100, **kwargs).tolist()
        tests = list()
        for name, rng in [('mean', randi), ('var', randi), ('std', randi)]:
            path_correct = 'numpy.%s' % name
            test = {
                'unlock': [path_correct],
                'import': ['math'],
                'params': [[{
                    'func': path_correct,
                    'args': [rng(size=(n,))],
                    'kwargs': {},
                }] for n in list(range(2, 11)) + [100] for _ in range(10)]
            }
            tests.append(test)

        for name in ['cov']:
            path_correct = 'numpy.%s' % name
            test = {
                'unlock': [path_correct],
                'import': ['math'],
                'params': [[{
                    'func': path_correct,
                    'args': [randi(size=(5, 100))],
                    'kwargs': {},
                }] for _ in range(10)]
            }
            tests.append(test)

        path_correct = 'sklearn.decomposition.PCA'
        with open('../data/pca_data.json', 'r') as fp:
            pca_data = np.array(json.load(fp))

        test = {
            'unlock': [path_correct],
            'import': ['numpy'],
            'params':
            [[
                {
                    'ret': 'pca',
                    'func': path_correct,
                    'args': [],
                    'kwargs': {'n_components': n_components},
                    'exception': False,
                },
                {
                    'obj': 'pca',
                    'func': 'fit',
                    'args': [x],
                    'kwargs': {},
                    'exception': False,
                },
                {
                    'obj': 'pca',
                    'attr': attr,
                    'equiv': equiv,
                }
            ] for n_components in [1, 2, None] for x in [iris.data, pca_data]
                for attr, equiv in zip(['components_', 'explained_variance_ratio_', 'mean_', 'n_components_'],
                                       ['rows_sign', None, None, None])
            ] + [[
                {
                    'ret': 'pca',
                    'func': path_correct,
                    'args': [],
                    'kwargs': {'n_components': n_components},
                    'exception': False,
                },
                {
                    'obj': 'pca',
                    'func': 'fit',
                    'args': [x],
                    'kwargs': {},
                    'exception': False,
                },
                {
                    'obj': 'pca',
                    'func': 'transform',
                    'args': [x],
                    'kwargs': {},
                    'exception': False,
                    'equiv': 'cols_sign',
                }
            ] for n_components in [1, 2, None] for x in [iris.data, pca_data]]
        }
        tests.append(test)

        path_correct = 'numpy.linalg.lstsq'
        test = {
            'unlock': [path_correct],
            'import': ['numpy.linalg.inv', 'numpy.array', 'numpy.linalg.svd', 'numpy.diag'],
            'params': [[{
                'func': path_correct,
                'args': [randi(size=(5, 3)), randi(size=(5, 1))],
                'kwargs': {},
            }] for _ in range(10)]
        }
        tests.append(test)



        return tests

    def make_statistics_tests(self):
        randi = lambda **kwargs: np.random.randint(100, **kwargs).tolist()
        nanrandi = lambda **kwargs: [x if x < 50 else np.nan for x in np.random.randint(100, **kwargs)]
        tests = list()
        for name, rng in [('mean', randi), ('median', randi), ('var', randi), ('std', randi),
                          ('nanmean', nanrandi), ('nanmedian', nanrandi), ('nanvar', nanrandi), ('nanstd', nanrandi)]:
            path_correct = 'numpy.%s' % name
            test = {
                'unlock': [path_correct],
                'import': ['math'],
                'params': [[{
                    'func': path_correct,
                    'args': [rng(size=(n,))],
                    'kwargs': {},
                }] for n in list(range(2, 11)) + [100] for _ in range(10)]
            }
            tests.append(test)

        for name, rng in [('percentile', randi), ('nanpercentile', nanrandi)]:
            path_correct = 'numpy.%s' % name
            test = {
                'unlock': [path_correct],
                'import': ['math'],
                'params': [[{
                    'func': path_correct,
                    'args': [randi(size=(100,)), q],
                    'kwargs': {},
                }] for q in [0, 10.5, 25.2, 33, 67, 74.5, 100]]
            }
            tests.append(test)

        for name in ['cov', 'corrcoef']:
            path_correct = 'numpy.%s' % name
            test = {
                'unlock': [path_correct],
                'import': ['math'],
                'params': [[{
                    'func': path_correct,
                    'args': [randi(size=(5, 100))],
                    'kwargs': {},
                }] for _ in range(10)]
            }
            tests.append(test)

        path_correct = 'numpy.histogram'
        tests.append({
            'unlock': [path_correct, 'matplotlib.pyplot.hist'],
            'import': ['math'],
            'params': [[{
                'func': path_correct,
                'args': [randi(size=(1000,))],
                'kwargs': {'bins': 10},
            }] for _ in range(10)]
        })
        path_correct = 'numpy.histogram2d'
        tests.append({
            'unlock': [path_correct, 'matplotlib.pyplot.hist2'],
            'import': ['math'],
            'params': [[{
                'func': path_correct,
                'args': [randi(size=(1000,)), randi(size=(1000,))],
                'kwargs': {'bins': 10},
            }] for _ in range(10)]
        })
        return tests

    def make_distance_tests(self):
        from sklearn.preprocessing import normalize
        rand_01 = lambda **kwargs: np.random.randint(2, **kwargs).tolist()
        rand_bool = lambda **kwargs: [x > 0 for x in normal(**kwargs)]
        sum_to_one = lambda **kwargs: normalize(np.random.uniform(**kwargs), axis=0, norm='l1').reshape(-1).tolist()
        normal = lambda **kwargs: np.random.normal(**kwargs).tolist()
        tests = list()
        for name, rng in [('euclidean', normal), ('cityblock', normal), ('chebyshev', normal), ('correlation', normal),
                          ('cosine', normal), ('hamming', rand_01), ('jaccard', rand_bool)]:
            path_correct = 'scipy.spatial.distance.%s' % name
            test = {
                'unlock': [path_correct],
                'import': ['math'],
                'params': [[{
                    'func': path_correct,
                    'args': [rng(size=(n,)), rng(size=(n,))],
                    'kwargs': {},
                }] for n in range(2, 11)]
            }
            tests.append(test)

        path_correct = 'scipy.spatial.distance.mahalanobis'
        tests.append({
            'unlock': [path_correct],
            'import': ['math'],
            'params': [[{
                'func': path_correct,
                'args': [normal(size=(2,)), normal(size=(2,)), [[2/3, -1/3], [-1/3, 2/3]]],
                'kwargs': {},
            }] for _ in range(10)],
        })
        path_correct = 'scipy.spatial.distance.minkowski'
        tests.append({
            'unlock': [path_correct],
            'import': ['math'],
            'params': [[{
                'func': path_correct,
                'args': [normal(size=(n,)), normal(size=(n,)), p],
                'kwargs': {},
            }] for n in range(1, 11) for p in range(3, 5)],
        })
        path_correct = 'scipy.spatial.distance.wminkowski'
        tests.append({
            'unlock': [path_correct],
            'import': ['math'],
            'params': [[{
                'func': path_correct,
                'args': [normal(size=(n,)), normal(size=(n,)), p, sum_to_one(size=(n, 1))],
                'kwargs': {},
                }] for n in range(1, 11) for p in range(3, 5)
            ],
        })
        path_correct = 'scipy.spatial.distance.pdist'
        tests.append({
            'unlock': [path_correct],
            'import': ['math'],
            'params': [[{
                'func': path_correct,
                'args': [normal(size=(10, 5))],
                'kwargs': {'metric': metric},
            }] for metric in ['euclidean', 'cityblock', 'cosine']]
        })
        path_correct = 'scipy.spatial.distance.cdist'
        tests.append({
            'unlock': [path_correct],
            'import': ['math'],
            'params': [[{
                'func': path_correct,
                'args': [normal(size=(10, 5)), normal(size=(10, 5))],
                'kwargs': {'metric': metric},
            }] for metric in ['euclidean', 'cityblock', 'cosine']]
        })
        path_correct = 'scipy.spatial.distance.squareform'
        tests.append({
            'unlock': [path_correct],
            'import': ['math'],
            'params': [[{
                'func': path_correct,
                'args': [normal(size=(45,))],
                'kwargs': {},
            }] for _ in range(10)]
        })
        return tests

    def check_imports(self, func, allowed_imports):
        import inspect
        file_path = inspect.getfile(func)
        with open(file_path) as fp:
            for line in fp.readlines():
                if re.match(r"^\s*(import|from)", line):
                    legal_import = False
                    for allowed_import in allowed_imports:
                        terms = allowed_import.split('.')
                        if re.match("\s*from %s import %s$" % ('.'.join(terms[:-1]), terms[-1]), line) \
                                or re.match("\s*import %s( as .+)?$" % allowed_import, line):
                            legal_import = True
                            break
                    self.assertTrue(legal_import, '"%s" is illegal import. Only the following imports are legal: %s' %
                                    (line, allowed_imports))

    def check_images(self, images):
        from scipy.misc import imread
        import requests
        from io import BytesIO
        for png_student in images:
            with self.subTest(test=png_student):
                url_correct = 'http://www.cs.appstate.edu/~rmp/cs5710/tests/%s' % png_student
                img_correct = imread(BytesIO(requests.get(url_correct).content))
                img_student = imread(png_student)
                self.assertEqual(img_correct.shape[0], img_student.shape[0], 'image heights do not match.')
                self.assertEqual(img_correct.shape[1], img_student.shape[1], 'image widths do not match.')
                # img_correct = np.dot(img_correct[:, :, :3], [0.299, 0.587, 0.114])
                # img_student = np.dot(img_student[:, :, :3], [0.299, 0.587, 0.114])
                # if png_student == 'Fig3.9.png':
                #     idx = np.where(img_student != img_correct)
                #     print(idx)
                #     print(img_student[idx])
                #     print(img_correct[idx])
                img_correct = img_correct[:, :, :3] / 255
                img_student = img_student[:, :, :3] / 255
                np.testing.assert_almost_equal(img_student, img_correct, decimal=2,
                                               err_msg='%s fails, compare to %s' % (png_student, url_correct))
                print('%s passes!' % png_student)

    def check_visualization(self, allowed_imports):
        student = 'cs5710.apps.visualization.main'
        f_student = get_function(student)
        with self.subTest(test='visualization'):
            self.check_imports(f_student, allowed_imports)
            f_student()
            images = ['figs/Fig3.%d.png' % n for n in [9, 11, 13, 15, 17, 19]]
            images += ['figs/Fig3.14-%d.png' % n for n in range(4)]
            images += ['figs/Fig3.16-%d%d.png' % (i, j) for i in range(4) for j in range(4) if i != j]
            self.check_images(images)

    def check_self_similarity(self, allowed_imports):
        student = 'cs5710.apps.self_similarity.main'
        f_student = get_function(student)
        with self.subTest(test='self_similarity'):
            self.check_imports(f_student, allowed_imports)
            q = 'Scary Monsters and Nice Sprites'
            f_student(q)
            images = ['figs/%s_%s.png' % (q, t) for t in ['timbre', 'pitch']]
            self.check_images(images)

    def check_overfitting(self, allowed_imports):
        student = 'cs5710.apps.overfitting.main'
        f_student = get_function(student)
        with self.subTest(test='overfitting'):
            self.check_imports(f_student, allowed_imports)
            data_file = 'Fig4.23'
            f_student(data_file)
            images = ['figs/Fig4.22.png'] + ['figs/Fig4.23_%s.png' % t for t in ['tree', 'knn']]
            self.check_images(images)

    def check_bayes_error(self, allowed_imports):
        student = 'cs5710.apps.bayes_error.UniformGMMClassifier'
        uniform_gmm = get_function(student)
        clf = uniform_gmm(gmm_n_components=2)
        with self.subTest(test='bayes_error.UniformGMMClassifier.__init__'):
            self.check_imports(uniform_gmm, allowed_imports)
            self.assertEqual(clf.gmm_n_components, 2, 'clf.gmm_n_components is not set correctly')
            self.assertIsNone(clf.uniform, 'clf.uniform should be None')
            self.assertIsNone(clf.class_weights, 'clf.class_weights should be None')
            self.assertEqual(clf.gmm.n_components, 2, 'clf.gmm.n_components is not set correctly')
            self.assertEqual(clf.gmm.covariance_type, 'full', 'clf.gmm.covariance_type should be \'full\'')
            self.assertEqual(clf.gmm.random_state, 1, 'clf.gmm.random_state should be 1')
            self.assertIsNone(clf.gmm.thresh, 'clf.gmm.thresh shold be None')
            self.assertEqual(clf.gmm.tol, 0.001, 'clf.gmm.tol should be 0.001')
            self.assertEqual(clf.gmm.min_covar, 0.001, 'clf.gmm.min_covar should be 0.001')
            self.assertEqual(clf.gmm.n_init, 1, 'clf.gmm.n_init should be 1')
            self.assertEqual(clf.gmm.params, 'wmc', 'clf.gmm.params should be \'wmc\'')

        clf = uniform_gmm(gmm_n_components=3)
        with self.subTest(test='bayes_error.UniformGMMClassifier.get_params'):
            params = clf.get_params()
            self.assertDictEqual(params, {'gmm_n_components': 3}, 'clf.get_params() returns the wrong thing')

        with np.load('../data/Fig4.23.npz') as data:
            x = data['x']
            y = data['y']
        clf.fit(x, y)

        with self.subTest(test='bayes_error.UniformGMMClassifier.fit'):
            self.assertAlmostEqual(clf.class_weights, [0.6, 0.4])
            np.testing.assert_almost_equal(clf.uniform, ([.00077285, .00720204], [19.9870281, 19.99981849]))
            np.testing.assert_almost_equal(clf.gmm.weights_, [.33529364, .33249472, .33221164])
            np.testing.assert_almost_equal(clf.gmm.means_, [[14.88384397, 15.0135191],
                                                            [9.82588486, 5.13209943],
                                                            [5.05326623, 15.02481669]])
            np.testing.assert_almost_equal(clf.gmm.covars_, [[[4.08920928, -0.12689568],
                                                              [-0.12689568, 4.63437171]],
                                                             [[3.46067856, -0.08910985],
                                                              [-0.08910985, 3.85919608]],
                                                             [[4.11361846, -0.02992279],
                                                              [-0.02992279, 3.86734937]]])

        x_test = [[10, 10], [5, 15], [10, 5], [16, 18], [17, 18], [10, -0.1], [10, 0.1], [5, 19.9], [5, 20.1]]
        x_likelihood = np.array([[0.00250264, 0.00072872], [0.00250264, 0.01325091], [0.00250264, 0.01439084],
                                 [0.00250264, 0.00393039], [0.00250264, 0.0025939], [0., 0.00041726],
                                 [0.00250264, 0.00054436], [0.00250264, 0.00061363], [0., 0.00047443]])
        with self.subTest(test='bayes_error.UniformGMMClassifier.predict_likelihood'):
            np.testing.assert_almost_equal(clf.predict_likelihood(x_test), x_likelihood)

        x_joint = np.array([[0.00150159, 0.00029149], [0.00150159, 0.00530036], [0.00150159, 0.00575634],
                            [0.00150159, 0.00157216], [0.00150159, 0.00103756], [0., 0.00016691],
                            [0.00150159, 0.00021774], [0.00150159, 0.00024545], [0., 0.00018977]])
        with self.subTest(test='bayes_error.UniformGMMClassifier.predict_joint'):
            np.testing.assert_almost_equal(clf.predict_joint(x_test), x_joint)

        x_total = np.array([[0.00179307], [0.00680195], [0.00725792], [0.00307374], [0.00253915], [0.00016691],
                            [0.00171933], [0.00174704], [0.00018977]])
        with self.subTest(test='bayes_error.UniformGMMClassifier.predict_total_probability'):
            np.testing.assert_almost_equal(clf.predict_total_probability(x_test), x_total)

        x_proba = np.array([[0.83743708, 0.16256292], [0.22075816, 0.77924184], [0.20688923, 0.79311077],
                            [0.48852021, 0.51147979], [0.59137402, 0.40862598], [0., 1.], [0.87335491, 0.12664509],
                            [0.85950335, 0.14049665], [0., 1.]])
        with self.subTest(test='bayes_error.UniformGMMClassifier.predict_proba'):
            np.testing.assert_almost_equal(clf.predict_proba(x_test), x_proba)

        x_predict = np.array([0.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  1.])
        with self.subTest(test='bayes_error.UniformGMMClassifier.predict'):
            np.testing.assert_almost_equal(clf.predict(x_test), x_predict)

        with self.subTest(test='bayes_error.UniformGMMClassifier.main'):
            student = 'cs5710.apps.bayes_error.main'
            f = get_function(student)
            accuracy_train, accuracy_test, accuracy_cv, accuracy_ideal, accuracy_bayes = f(1)
            self.assertAlmostEqual(accuracy_train, 0.74767, places=5)
            self.assertAlmostEqual(accuracy_test, 0.77000, places=5)
            self.assertAlmostEqual(accuracy_cv, 0.74200, places=5)
            self.assertAlmostEqual(accuracy_ideal, 0.74733, places=5)
            self.assertAlmostEqual(accuracy_bayes, 0.74822, places=5)
            self.check_images(['figs/bayes_error.png'])

    def check_against_standard(self, unlocked_imports, tests):
        import warnings
        additional_imports = []
        for test in tests:
            unlock = test['unlock']
            test_imports = test['import']
            test_method = unlock[0]
            with self.subTest(test=test_method):
                # print('testing %s' % test_method)
                results = {'student': {}, 'correct': {}}
                final = {}
                for sequence in test['params']:
                    # print(sequence)
                    for r in results.keys():
                        res = None
                        for param in sequence:
                            if all([s in param for s in ['func', 'args', 'kwargs']]):
                                # ret = obj.func(*args, **kwargs)
                                if 'obj' in param:
                                    f = getattr(results[r][param['obj']], param['func'])
                                else:
                                    f_path = '%s%s' % ('' if r == 'correct' else 'cs5710.', param['func'])
                                    f = get_function(f_path)

                                args = param['args']
                                kwargs = param['kwargs']
                                if r == 'student':
                                    if hasattr(f, '__call__'):
                                        self.check_imports(f, unlocked_imports + test_imports)
                                    args = tolist(deepcopy(args))
                                    kwargs = tolist(deepcopy(kwargs))

                                # noinspection PyBroadException
                                try:
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
                                        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
                                        warnings.filterwarnings('ignore', r'Mean of empty slice')
                                        if 'ret' in param:
                                            res = f(*args, **kwargs)
                                            results[r][param['ret']] = res
                                        else:
                                            res = f(*args, **kwargs)
                                except Exception as res:
                                    self.assertIn('exception', param, 'Unexpected Exception: %s' % sequence)
                                    self.assertTrue(param['exception'], 'Unexpected Exception: %s' % sequence)
                            elif all([s in param for s in ['obj', 'attr', 'value']]):
                                setattr(results[r][param['obj']], param['attr'], param['value'])
                            elif all([s in param for s in ['obj', 'attr']]):
                                res = getattr(results[r][param['obj']], param['attr'])
                                if 'ret' in param:
                                    results[r][param['ret']] = res
                            else:
                                raise ValueError('Can\'t parse params: %s' % param)
                        final[r] = res
                    # check results
                    if isinstance(final['correct'], Exception):
                        self.assertIsInstance(final['student'], Exception)
                    else:
                        final['student'] = tonumpy(final['student'])
                        places = 6
                        if 'places' in test:
                            places = test['places']
                        equiv = None
                        if 'equiv' in sequence[-1]:
                            equiv = sequence[-1]['equiv']

                        member = sequence[-1].get('func', None) or sequence[-1].get('attr', None)
                        msg = 'Error checking %s.%s' % (test_method, member if member != test_method else '')
                        self.compare(final['student'], final['correct'], msg, places=places, equiv=equiv)

                    # student_answer = None
                    # correct_answer = None
                    # o_correct = None
                    # o_student = None
                    # for param in sequence:
                    #     if o_correct is not None:
                    #         f_correct = getattr(o_correct, param['func'])
                    #         f_student = getattr(o_student, param['func'])
                    #     else:
                    #         f_correct = get_function(param['func'])
                    #         f_student = get_function('cs5710.%s' % param['func'])
                    #
                    #     if hasattr(f_student, '__call__'):
                    #         self.check_imports(f_student, allowed_imports)
                    #         args = param['args']
                    #         kwargs = param['kwargs']
                    #         student_args = tolist(deepcopy(args))
                    #         student_kwargs = tolist(deepcopy(kwargs))
                    #
                    #         # noinspection PyBroadException
                    #         try:
                    #             # check if 'golden' raises exception
                    #             with warnings.catch_warnings():
                    #                 warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice')
                    #                 warnings.filterwarnings('ignore', r'All-NaN slice encountered')
                    #                 warnings.filterwarnings('ignore', r'Mean of empty slice')
                    #                 correct_answer = f_correct(*args, **kwargs)
                    #         except Exception as e:
                    #             if param['exception']:
                    #                 # if it does, make sure 'method' raises an exception too
                    #                 with self.assertRaises(Exception):
                    #                     f_student(*args, **kwargs)
                    #             else:
                    #                 print('Target function %s unexpectedly raised exception!' % test_method)
                    #                 raise e
                    #         else:
                    #             # otherwise compare their return values
                    #             student_answer = f_student(*student_args, **student_kwargs)
                    #             if (o_correct is None and inspect.isclass(f_correct)) or
                    #                 param.get('overwrite', False):
                    #                 o_correct = correct_answer
                    #                 o_student = student_answer
                    #     else:
                    #         student_answer = f_student
                    #         correct_answer = f_correct
                    #         if 'value' in param:
                    #             setattr(o_correct, param['func'], param['value'])
                    #             setattr(o_student, param['func'], param['value'])
                    #
                    #         if (o_correct is None and inspect.isclass(f_correct)) or param.get('overwrite', False):
                    #             o_correct = correct_answer
                    #             o_student = student_answer
                    #
                    #
                    # student_answer = tonumpy(student_answer)
                    # self.compare(student_answer, correct_answer, str(sequence))

                for m in unlock:
                    print('%s unlocked!' % m)
                additional_imports.extend(unlock)
        return additional_imports

    def test_all(self):
        unlocked_imports = []
        unlocked_imports += self.check_against_standard(unlocked_imports, self.make_prelim_tests())
        # unlocked_imports += self.check_against_standard(unlocked_imports, self.make_distance_tests())
        # self.check_self_similarity(['spotipy', 'json', 'matplotlib.pyplot', 'numpy.exp', 'os'] + additional_imports)
        # unlocked_imports += self.check_against_standard(unlocked_imports, self.make_statistics_tests())
        # self.check_visualization(['matplotlib.pyplot', 'mpl_toolkits.mplot3d.Axes3D', 'scipy.misc.imread'] +
        #                          additional_imports)
        # unlocked_imports += self.check_against_standard(unlocked_imports, self.make_classification_tests())
        # unlocked_imports += self.check_against_standard(unlocked_imports, self.make_metrics_tests())
        # unlocked_imports += self.check_against_standard(unlocked_imports, self.make_cross_validation_tests())
        # self.check_overfitting(['matplotlib.pyplot', 'numpy'] + additional_imports)
        # unlocked_imports += self.check_against_standard(unlocked_imports, self.make_cluster_tests())
        # self.check_bayes_error(['matplotlib.pyplot', 'numpy'] + additional_imports)
        # unlocked_imports += self.check_against_standard(unlocked_imports, self.make_naive_bayes_tests())
        # unlocked_imports += self.check_against_standard(unlocked_imports, self.make_text_tests())


if __name__ == '__main__':
    unittest.main()
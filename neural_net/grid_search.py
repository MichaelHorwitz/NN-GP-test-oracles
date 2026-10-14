import argparse
import os
import sys
import logging
from typing import Any

# Ensure project root is on sys.path so imports like `generation_methods` work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

from generation_methods.RandomSampler import RandomSampler
from generation_methods.LinearGeneration import LinearGeneration
from gp.GP import GeneticProgram
from gp.Node import AddNode, SubtractionNode, MultiplicationNode, DivisionNode, GreaterThanNode, LessThanNode, NotEqualNode, IfNode
from software_under_test.TriangleClassifier import TriangleClassifier
from software_under_test.CreditCheck import CreditCheck
from software_under_test.Collatz import Collatz


class KerasSklearnWrapper(BaseEstimator, ClassifierMixin):
    """A lightweight scikit-learn wrapper around a Keras Sequential model.

    Supports both classification (softmax / binary) and regression (single output).
    """
    def __init__(self,
                 num_outputs: int,
                 input_dim: int,
                 hidden_units: int = 8,
                 hidden_layers: int = 2,
                 activation: str = 'relu',
                 learning_rate: float = 1e-3,
                 optimizer: str = 'adam',
                 epochs: int = 10,
                 batch_size: int = 32,
                 random_state: int = 42):
        self.num_outputs = num_outputs
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._model = None

    def _build_model(self):
        model = Sequential()
        # explicit Input layer to avoid Dense(input_shape=...) warning
        model.add(Input(shape=(self.input_dim,)))
        model.add(Dense(self.hidden_units, activation=self.activation))
        # hidden layers
        for _ in range(self.hidden_layers - 1):
            model.add(Dense(self.hidden_units, activation=self.activation))

        # final layer
        if self.num_outputs == -1:
            # regression (single output)
            model.add(Dense(1, activation='linear'))
            loss = 'mse'
        elif self.num_outputs == 2:
            model.add(Dense(2, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(self.num_outputs, activation='softmax'))
            loss = 'categorical_crossentropy'

        opt = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        return model

    def fit(self, X, y, **kwargs):
        # Ensure numpy arrays
        X = np.asarray(X, dtype='float32')
        y = np.asarray(y, dtype='float32')
        self._model = self._build_model()
        # For regression targets, reshape if needed
        if self.num_outputs == -1 and y.ndim == 2 and y.shape[1] == 1:
            y_fit = y.ravel()
        else:
            y_fit = y

        # Keras expects 2D y for classification one-hot; leave as-is
        self._model.fit(X, y_fit, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X: Any):
        X = np.asarray(X, dtype='float32')
        probs = self._model.predict(X, verbose=0)
        if self.num_outputs == -1:
            return probs.ravel()
        # If output is one-hot / probabilities, return class indices
        if probs.ndim == 2 and probs.shape[1] > 1:
            return np.argmax(probs, axis=1)
        # binary (2 outputs with sigmoid)
        if probs.ndim == 2 and probs.shape[1] == 1:
            return (probs.ravel() > 0.5).astype(int)
        return probs

    def score(self, X, y):
        # Default scoring used by GridSearchCV when scoring is None
        X = np.asarray(X, dtype='float32')
        y = np.asarray(y, dtype='float32')
        if self.num_outputs == -1:
            preds = self.predict(X)
            mse = mean_squared_error(y.ravel(), preds.ravel())
            return -mse
        else:
            preds = self.predict(X)
            # y might be one-hot encoded
            if y.ndim == 2 and y.shape[1] > 1:
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y.ravel()
            return accuracy_score(y_true, preds)


def prepare_data(sut_obj, generator: str = 'random', test_suite_size: int = 256):
    if generator == 'random':
        rng = RandomSampler(sut_obj, test_suite_size=test_suite_size)
        rng.generate_test_suite()
        tests = rng.test_suite
    elif generator == 'linear':
        linear = LinearGeneration(sut_obj, test_suite_size=test_suite_size)
        linear.generate_test_suite()
        tests = linear.test_suite
    elif generator == 'gp':
        prim_list = [AddNode, SubtractionNode, MultiplicationNode, DivisionNode, GreaterThanNode, LessThanNode, NotEqualNode, IfNode]
        term_list = [lambda: None]  # placeholder; GeneticProgram handles terminals internally in this codebase
        gp = GeneticProgram(prim_list, term_list, sut_obj, pop_size=50, test_suite_size=test_suite_size)
        gp.train()
        tests = gp.best_test_suite
    else:
        raise ValueError(f"Unknown generator: {generator}")

    X, y = sut_obj.program_input_to_pandas(tests)
    # convert pandas to numpy for GridSearch
    X_np = X.to_numpy().astype('float32') if hasattr(X, 'to_numpy') else np.asarray(X, dtype='float32')
    y_np = y.to_numpy().astype('float32') if hasattr(y, 'to_numpy') else np.asarray(y, dtype='float32')
    return X_np, y_np



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='neural_net/grid_search_results')
    parser.add_argument('--quick', action='store_true', help='Run a reduced quick grid for faster runs')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    suts = [TriangleClassifier, CreditCheck, Collatz]
    generators = ['random', 'linear', 'gp']
    param_grid = {
        'activation': ['relu', 'tanh'],
        'hidden_units': [4, 8, 16],
        'hidden_layers': [1, 2, 4],
        'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
        'epochs': [3,5,9],
        'batch_size': [64]
    }

    os.makedirs(args.out_dir, exist_ok=True)

    for sut_cls in suts:
        for gen in generators:
            sut = sut_cls()
            print(f"\n=== Running GridSearch for SUT: {sut.__class__.__name__}, generator: {gen} ===")
            X, y = prepare_data(sut, generator=gen, test_suite_size=256)
            num_outputs = sut.get_num_outputs()
            input_dim = X.shape[1] if X.ndim > 1 else 1
            wrapper = KerasSklearnWrapper(num_outputs=num_outputs, input_dim=input_dim)

            gs = GridSearchCV(wrapper, param_grid, cv=2, verbose=2)
            gs.fit(X, y)

            out_file = os.path.join(args.out_dir, f"{sut.__class__.__name__}_{gen}_grid_results.csv")
            results_df = pd.DataFrame(gs.cv_results_)
            results_df.to_csv(out_file, index=False)
            print(f"Saved GridSearch results to {out_file}")
            print("Best params:", gs.best_params_)
            print("Best score:", gs.best_score_)


if __name__ == '__main__':
    main()

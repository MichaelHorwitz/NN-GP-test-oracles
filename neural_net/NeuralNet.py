import logging
from typing import Any, List, Optional
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam, SGD, RMSprop
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from software_under_test.SoftwareUnderTest import SoftwareUnderTest


class NeuralNetwork:
    def __init__(self, 
                 program_inputs: List[Any], 
                 software_under_test: SoftwareUnderTest, 
                 test_size: float=0.15,
                 num_neurons: int = 32,
                 hidden_layer_sizes: Optional[List[int]] = None,
                 activation: str = "relu",
                 epochs: int = 20,
                 batch_size: int = 64,
                 optimizer: Any = "adam",
                 learning_rate: Optional[float] = None,
                 loss: Optional[str] = None,
                 metrics: Optional[List[str]] = None,
                 verbose: int = 0,
                 dropout: Optional[float] = None,
                 ):
        self.program_inputs = program_inputs
        self.software_under_test = software_under_test
        self.training_data: tuple[pd.DataFrame, pd.DataFrame]
        self.test_size = test_size
        self.num_neurons = num_neurons
        # Backwards compatible default: two hidden layers using num_neurons
        if hidden_layer_sizes is None:
            self.hidden_layer_sizes = [self.num_neurons, self.num_neurons]
        else:
            self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose
        self.dropout = dropout
        self.logger = logging.getLogger(f"{__name__}.NeuralNetwork")
    def _to_numpy(self, obj, name: str):
        # Convert pandas Series/DataFrame or generic iterables to numeric numpy arrays
        try:
            if hasattr(obj, 'to_numpy'):
                arr = obj.to_numpy()
            else:
                arr = np.asarray(obj)
        except Exception:
            arr = np.asarray(obj)
        # If object dtype, try to coerce elementwise
        if arr.dtype == object:
            try:
                # First convert to float64 to handle large integers
                arr = arr.astype('float64')
            except Exception:
                # Try to convert sequences (e.g., one-hot lists) into 2D float array
                try:
                    arr = np.stack([np.asarray(x, dtype='float64') for x in arr])
                except Exception as e:
                    self.logger.error(f"Failed to convert {name} to numeric array: {e}")
                    raise
        else:
            # First convert to float64 to handle large integers without overflow
            arr = arr.astype('float64')
        
        # Clip values to float32 range to prevent overflow
        # float32 range: approximately -3.4e38 to 3.4e38
        float32_max = np.finfo(np.float32).max
        float32_min = np.finfo(np.float32).min
        arr = np.clip(arr, float32_min, float32_max)
        
        # Now safely convert to float32
        arr = arr.astype('float32')
        return arr
    def genenrate_training_data(self):
        self.training_data = self.software_under_test.program_input_to_pandas(self.program_inputs)
        self.logger.debug(f"{self.training_data = }")
    def train(self, validation_data: Optional[tuple] = None):
        self.logger.debug(f"Training with {self.training_data[0]=} and {self.training_data[1]=}")
        # Keep pandas DataFrames for logging, convert to numeric arrays for Keras
        X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
            self.training_data[0], self.training_data[1], test_size=self.test_size
        )

        # store pandas copies for logging/debug
        self.X_train_df = X_train_df
        self.X_test_df = X_test_df
        self.y_train_df = y_train_df
        self.y_test_df = y_test_df

        # Convert to numeric numpy arrays for Keras
        self.X_train = self._to_numpy(self.X_train_df, 'X_train')
        self.X_test = self._to_numpy(self.X_test_df, 'X_test')
        self.y_train = self._to_numpy(self.y_train_df, 'y_train')
        self.y_test = self._to_numpy(self.y_test_df, 'y_test')

        # If regression target is 1D, reshape to (n,1) for Keras
        num_outputs = self.software_under_test.get_num_outputs()
        if num_outputs == -1:
            if self.y_train.ndim == 1:
                self.y_train = self.y_train.reshape(-1, 1)
            if self.y_test.ndim == 1:
                self.y_test = self.y_test.reshape(-1, 1)
        # Log class distribution from pandas view when possible
        try:
            class_counts = self.y_train_df.value_counts() if hasattr(self.y_train_df, "value_counts") else pd.Series(self.y_train_df).value_counts()
            self.logger.debug(f"Class distribution in y_train:\n{class_counts}")
        except Exception:
            self.logger.debug(f"y_train shape: {self.y_train.shape}")
        num_outputs = self.software_under_test.get_num_outputs()
        if num_outputs == -1:
            final_layer = Dense(1, activation='linear')
            default_loss = "MSE"
        elif num_outputs == 2:
            final_layer = Dense(2, activation='sigmoid')
            default_loss = "binary_crossentropy"
        else:
            final_layer = Dense(num_outputs, activation='softmax')
            default_loss = "categorical_crossentropy"
        loss_to_use = self.loss if self.loss is not None else default_loss
        input_dim = self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1
        layers = [Input(shape=(input_dim,))]
        for size in self.hidden_layer_sizes:
            layers.append(Dense(size, activation=self.activation))
            if self.dropout is not None and self.dropout > 0:
                layers.append(Dropout(self.dropout))
        layers.append(final_layer)
        self.model = Sequential(layers)
        # Log a small sample using pandas views when available
        try:
            self.logger.debug(f"X_train (head):\n{self.X_train_df.head()}")
            self.logger.debug(f"y_train (head):\n{self.y_train_df.head()}")
        except Exception:
            self.logger.debug(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        # Prepare optimizer instance if a learning rate override is provided
        optimizer_to_use = self.optimizer
        if isinstance(self.optimizer, str) and self.learning_rate is not None:
            opt_name = self.optimizer.lower()
            if opt_name == 'adam':
                optimizer_to_use = Adam(learning_rate=self.learning_rate)
            elif opt_name in ('sgd', 'gradientdescent'):
                optimizer_to_use = SGD(learning_rate=self.learning_rate)
            elif opt_name == 'rmsprop':
                optimizer_to_use = RMSprop(learning_rate=self.learning_rate)
            else:
                optimizer_to_use = self.optimizer

        metrics_to_use = self.metrics
        if metrics_to_use is None:
            if num_outputs >= 2:
                metrics_to_use = ["accuracy"]
            elif num_outputs == -1:
                metrics_to_use = ["mse"]

        self.model.compile(optimizer=optimizer_to_use,  # type: ignore
                           loss=loss_to_use,
                           metrics=metrics_to_use)

        # Run training once and capture the returned History object
        # If caller provided an explicit validation dataset (e.g., tailored suite), use it
        if validation_data is not None:
            try:
                val_X, val_Y = validation_data
                val_X_np = self._to_numpy(val_X, 'validation_X')
                val_Y_np = self._to_numpy(val_Y, 'validation_Y')
                validation_tuple = (val_X_np, val_Y_np)
            except Exception:
                # Fallback to held-out test split
                validation_tuple = (self.X_test, self.y_test)
        else:
            validation_tuple = (self.X_test, self.y_test)

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=validation_tuple,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose # type: ignore
        )

        # store history dictionary for convenience
        self.last_history = history.history if history is not None else None
        return history
    def evaluate_model(self, validation_X = None, validation_Y = None):
        # By default evaluate on the held-out test set
        if validation_X is None and validation_Y is None:
            self.logger.info("using test set")
            X = self.X_test
            Y = self.y_test
        elif validation_X is None or validation_Y is None:
            raise ValueError("Both validation_X and validation_Y must be provided together")
        else:
            self.logger.info("Using provided validation set")
            X = self._to_numpy(validation_X, 'validation_X')
            Y = self._to_numpy(validation_Y, 'validation_Y')
        self.loss, self.acc = self.model.evaluate(X, Y, verbose=0) # type: ignore
        print(f"Test loss: {self.loss:.4f},  Test acc or mse: {self.acc:.4f}")
    def evaluate_as_oracle(self, x_data, y_data):
        y_pred_probs = self.model.predict(x_data)
        # Get predicted classes (e.g., for softmax output)
        y_pred = np.argmax(y_pred_probs, axis=1)
        # Get true classes
        y_true = np.argmax(y_data, axis=1)
        # Count mismatches
        num_diffs = np.sum(y_pred != y_true)
        print(f"Number of differing predictions: {num_diffs}")
    

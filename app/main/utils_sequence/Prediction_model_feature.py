import tensorflow as tf
import numpy as np
import os


class Windowed_Dataset:

    #TODO: split X
    def __init__(self, X, split_time, window_size, shuffle_buffer, train_batch_size, val_batch_size, prediction_length=1):
        self._split_time = split_time
        self._x_train = X[:self._split_time]
        self._x_valid = X[self._split_time:]
        self._window_size = window_size
        self._prediction_length = prediction_length
        self._train_dataset = self._create_train_dataset(train_batch_size, shuffle_buffer)
        self._val_dataset = self._create_validation_dataset(val_batch_size)

    
    
    def _windowed_dataset(self, series, batch_size):
        if len(series.shape) == 1:
            series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(self._window_size + self._prediction_length, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self._window_size + self._prediction_length))
        return ds


    #Define a train dataset
    def _create_train_dataset(self, batch_size, shuffle_buffer):
        ds = self._windowed_dataset(self._x_train, batch_size)
        # Take care because it can destroy the order of the sequence
        #ds = ds.shuffle(shuffle_buffer)
        y_true_slice = self._prediction_length * (-1)
        ds = ds.map(lambda w: (w[:y_true_slice], w[y_true_slice:,0]))
        return ds.batch(batch_size).prefetch(1)

    
    #Define a validation dataset
    def _create_validation_dataset(self, batch_size):
        ds = self._windowed_dataset(self._x_valid, batch_size)
        y_true_slice = self._prediction_length * (-1)
        ds = ds.map(lambda w: (w[:y_true_slice], w[y_true_slice:,0]))
        return ds.batch(batch_size).prefetch(1)

    
    #Doesn't work well with this dataset
    def normalize_data(self, x):
        x_mean = x.mean()
        x_std = x.std()
        x_norm = (x-x_mean)/x_std
        return x_norm
    
    
    def get_train_dataset(self):
        return self._train_dataset
    
    
    def get_val_dataset(self):
        return self._val_dataset


class Prediction_model_feature:
    
    
    def __init__(self, X, window_size, train_batch_size, val_batch_size, shuffle_buffer, name, index, n_features=1):
        self._X = X
        self._window_size = window_size
        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._shuffle_buffer = shuffle_buffer
        self._name = name
        self._n_features = n_features
        self._index = index
        self._model = None

    
    def define_model(self, conv_filter=20, lstm_filter=40, dense_filter=16, prediction_length=1):
        
        #To reset any variable in Tensorflow
        tf.random.set_seed(51)
        np.random.seed(51)
    
        self._model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=conv_filter, kernel_size=5,
                                  strides=1, padding="causal",
                                  activation="relu",
                                  input_shape=[self._window_size, self._n_features]),
            tf.keras.layers.LSTM(lstm_filter, return_sequences=True),
            tf.keras.layers.LSTM(lstm_filter//2),
            tf.keras.layers.Dense(dense_filter, activation="relu"),
            tf.keras.layers.Dense(dense_filter//2, activation="relu"),
            tf.keras.layers.Dense(prediction_length),
            tf.keras.layers.Lambda(lambda x: x * 400)
        ],
        name=self._name.replace(' ', '_'))
        
        return self._model
    
    
    def _define_x_features(self):
        self._x_features = self._X[:, self._index]
        return self._x_features
    
    
    def train_model(self, lr, epochs, split_time):
        self._split_time = split_time
        #Define features involved in the dataset
        self._x_features = self._define_x_features()
        self._dataset = Windowed_Dataset(self._x_features, self._split_time, self._window_size, self._shuffle_buffer, self._train_batch_size, self._val_batch_size)
        #Train model
        optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
        self._model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=optimizer,
                      metrics=["mae"])
        history = self._model.fit(self._dataset.get_train_dataset(),
                                 epochs=epochs,
                                 validation_data=self._dataset.get_val_dataset())
        return history
    
    
    def get_model(self):
        if self._model != None:
            return self._model
        else:
            return self.define_model()
        
    
    def save_weights(self, museum_sequence_path):
        #Create the folder where the weights are saved
        model_feature_folder = os.path.join(museum_sequence_path['weights_folder'], 'model_feature_'+str(self._index))
        if not os.path.exists(model_feature_folder):
            os.makedirs(model_feature_folder)
        #Save weights
        self._model.save_weights(os.path.join(model_feature_folder, 'weights_feature_'+str(self._index)))
        
        
    def load_weights(self, museum_sequence_path):
        #Find the folder where the weights are saved
        model_feature_folder = os.path.join(museum_sequence_path['weights_folder'], 'model_feature_'+str(self._index))    
        #Load weights
        self._model.load_weights(os.path.join(model_feature_folder, 'weights_feature_'+str(self._index)))
        return self._model
    
    
    def get_indexes_features(self):
        return []
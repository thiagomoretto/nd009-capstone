import numpy as np
import keras


class TrainTestDataset(object):
    """Utility class to represent train, test and validation data sets"""
    def __init__(self, X_train, X_test, X_valid, y_train, y_test, y_valid):
        self.X_train = X_train
        self.X_test  = X_test
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_test  = y_test
        self.y_valid = y_valid

    @property
    def shape(self):
        return self.X_train.shape, self.X_test.shape, self.X_valid.shape, \
            self.y_train.shape, self.y_test.shape, self.y_valid.shape
    
    def get_train_generator(self):
        path_to_label = {
            self.X_train[idx]: self.y_train[idx]
            for idx in range(len(self.X_train))
        }
        return DataGenerator(self.X_train, path_to_label)
    
    def get_validation_generator(self):
        path_to_label = {
            self.X_valid[idx]: self.y_valid[idx]
            for idx in range(len(self.X_valid))
        }
        return DataGenerator(self.X_valid, path_to_label)
                
    @staticmethod
    def split(X, y, test_size, valid_size, random_state=42):
        from sklearn.model_selection import train_test_split
        test_val_size = test_size + valid_size
        X_train, X_test_val, y_train, y_test_val =\
            train_test_split(X, y, test_size=test_val_size, random_state=random_state)
        X_valid, X_test, y_valid, y_test =\
            train_test_split(X_test_val, y_test_val, test_size=test_size / (test_size + valid_size),
                             random_state=random_state)
        return TrainTestDataset(X_train, X_test, X_valid, y_train, y_test, y_valid)
    
    
class TrainedModel(object):
    """Wraps a trained model providing useful methods"""
    def __init__(self, name, model, train_history):
        self.name = name
        self.model = model
        self.train_history = train_history
        
    def test(self, X_test, y_test):
        import numpy as np
        import tensorflow as tf
        
        with tf.Session() as sess:
            self.model.load_weights('weights.best.{}.hdf5'.format(self.name))           
            predictions = [
                np.argmax(self.model.predict(np.expand_dims(feature, axis=0)))
                for feature in X_test
            ]
            test_accuracy = 100 *\
                np.sum(np.array(predictions) == np.argmax(y_test, axis=1)) / \
                len(predictions)
        return test_accuracy
    
    def predict(self, X):
        import numpy as np
        import tensorflow as tf

        with tf.Session() as sess:
            self.model.load_weights('weights.best.{}.hdf5'.format(self.name))           
            y_pred = np.array([
                np.argmax(self.model.predict(np.expand_dims(feature, axis=0)))
                for feature in X
            ])
        return y_pred


class Trainer(object):
    """Utility class to train models"""
    def __init__(self, name, model):
        self.name = name
        self.model = model
        
    def train(self, train_dataset, epochs=5, batch_size=16, verbose=1, verbose_checkpointer=1, use_generator=False):
        from keras.callbacks import ModelCheckpoint 
        import tensorflow as tf
        import keras.backend as K

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            K.set_session(sess)
            K.set_learning_phase(1) 
    
            filepath = 'weights.best.{}.hdf5'.format(self.name)
            checkpointer = ModelCheckpoint(filepath=filepath, verbose=verbose_checkpointer,
                                           save_best_only=True)

            if use_generator:
                self.history =\
                    self.model.fit_generator(generator=train_dataset.get_train_generator(),
                                             validation_data=train_dataset.get_validation_generator(),
                                             epochs=epochs,
                                             callbacks=[checkpointer],
                                             verbose=verbose,
                                             use_multiprocessing=False,
                                             workers=1);

            else:
                self.history =\
                    self.model.fit(train_dataset.X_train,
                                   train_dataset.y_train,
                                   validation_data=(train_dataset.X_valid,
                                                    train_dataset.y_valid),
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   callbacks=[checkpointer],
                                   verbose=verbose);
                    
        return TrainedModel(self.name, self.model, self.history)
    
    def __repr__(self):
        return "Trainer(name={}, model={})".format(
            self.name, self.model)
    

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=5, dim=(224, 224), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 5), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = path_to_tensor(ID)
            y[i,] = self.labels[ID]

        return X, y
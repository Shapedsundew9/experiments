"""A boiler plate DNN to compare with Hurlabbab optimisation.

Run as script.
Currently used to explore keras classes & methods.
"""

# Silence Tesnorflow info and warnings ('2')
# Use '1' just to silence info or '0' for everything including debug.
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from json import dumps, loads
from hashlib import blake2b
from os.path import exists
from tqdm import tqdm
from sys import exit
from functools import partial
from gzip import open as gz_open


_DATA_HASHES_FILE_BASE = 'data_hashes'
_VALIDATION_SAMPLE_NUM = 10000


def create_model():
    """Create a standard model to use in all the experiments.

    Also returns the definition of the early stopping callback.
    
    Returns
    -------
    (keras.model, EarlyStopping)
    """
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()
    return model, es


def dumpz(obj, filepath):
    """Save obj (overwriting) as filepath as a compressed JSON.
    
    Args
    ----
    obj (dict/list): JSON representable python object
    filepath (str): File path to save obj as a compressed JSON
    """
    with gz_open(filepath + '.json.gz', 'wt') as file:
        file.write(dumps(obj, sort_keys=True, indent=4))


def loadz(filepath):
    """Load a JSON object from the compressed filepath.
    
    Args
    ----
    filepath (str): File path to load obj from as a compressed JSON

    Returns
    -------
    (obj): A compatible python JSON object
    """
    with gz_open(filepath + '.json.gz', 'rt') as file:
        return loads(file.read())


def load_data():
    """Load the MNIST data set from the Keras library.
    
    The data set is converted to numpy arrays and partitioned into a training
    and test set. If an image data hash file exists it will be used to validate
    the data is as expected (could change with versions of libraries). If the file
    does not exists it is generated.

    Returns
    -------
    (dict): Of structure:
    {
        'x_val': numpy.array(),
        'y_val': numpy.array(),
        'x_train': numpy.array(),
        'y_train': numpy.array(),
        'x_test': numpy.array(),
        'y_test': numpy.array()
    }
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are NumPy arrays)
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve 10,000 samples for validation
    data = {
        'x_val': x_train[-_VALIDATION_SAMPLE_NUM:],
        'y_val': y_train[-_VALIDATION_SAMPLE_NUM:],
        'x_train': x_train[:-_VALIDATION_SAMPLE_NUM],
        'y_train': y_train[:-_VALIDATION_SAMPLE_NUM],
        'x_test': x_test,
        'y_test': y_test
    }

    # Create a hash of every image and check it against the reference if it exists
    hashes = {k: [blake2b(x, digest_size=8).hexdigest() for x in data[k]] for k, v in data.items()}
    if not exists(_DATA_HASHES_FILE_BASE):
        dumpz(hashes, _DATA_HASHES_FILE_BASE)
    else:
        valid_hashes = loadz(_DATA_HASHES_FILE_BASE, hashes)
        print('Validating data reproducibility with image data hash file: ' + _DATA_HASHES_FILE_BASE)
        for data, valid in tqdm(zip(hashes.items(), valid_hashes.items()), desc='Data sets'):
            if not all([d == v for d, v in zip(data[1], valid[1])]):
                print('Loaded ' + data[0] + ' data set not consistent with image data hash file contents.')
                exit(1)
    return data


def _fit(model, es, data):
    """Fit function for the model.
    
    Args
    ----
    model (keras.model): The DNN model (is modified)
    es (EarlyStopping): Early stopping callback function.
    data (dict): See load_data() return value for structure.
    """
    model.fit(
        data['x_train'],
        data['y_train'],
        batch_size=64,
        epochs=50,
        validation_data=(data['x_val'], data['y_val']),
        callbacks=[es]
    )


def align_weights(model, ref_model):
    """Order the weights in model to minimise the euclidean distance from the weights in ref_model.
    
    DNN's have degenerate structures. In order to understand the relative performance of
    different initial states it is important to have a consistent relative starting point.
    
    align_weights() orders neurons in model to minimise the the euclidean distance to
    the weights of equivilent neuron positions in ref_model.

    This function assumes dense layers between input and output.
    e.g.
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    digits (InputLayer)         [(None, 784)]             0         
                                                                    
    dense_1 (Dense)             (None, 64)                50240     
                                                                    
    dense_2 (Dense)             (None, 64)                4160      
                                                                    
    predictions (Dense)         (None, 10)                650       

    and model.get_weights() returns a flat list of numpy arrays e.g.

    <class 'list'> :  len == 6
    <class 'numpy.ndarray'> :  (784, 64)    -- dense_1 weights
    <class 'numpy.ndarray'> :  (64,)        -- dense_1 bias
    <class 'numpy.ndarray'> :  (64, 64)     -- dense_2 weights
    <class 'numpy.ndarray'> :  (64,)        -- dense_2 bias
    <class 'numpy.ndarray'> :  (64, 10)     -- predictions weights
    <class 'numpy.ndarray'> :  (10,)        -- predictions bias

    Args
    ----
    model (keras.model): Model with weights to align (must be the same shape as ref_model)
    ref_model (keras.model): Model to align weights with
    """
    pass
    # TODO: Can draw this out as a graph with fixed neuron positions as an example.

if __name__ == '__main__':
    data = load_data()
    model, es = create_model()
    fit = partial(_fit, es=es, data=data)
    a=model.get_weights()
    print(type(a), ': ', print(len(a)))
    for i in a: print(type(i), ': ', i.shape)


"""
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)

# Weights
print('Layer 1 weights: ', ','.join(map(lambda x: str(x.shape), model.get_layer('dense_1').get_weights())))
print('Layer 2 weights: ', ','.join(map(lambda x: str(x.shape), model.get_layer('dense_2').get_weights())))
print('Output weights: ', ','.join(map(lambda x: str(x.shape), model.get_layer('predictions').get_weights())))
"""

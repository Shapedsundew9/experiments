"""A boiler plate DNN to compare with Hurlabbab optimisation.

Run as script.
Currently used to explore keras classes & methods.
"""

# Silence Tesnorflow info and warnings ('2')
# Use '1' just to silence info or '0' for everything including debug.
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.random import set_seed as tf_seed
from tensorflow.keras import layers, initializers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from json import dumps, loads
from hashlib import blake2b
from os.path import exists, join
from os import makedirs
from tqdm import tqdm, trange
from sys import exit
from functools import partial
from gzip import open as gz_open
from numpy.linalg import norm
from numpy import isclose, newaxis, concatenate, empty, var, mean, array, save, load, sort
from numpy.random import seed as np_seed
from random import seed as py_seed
from gc import collect
from scipy.optimize import linear_sum_assignment
from matplotlib import use
use('GTK3Cairo')
import matplotlib.figure as figure 

# Setup
_DATA_HASHES_FILE_BASE = 'data_hashes'
_VALIDATION_SAMPLE_NUM = 10000
_SEED = 42
_POPULATION = 1000
_WEIGHTS_INITIALIZER = initializers.GlorotUniform()
_BIAS_INITIALIZER = initializers.Zeros()

# Experiment 1
_EXP1_DIR = 'exp1'
_EXP1_SW_BASE = join(_EXP1_DIR, 'start_weights')
_EXP1_EW_BASE = join(_EXP1_DIR, 'end_weights')
_EXP1_SD_BASE = join(_EXP1_DIR, 'start_distances')
_EXP1_ED_BASE = join(_EXP1_DIR, 'end_distances')
_EXP1_TD_BASE = join(_EXP1_DIR, 'travel_distances')
_EXP1_SA_BASE = join(_EXP1_DIR, 'start_accuracy')
_EXP1_EA_BASE = join(_EXP1_DIR, 'end_accuracy')
_EXP1_CP_BASE = join(_EXP1_DIR, 'centre_point')

# Experiment 2
_EXP2_DIR = 'exp2'
_EXP2_EV_BASE = join(_EXP2_DIR, 'end_variances')
_EXP2_EA_BASE = join(_EXP2_DIR, 'end_averages')
_EXP2_EO_BASE = join(_EXP2_DIR, 'end_outputs')
_EXP2_OS_BASE = join(_EXP2_DIR, 'output_seperations')

# Experiment 3
_EXP3_DIR = 'exp3'
_EXP3_IA_BASE = join(_EXP3_DIR, 'intermediate_accuracy')
_EXP3_IS_BASE = join(_EXP3_DIR, 'intermediate_separations')



# Functions defined at runtime
fit = None
evaluate = None
predict = None

def create_model():
    """Create a standard model to use in all the experiments.
    
    Returns
    -------
    (keras.model) The model is compiled.
    """
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


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


def save_weights(model, filepath):
    """Save model weights in an hdf5 file in folder at filepath.
    
    Args
    ----
    model (keras.model): Model from which to store the weights.
    filepath (str): Base filepath
    """
    model.save_weights(filepath + '.h5')


def load_weights(model, filepath):
    """Load model weights from folder at filepath.
    
    Args
    ----
    model (keras.model): Model from which to load the weights.
    filepath (str): Base filepath
    """
    model.load_weights(filepath + '.h5')
    return model


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
        callbacks=[es],
        verbose=0
    )


def _evaluate(model, data):
    """Evaluate the model.
    
    Args
    ----
    model (keras.model): The DNN model (is modified)
    data (dict): See load_data() return value for structure.
    """
    results = model.evaluate(
        data['x_test'],
        data['y_test'],
        batch_size=64,
        verbose=0,
        return_dict=True
    )
    return results['sparse_categorical_accuracy']


def _predict(model, data):
    """Generate the last layer outputs for data.
    
    Args
    ----
    model (keras.model): The DNN model (is modified)
    data (dict): See load_data() return value for structure.
    """
    return model.predict(
        data['x_test'],
        batch_size=64,
        verbose=0
    )


def new_weights(model):
    """Re-initialise weights (without re-creating) model.
    
    Args
    ----
    model (keras.model): Model with weights to re-initialize.
    """
    new_weights = []
    for layer in model.get_weights():
        if len(layer.shape) > 1:
            new_weights.append(_WEIGHTS_INITIALIZER(shape=layer.shape))
        else:
            new_weights.append(_BIAS_INITIALIZER(shape=layer.shape))
    model.set_weights(new_weights)


def flatten_weights(model):
    """Extract the weights from model as a flattened numpy array.
    
    Args
    ----
    model (keras.model): Model with weights to extract.

    Returns
    -------
    (1D-ndarray)
    """
    return concatenate(tuple(w.flatten() for w in model.get_weights()))


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
    layers, ref_layers = model.get_weights(), ref_model.get_weights()
    new_layers = []
    pairs = tuple(zip(layers, ref_layers))
    for i, (layer, ref_layer) in enumerate(pairs):
        if not i & 1:
            layer = layer.reshape(layer.shape[::-1])
            ref_layer = ref_layer.reshape(ref_layer.shape[::-1])

            # Savage bit of numpy.
            # Calculate the Euclidean distance - "norm()""
            # between the weights going into a neuron in layer and the weights for each of the neurons
            # in ref_layer, for all neurons in layer - "layer[:, newaxis] - ref_layer, axis=2". 
            # ...
            new_order = linear_sum_assignment(norm(layer[:, newaxis] - ref_layer, axis=2))[1]
            new_layers.append(layer[new_order].reshape(layer.shape[::-1]))
        else:
            new_layers.append(layer[new_order])

    for layer, new_layer in zip(layers, new_layers):
        print(layer.shape, new_layer.shape)

        if len(layer.shape) == 2:
            old_hashes = {hash(sort(row).data.tobytes()) for row in layer.reshape(layer.shape[::-1])}
            new_hashes = {hash(sort(row).data.tobytes()) for row in new_layer.reshape(new_layer.shape[::-1])}
            print(sorted(old_hashes)[:5])
            print(sorted(new_hashes)[:5])
            assert old_hashes == new_hashes

    old_predictions = predict(model)
    tmp_layers = [new_layers[0]]
    tmp_layers.extend(layers[1:])
    model.set_weights(tmp_layers)
    new_predictions = predict(model)
    for old, new in zip(old_predictions, new_predictions):
        print(old)
        print(new)
        barf()
    # TODO: Can draw this out as a graph with fixed neuron positions as an example.


def set_seed(seed=_SEED):
    """Set the seed of all random generators.
    
    This is necessary to reproduce training a model.

    Args
    ----
    seed (int): The seed.
    """
    np_seed(_SEED)
    py_seed(_SEED)
    tf_seed(_SEED)


def reproducibility():
    """Validates results can be reproduced."""
    print("Validating reproducibility...", end='', flush=True)
    model1 = create_model()
    model2 = create_model()
    model2.set_weights(model1.get_weights())
    set_seed()
    fit(model1)
    set_seed()
    fit(model2)
    for l1, l2 in zip(model1.get_weights(), model2.get_weights()):
        assert isclose(l1, l2).all(), 'Training is not reproducible! Not even close!'
        assert (l1 == l2).all(), 'Training is not reproducible! Not exact!'
    print('OK', flush=True)


def plot_histogram(data, title, filename):
    """Plot a histogram of data with title.

    Args
    ----
    data (1D-array-like): Data to histogram
    title (str): Title to put on histogram
    """
    fig = figure.Figure(figsize=(12.0, 8.0))
    ax = fig.add_subplot()
    ax.hist(data, 100)
    ax.set_title(title)
    ax.set_xlabel('Separation distance')
    ax.set_ylabel('Count')
    fig.savefig(f'{filename}.png')


def experiment_1(model):
    """Separations of initialisation and trained states in hyperspace.
        
    Args
    ----
    model (keras.model): Model to use.
    """
    if not exists(_EXP1_DIR):
        makedirs(_EXP1_DIR)

    # Generate initial weights an separations from each other
    if not exists(f'{_EXP1_SD_BASE}.json.gz'):
        data = empty((_POPULATION, len(flatten_weights(model))))
        for i in trange(_POPULATION, desc='Exp 1: Generation'):
            if not exists(f'{_EXP1_SW_BASE}_{i:06d}.h5'):
                new_weights(model)
                save_weights(model, f'{_EXP1_SW_BASE}_{i:06d}')
            else:
                load_weights(model, f'{_EXP1_SW_BASE}_{i:06d}')
            data[i] = flatten_weights(model)

        start_distances = []
        centre_point = mean(data, axis=0)
        for i in trange(_POPULATION, desc='Exp 1: Initial separation'):
            for j in range(i + 1, _POPULATION):
                start_distances.append({'model_i': i, 'model_j': j, 'distance': norm(data[i]-data[j])})
        dumpz(start_distances, _EXP1_SD_BASE)
        dumpz(centre_point.tolist(), _EXP1_CP_BASE)
    else:
        start_distances = loadz(_EXP1_SD_BASE)
        centre_point = array(loadz(_EXP1_CP_BASE))
    centre_distance_from_origin = norm(centre_point)
    # TODO: Add to plot
    plot_histogram([d['distance'] for d in start_distances], 'Separations of initial starting positions.', _EXP1_SD_BASE)

    # Fit the DNN's and calaculate the distance travelled to the minimum from the starting weights
    # and the separations of all the minima found
    if not exists(f'{_EXP1_ED_BASE}.json.gz'):
        travel_distances = []
        data = empty((_POPULATION, len(flatten_weights(model))))
        for i in trange(_POPULATION, desc='Exp 1: Fitting'):
            load_weights(model, f'{_EXP1_SW_BASE}_{i:06d}')
            start_position = flatten_weights(model)
            if not exists(f'{_EXP1_EW_BASE}_{i:06d}.h5'):
                fit(model)
                save_weights(model, f'{_EXP1_EW_BASE}_{i:06d}')
                collect()
                clear_session()
            else:
                load_weights(model, f'{_EXP1_EW_BASE}_{i:06d}')
            data[i] = flatten_weights(model)
            travel_distances.append(norm(start_position - data[i]))
        dumpz(travel_distances, _EXP1_TD_BASE)

        end_distances = []
        for i in trange(_POPULATION, desc='Exp 1: Final separation'):
            for j in range(i + 1, _POPULATION):
                end_distances.append(norm(data[i]-data[j]))
        dumpz(end_distances, _EXP1_ED_BASE)
    else:
        end_distances = loadz(_EXP1_ED_BASE)
        travel_distances = loadz(_EXP1_TD_BASE)
    plot_histogram(end_distances, 'Separations of found minima positions.', _EXP1_ED_BASE)
    plot_histogram(travel_distances, 'Distance travelled from start position to minima position.', _EXP1_TD_BASE)

    # Initial accuracy (before training)
    if not exists(f'{_EXP1_SA_BASE}.json.gz'):
        start_accuracy = []
        for i in trange(_POPULATION, desc='Exp 1: Initial accuracy'):
            start_accuracy.append(evaluate(load_weights(model, f'{_EXP1_SW_BASE}_{i:06d}')))
        dumpz(start_accuracy, _EXP1_SA_BASE)            
    else:
        start_accuracy = loadz(_EXP1_SA_BASE)            
    plot_histogram(start_accuracy, 'Initial accuracy (before training)', _EXP1_SA_BASE)

    # Final accuracy (after training)
    if not exists(f'{_EXP1_EA_BASE}.json.gz'):
        end_accuracy = []
        for i in trange(_POPULATION, desc='Exp 1: Final accuracy'):
            end_accuracy.append(evaluate(load_weights(model, f'{_EXP1_EW_BASE}_{i:06d}')))
        dumpz(end_accuracy, _EXP1_EA_BASE)            
    else:
        end_accuracy = loadz(_EXP1_EA_BASE)            
    plot_histogram(end_accuracy, 'Final accuracy (after training)', _EXP1_EA_BASE)


def experiment_2(model):
    """Distribution of starting states in hyperspace & separations of prediction probabilities.
        
    Args
    ----
    model (keras.model): Model to use.
    """
    if not exists(_EXP2_DIR):
        makedirs(_EXP2_DIR)

    if not exists(f'{_EXP2_EV_BASE}.npy') or not exists(f'{_EXP2_EA_BASE}.npy'):
        data = empty((_POPULATION, len(flatten_weights(model))))
        for i in trange(_POPULATION, desc='Exp 2: Loading'):
            data[i] = flatten_weights(load_weights(model, f'{_EXP1_SW_BASE}_{i:06d}'))
        variances = var(data, axis=1)
        averages = mean(data, axis=1)
        save(_EXP2_EV_BASE, variances)
        save(_EXP2_EA_BASE, averages)
    else:
        variances = load(f'{_EXP2_EV_BASE}.npy')
        averages = load(f'{_EXP2_EA_BASE}.npy')
    plot_histogram(variances, 'Final weight variances', _EXP2_EV_BASE)
    plot_histogram(averages, 'Final weight averages', _EXP2_EA_BASE)

    if not exists(f'{_EXP2_EO_BASE}.npy'):
        data = empty((_POPULATION, _VALIDATION_SAMPLE_NUM, model.get_layer('predictions').output_shape[1]))
        for i in trange(_POPULATION, desc='Exp 2: Predicting'):
            load_weights(model, f'{_EXP1_SW_BASE}_{i:06d}')
            data[i] = predict(model)
        save(_EXP2_EO_BASE, data)

    if not exists(f'{_EXP2_OS_BASE}.json.gz'):
        outputs = load(f'{_EXP2_EO_BASE}.npy')
        output_distances = []
        for i in trange(_POPULATION, desc='Exp 2: Distances'):
            for j in range(i + 1, _POPULATION):
                output_distances.append({'model_i': i, 'model_j': j, 'distance': norm(outputs[i]-outputs[j])})
        dumpz(output_distances, _EXP2_OS_BASE)
    else:
        output_distances = loadz(_EXP2_OS_BASE)
    plot_histogram([d['distance'] for d in output_distances], 'Separations of test set predictions.', _EXP2_OS_BASE)


def experiment_3(model):
    """Catchment area.

    Args
    ----
    model (keras.model): Model to use.
    """
    if not exists(_EXP3_DIR):
        makedirs(_EXP3_DIR)

    # Need data from experiment 1
    if not exists(f'{_EXP1_ED_BASE}.json.gz'):
        experiment_1(model)
    end_distances = loadz(_EXP1_ED_BASE)

    # Find the maximum radius of the catchment area for a minimum.
    closest_pair = sorted(end_distances, key=lambda x: x['distance'])[0]
    weights_i = flatten_weights(load_weights(model, f'{_EXP1_SW_BASE}_{closest_pair["model_i"]:06d}'))
    max_catchment_radius = closest_pair['distance'] / 2.0
    delta_radius = max_catchment_radius / _POPULATION
    tmp_model = create_model()

    # Randomly fuzz model_i trained weights so they sit at a specific radius from
    # the trained weights.
    for nr in range(_POPULATION):

    



    # Binary c
    if not exists(f'{_EXP2_EV_BASE}.npy') or not exists(f'{_EXP2_EA_BASE}.npy'):
        pass


if __name__ == '__main__':

    # Set up
    data = load_data()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=2)
    model = create_model()
    model.summary()
    fit = partial(_fit, es=es, data=data)
    evaluate = partial(_evaluate, data=data)
    predict = partial(_predict, data=data)

    # Sanity
    #reproducibility()

    # Experiments
    #experiment_1(model)
    #experiment_2(model)
    #align_weights(model, create_model())
"""
    a=model.get_weights()
    print(type(a), ': ', print(len(a)))
    for i in a: print(type(i), ': ', i.shape)


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

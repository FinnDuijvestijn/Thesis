import copy
import datetime
import logging
import os
import pickle
import bz2
import psutil
from multiprocessing import Pool

import numpy as np

from fmnist_data import load_fashion_mnist_data
from set_mlp import SET_MLP, Relu, Softmax , CrossEntropy

# new
from set_motif_mlp import SET_MOTIF_MLP

FOLDER = "benchmarks"

def single_run_density(Motif, run_id, set_params, density_levels, n_training_epochs,
                       fname="", save_compressed=True):
    """
    the density levels are the set epsilon sparsity levels
    """
    print(f"[run={run_id}] Job started")
    n_training_samples = 5000  # max 60000 for Fashion MNIST
    n_testing_samples = 1000  # max 10000 for Fashion MNIST
    n_features = 784  # Fashion MNIST has 28*28=784 pixels as features

    # SET model parameters
    n_hidden_neurons_layer = set_params['n_hidden_neurons_layer']
    zeta = set_params['zeta']
    batch_size = set_params['batch_size']
    dropout_rate = set_params['dropout_rate']
    learning_rate = set_params['learning_rate']
    momentum = set_params['momentum']
    weight_decay = set_params['weight_decay']

    # sum_training_time = 0

    np.random.seed(run_id)

    x_train, y_train, x_test, y_test = load_fashion_mnist_data(n_training_samples, n_testing_samples, run_id)

    if os.path.isfile(fname):
        with open(fname, "rb") as h:
            results = pickle.load(h)
    else:
        results = {'density_levels': density_levels, 'runs': []}

    for epsilon in density_levels:
        logging.info(f"[run_id={run_id}] Starting SET-Sparsity: epsilon={epsilon}")
        set_params['epsilon'] = epsilon
        # create SET-MLP (Multilayer Perceptron w/ adaptive sparse connectivity trained & Sparse Evolutionary Training)

        if Motif:
            set_mlp = SET_MOTIF_MLP((x_train.shape[1], n_hidden_neurons_layer, n_hidden_neurons_layer, n_hidden_neurons_layer,
                           y_train.shape[1]), (Relu, Relu, Relu, Softmax), epsilon=epsilon)
        else:
            set_mlp = SET_MLP((x_train.shape[1], n_hidden_neurons_layer, n_hidden_neurons_layer, n_hidden_neurons_layer,
                           y_train.shape[1]), (Relu, Relu, Relu, Softmax), epsilon=epsilon)

        start_time = datetime.datetime.now()
        # train SET-MLP to find important features
        set_metrics = set_mlp.fit(x_train, y_train, x_test, y_test, loss=CrossEntropy, epochs=n_training_epochs,
                              batch_size=batch_size, learning_rate=learning_rate,
                              momentum=momentum, weight_decay=weight_decay, zeta=zeta, dropout_rate=dropout_rate,
                              testing=True, run_id=run_id,
                              save_filename="", monitor=False)

        dt = datetime.datetime.now() - start_time

        run_result = {'run_id': run_id, 'set_params': copy.deepcopy(set_params), 'set_metrics': set_metrics,
                      'evolved_weights': set_mlp.weights_evolution, 'training_time': dt}

        results['runs'].append({'set_sparsity': epsilon, 'run': run_result})

        fname = f"{FOLDER}/set_mlp_density_run_{run_id}.pickle"
        # save preliminary results
        if save_compressed:
            with bz2.BZ2File(f"{fname}.pbz2", "w") as h:
                pickle.dump(results, h)
        else:
            with open(fname, "wb") as h:
                pickle.dump(results, h)


def fmnist_train_set_differnt_densities_sequential(Motif=False, runs=10, n_training_epochs=100, set_sparsity_levels=None, use_logical_cores=True):
    # SET model parameters
    set_params = {'n_hidden_neurons_layer': 3000,
                #   'epsilon': 13,  # set the sparsity level
                  'zeta': 0.3,  # in [0..1]. Percentage of unimportant connections to be removed and replaced
                  'batch_size': 40, 'dropout_rate': 0, 'learning_rate': 0.05, 'momentum': 0.9, 'weight_decay': 0.0002}

    start_test = datetime.datetime.now()

    if use_logical_cores:
        n_cores = psutil.cpu_count(logical=use_logical_cores)
        with Pool(processes=n_cores) as pool:
            futures = []
            for i in range(runs):
                fname = f"{FOLDER}/set_mlp_density_run_{i}.pickle"

                futures.append(pool.apply_async(single_run_density, (Motif, i, set_params,
                    set_sparsity_levels, n_training_epochs, fname)))

            for i, future in enumerate(futures):
                print(f'[run={i}] Starting job')
                future.get()
                print(f'-----------------------------[run={i}] Finished job')

    else:
        for i in range(runs):
            fname = f"{FOLDER}/set_mlp_density_run_{i}.pickle"
            print(f'[run={i}] Starting job')
            single_run_density(Motif, i, set_params, set_sparsity_levels, n_training_epochs, fname)
            print(f'-----------------------------[run={i}] Finished job')

    delta_time = datetime.datetime.now() - start_test

    print("-" * 30)
    print(f"Finished the entire process after: {delta_time.seconds}s")


if __name__ == "__main__":
    FOLDER = "benchmarks"

    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    # if set to True the set_motif applies
    Motif = True

    # if logical_cores true : runs are parallel --> faster
    use_logical_cores = True
    test_density = True

    if test_density:
        runs = 6
        n_training_epochs = 100
        # set_sparsity_levels = [1, 3, 5, 7, 9, 11] # for short test
        set_sparsity_levels = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] # , 512, 1024]
        # the levels are chosen to have [0.16, 0.5, 1, 2, 5, 10, 20, 40, 80, 100] % density in the first layer
        fmnist_train_set_differnt_densities_sequential(Motif, runs, n_training_epochs, set_sparsity_levels, use_logical_cores=use_logical_cores)
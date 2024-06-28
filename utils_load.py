import pickle
import bz2
import os
from collections import defaultdict
import numpy as np


def load_file(fname):
    if fname.endswith(".pbz2"):
        with bz2.BZ2File(fname, 'r') as h:
            data = pickle.load(h)
    else:
        with open(fname, "rb") as h:
            data = pickle.load(h)

    return data


def load_files_in_folder(folder_path='benchmarks'):
    """Function to load all files in a folder."""
    files = os.listdir(folder_path)
    loaded_files = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        loaded_file = load_file(file_path)
        loaded_files.append(loaded_file)
    return loaded_files


def extract_data(loaded_files):
    """Extracts desired information from loaded files."""
    extracted_data = {}

    for i, file_data in enumerate(loaded_files):

        sparsitys = []

        for sparsity_run in file_data['runs']:
            set_sparsity = sparsity_run['set_sparsity']
            set_metrics = sparsity_run['run']['set_metrics']
            training_time = sparsity_run['run']['training_time'].total_seconds()

            info = {
                'sparsity' :  set_sparsity,
                'set_metrics': set_metrics,
                'training_time_seconds': training_time
            }
            sparsitys.append(info)

        extracted_data[f'run {i}'] = sparsitys

    return extracted_data


def get_sparsity_info(extracted_data):
    """Aggregates data into functional statistics."""
    aggregation_info = defaultdict(lambda: {'min_time': float('inf'), 'avg_time': 0, 'max_time': 0, 
                                            'min_test_accuracy': float('inf'), 'avg_test_accuracy': 0, 'max_test_accuracy': 0,
                                            'min_train_accuracy': float('inf'), 'avg_train_accuracy': 0, 'max_train_accuracy': 0})

    for run_id, sparsity_data in extracted_data.items():
        for sparsity_lvl in sparsity_data:
            sparsity = sparsity_lvl['sparsity']
            time = sparsity_lvl['training_time_seconds']
            metrics = sparsity_lvl['set_metrics']

            aggregation_info[sparsity]['min_time'] = min(aggregation_info[sparsity]['min_time'], time)
            aggregation_info[sparsity]['avg_time'] += time
            aggregation_info[sparsity]['max_time'] = max(aggregation_info[sparsity]['max_time'], time)

            aggregation_info[sparsity]['min_train_accuracy'] = min(aggregation_info[sparsity]['min_train_accuracy'], metrics[-1][-2])
            aggregation_info[sparsity]['avg_train_accuracy'] += metrics[-1][-2]
            aggregation_info[sparsity]['max_train_accuracy'] = max(aggregation_info[sparsity]['max_train_accuracy'], metrics[-1][-2])

            aggregation_info[sparsity]['min_test_accuracy'] = min(aggregation_info[sparsity]['min_test_accuracy'], metrics[-1][-1])
            aggregation_info[sparsity]['avg_test_accuracy'] += metrics[-1][-1]
            aggregation_info[sparsity]['max_test_accuracy'] = max(aggregation_info[sparsity]['max_test_accuracy'], metrics[-1][-1])

    for sparsity in aggregation_info:
        aggregation_info[sparsity]['avg_time'] = round(aggregation_info[sparsity]['avg_time'] / len(extracted_data), 3)
        aggregation_info[sparsity]['avg_train_accuracy'] = round(aggregation_info[sparsity]['avg_train_accuracy'] / len(extracted_data), 3)
        aggregation_info[sparsity]['avg_test_accuracy'] = round(aggregation_info[sparsity]['avg_test_accuracy'] / len(extracted_data), 3)

    return dict(aggregation_info)


def get_epoch_info(extracted_data):
    """Aggregates epoch data into functional statistics."""
    aggregation_info = defaultdict(lambda: {'min_epochs': float('inf'), 'avg_epochs': 0, 'max_epochs': 0})
    
    for run_id, sparsity_data in extracted_data.items():
        for sparsity_lvl in sparsity_data:
            sparsity = sparsity_lvl['sparsity']
            metrics = sparsity_lvl['set_metrics'][:,2:]
            aggregation_info[sparsity]['min_epochs'] = np.minimum(metrics, aggregation_info[sparsity]['min_epochs'])
            aggregation_info[sparsity]['avg_epochs'] += metrics
            aggregation_info[sparsity]['max_epochs'] = np.maximum(metrics, aggregation_info[sparsity]['max_epochs'])

    for sparsity in aggregation_info:
        aggregation_info[sparsity]['avg_epochs'] = np.round(aggregation_info[sparsity]['avg_epochs'] / len(extracted_data), 3)

    return dict(aggregation_info)


def total_time(aggregation_info):
    """Calculates time from all sparsity levels into total"""
    values = np.array([list(info.values()) for info in aggregation_info.values()])
    total_min_time = round(np.sum(values[:, 0]), 3)
    total_avg_time = round(np.sum(values[:, 1]), 3)
    total_max_time = round(np.sum(values[:, 2]), 3)

    return total_min_time, total_avg_time, total_max_time


def load_folder(folder):
    """Loades all files in a folder for plotting or comparison"""
    # Lists to store results for each file
    results_listd_all = []
    results_for_plt_all = []

    # Iterate through each file in the folder
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        
        # Load the file
        results_set_og = load_files_in_folder(filepath)
        
        # Extract data
        results_listd = extract_data(results_set_og)
        
        # Get sparsity info
        results_for_plt = get_sparsity_info(results_listd)
        
        # Save results in lists
        results_listd_all.append(results_listd)
        results_for_plt_all.append(results_for_plt)

    return results_listd_all, results_for_plt_all
    
    
def calculate_best_motif(results_listd_all, results_for_plt_all):
    """Caculates the best motif from given benchmarks"""
    # Initialize the results_dict with the correct keys
    results_dict = {f'motif_{i+1}': {} for i in range(len(results_for_plt_all))}

    # Populate the results_dict with accuracies and times
    for i, key in enumerate(results_dict):
        accuracies = [j['avg_test_accuracy'] for j in results_for_plt_all[i].values()]
        times = [j['avg_time'] for j in results_for_plt_all[i].values()]
        avg_epochs = [round(np.mean(j['avg_epochs'][:,1:]), 3) for j in get_epoch_info(results_listd_all[i]).values()]
        results_dict[key] = {'accuracies': accuracies, 'times': times, 'epochs' : avg_epochs }

    # Extract sparsity levels (assuming same for all)
    sparsity_levels = results_for_plt_all[0].keys()

    # Calculate the best accuracy for each sparsity level
    best_accuracies = [max(results_dict[motif]['accuracies'][i] for motif in results_dict) for i in range(len(sparsity_levels))]

    # Initialize dictionaries to store normalized scores and composite scores
    normalized_accuracy = {motif: [] for motif in results_dict}
    normalized_time = {motif: [] for motif in results_dict}
    normalized_epoch_accuracy = {motif: [] for motif in results_dict}
    composite_score = {}

    # Min-max normalization for times
    all_times = [time for motif in results_dict.values() for time in motif['times']]
    min_time, max_time = min(all_times), max(all_times)

    all_avg_epoch_accuracies = [acc for motif in results_dict.values() for acc in motif['epochs']]
    min_epoch_acc, max_epoch_acc = min(all_avg_epoch_accuracies), max(all_avg_epoch_accuracies)

    # Normalize accuracy per sparsity level
    for motif, data in results_dict.items():
        for i, acc in enumerate(data['accuracies']):
            norm_acc = acc / best_accuracies[i]  # Normalize accuracy by the best accuracy at that sparsity level
            normalized_accuracy[motif].append(norm_acc)
        norm_times = [(time - min_time) / (max_time - min_time) for time in data['times']]
        norm_epoch_accuracies = [(acc - min_epoch_acc) / (max_epoch_acc - min_epoch_acc) for acc in data['epochs']]
        normalized_time[motif] = np.mean(norm_times)
        normalized_epoch_accuracy[motif] = np.mean(norm_epoch_accuracies)

    # Calculate the composite score
    for motif in normalized_accuracy:
        avg_normalized_acc = np.mean(normalized_accuracy[motif])
        normalized_time_component = 1 - normalized_time[motif]  # Lower time is better
        avg_normalized_epoch_acc = np.mean(normalized_epoch_accuracy[motif])
        
        # Composite score can be adjusted based on importance of each metric
        composite_score[motif] = (0.35 * avg_normalized_acc +
                                0.35 * normalized_time_component +
                                0.3 * avg_normalized_epoch_acc)  # Adjust weights as needed

    # Print the results
    for motif in composite_score:
        print(f"Motif: {motif}, Composite Score: {composite_score[motif]:.4f}")

    # Determine the best motif based on the highest composite score
    best_motif = max(composite_score, key=composite_score.get)
    print(f"The best motif is: {best_motif} with a composite score of {composite_score[best_motif]:.4f}")

    return composite_score

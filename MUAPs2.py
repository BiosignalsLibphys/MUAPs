
# Importing packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, welch, spectrogram
from scipy.integrate import simps
from scipy.stats import kurtosis, skew
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import pickle

plux = 'no'

if plux == 'yes':
    # Loading data and defining the EMG channel
    emg = np.loadtxt('Group2_EMG_Rep1.txt')
    signal = emg[20000:63000, 6]

    # Converting the signal into mV
    def emg_transfer_function(signal):
        '''
        Convert to mV using Plux's transfer function
        '''

        bits = 16
        vcc = 2.5
        gain = 1100
        emg_v = ((signal/(2**bits-1)) * vcc) / gain
        emg_mv = emg_v * 1000
        return emg_mv


    emg_mv = emg_transfer_function(signal)
    fs = 1000

else:

    def clean_and_convert(value):
        try:
            cleaned_value = ''.join(c for c in value if c.isdigit() or c == '.' or c == '-')
            return float(cleaned_value)
        except ValueError:
            return None


    def process_emg_file(file_path):
        processed_data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    val1, val2 = line.strip().split()
                    clean_val1 = clean_and_convert(val1)
                    clean_val2 = clean_and_convert(val2)
                    if clean_val1 is not None and clean_val2 is not None:
                        processed_data.append((clean_val1, clean_val2))
                except ValueError as e:
                    print(f"Error processing line: {line}. Error: {e}")
        return processed_data


    # Processing each file
    emg_mv = {}
    for condition in ['healthy', 'myopathy', 'neuropathy']:
        file_name = 'emg_' + condition + '.txt'
        emg_mv[condition] = process_emg_file(file_name)
        fs = 4000

    start = 1
    end = 11
    emg_healthy = [item[1] for item in emg_mv['healthy']]
    emg_healthy = emg_healthy[fs*start: fs*end]
    emg_myopathy = [item[1] for item in emg_mv['myopathy']]
    emg_myopathy = emg_myopathy[fs*start: fs * end]
    emg_neuropathy = [item[1] for item in emg_mv['neuropathy']]
    emg_neuropathy = emg_neuropathy[fs*start: fs * end]

    # Plotting
    plt.figure()
    plt.subplot(311)
    plt.plot(emg_healthy)
    plt.title('EMG Data - Healthy')
    plt.ylabel('Value')
    plt.subplot(312)
    plt.plot(emg_myopathy)
    plt.title('EMG Data - Myopathy')
    plt.ylabel('Value')
    plt.subplot(313)
    plt.plot(emg_neuropathy)
    plt.title('EMG Data - Neuropathy')
    plt.xlabel('Sample Number')
    plt.ylabel('Value')
    plt.show()

    emg_healthy -= np.mean(emg_healthy)
    emg_myopathy -= np.mean(emg_myopathy)
    emg_neuropathy -= np.mean(emg_neuropathy)

    emg_mv = [emg_healthy, emg_myopathy, emg_neuropathy]


# BandPass Filtering between 10-500
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Parameters:
    data (array-like): The input signal.
    lowcut (float): The lower frequency of the passband.
    highcut (float): The higher frequency of the passband.
    fs (float): The sampling rate of the signal.
    order (int, optional): The order of the filter. Default is 4.

    Returns:
    array-like: The filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


lowcut = 10
highcut = 499

if plux == 'yes':
    emg_filtered = bandpass_filter(emg_mv, lowcut, highcut, fs, order=4)
    # Plot signal
    x_axis = np.arange(emg_filtered.shape[0]) / fs # seconds
    plt.plot(x_axis, emg_filtered)

    # with open('emg_filtered.pkl', 'wb') as file:
    #     pickle.dump(emg_filtered, file)

    # Segmenting the signal
    # Choosing a segment length - this is a heuristic and might need adjustment
    segment_length = 500  # Example segment length, this may need to be adjusted

    # Calculate the number of segments
    num_segments = emg_filtered.shape[0] // segment_length

    # Reshape the data into segments
    segmented_data = emg_filtered[:num_segments * segment_length].reshape(num_segments, segment_length)

    # Apply ICA
    ica = FastICA(n_components=num_segments, random_state=0)
    components = ica.fit_transform(segmented_data.T).T

    # Plot the first few independent components
    plt.figure(figsize=(15, 10))
    for ii, component in enumerate(components[:5]):
        plt.subplot(5, 1, ii + 1)
        plt.plot(component)
        plt.title(f'Independent Component {ii+1}')
    plt.tight_layout()
    plt.show()

else:
    emg_filtered = [bandpass_filter(emg_mv[ii], lowcut, highcut, fs, order=4) for ii in range(3)]

    segmented_data = np.zeros(3, dtype=object)
    components = np.zeros(3, dtype=object)

    # Segmenting the signal
    # Choosing a segment length - this is a heuristic and might need adjustment
    segment_length = 4000

    # Calculate the number of segments
    num_segments = [emg_filtered[ii].shape[0] // segment_length for ii in range(3)]

    # Reshape the data into segments
    def segment_signal(signal, segment_length):
        """Segment the signal into smaller segments of given length."""
        return [signal[ii:ii + segment_length] for ii in range(0, len(signal), segment_length)]


    segmented_signals = [segment_signal(signal, segment_length) for signal in emg_filtered]
    segmented_signals = np.asarray(segmented_signals)

    # Applying ICA to each segment of each signal
    ica_components = []
    for ii, signal in enumerate(segmented_signals):
        ica = FastICA(n_components=num_segments[ii], random_state=0)
        ica_transformed = ica.fit_transform(segmented_signals[ii].T).T
        ica_components.append(ica_transformed)
    ica_components = np.asarray(ica_components)


# Components selection
def calculate_descriptive_statistics(ica_components, sampling_rate=1000):
    """
    Calculate descriptive statistics for each component.
    Args:
    components (np.ndarray): The independent components.
    sampling_rate (int): The sampling rate of the EMG signal in Hz.

    Returns:
    dict: A dictionary containing the descriptive statistics for each component.
    """
    stats = {}
    for ii, component in enumerate(ica_components):
        # Calculate skewness and kurtosis
        component_skewness = [skew(component[:, jj]) for jj in range(component.shape[1])]
        component_kurtosis = [kurtosis(component[:, jj], fisher=False) for jj in range(component.shape[1])] # Fisher=False for normal kurtosis

        # Peak-to-peak amplitude
        ptp_amplitude = [np.ptp(component[:, jj]) for jj in range(component.shape[1])]

        # Duration (in milliseconds)
        duration = len(component) / sampling_rate * 1000  # Convert to ms

        # Store the statistics
        stats[f'signal {ii+1}'] = {
            'Skewness': component_skewness,
            'Kurtosis': component_kurtosis,
            'Peak-to-Peak Amplitude': ptp_amplitude,
            'Duration (ms)': duration
        }

    return stats

# Calculate descriptive statistics for each component
descriptive_stats = calculate_descriptive_statistics(ica_components)
descriptive_stats


# Feature Selection for Each Signal: Select the representative MUAPs. Here there is the need of see in leterature what should be the parameters
## Threshold-based

def select_muap_components(descriptive_stats, amplitude_threshold, skewness_range, kurtosis_range):
    muap_components = {}

    for signal, stats in descriptive_stats.items():
        muap_components[signal] = []
        for i in range(len(stats['Peak-to-Peak Amplitude'])):
            amplitude = stats['Peak-to-Peak Amplitude'][i]
            skewness = stats['Skewness'][i]
            kurtosis = stats['Kurtosis'][i]

            # Check if the component meets the criteria
            if amplitude > amplitude_threshold and skewness_range[0] <= skewness <= skewness_range[1] and kurtosis_range[0] <= kurtosis <= kurtosis_range[1]:
                muap_components[signal].append({
                    'Component': i + 1,  # Components are 1-indexed
                    'Skewness': skewness,
                    'Kurtosis': kurtosis,
                    'Peak-to-Peak Amplitude': amplitude
                })

    return muap_components


# Define your thresholds and ranges
amplitude_threshold = 0.15
skewness_range = (-1, 1)
kurtosis_range = (5, 15)

# Select MUAP components
muap_components = select_muap_components(descriptive_stats, amplitude_threshold, skewness_range, kurtosis_range)
muap_components


## Clustering-based
def cluster_components(descriptive_stats, eps, min_samples):
    clustered_components = {}

    for signal, stats in descriptive_stats.items():
        # Prepare data for clustering
        features = np.column_stack((stats['Skewness'], stats['Kurtosis'], stats['Peak-to-Peak Amplitude']))

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)

        # Store the clustering results
        clustered_components[signal] = clustering.labels_

    return clustered_components


# Cluster Analysis Parameters
eps = 0.95
min_samples = 2
clustered_results = cluster_components(descriptive_stats, eps, min_samples)


# Cluster Analysis Across All Signals

def cluster_all_signals(descriptive_stats, eps, min_samples):
    all_features = []
    signal_labels = []

    # Combine data from all signals
    for signal, stats in descriptive_stats.items():
        features = np.column_stack((stats['Skewness'], stats['Kurtosis'], stats['Peak-to-Peak Amplitude']))
        all_features.append(features)
        signal_labels.extend([signal] * len(features))

    all_features = np.vstack(all_features)

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(all_features)

    return clustering.labels_, signal_labels

# Example usage
all_signal_clusters, signal_labels = cluster_all_signals(descriptive_stats, eps, min_samples)
all_signal_clusters

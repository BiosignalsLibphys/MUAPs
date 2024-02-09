import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pywt

# --- Step 1: Signal Simulation ---
sample_rate = 1000  # Hz
duration = 10  # seconds
time = np.arange(0, duration, 1 / sample_rate)  # Time vector

# Simulating baseline noise and muscle contractions
baseline_noise = np.random.normal(0, 0.1, time.shape)
contraction1 = ((time > 2) & (time < 4)).astype(float)
contraction2 = ((time > 6) & (time < 8)).astype(float)
contraction_noise = np.random.normal(0, 0.5, time.shape) * (contraction1 + contraction2)
emg_signal = baseline_noise + contraction_noise


# --- Step 2: Bandpass Filtering ---
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


filtered_emg_signal = bandpass_filter(emg_signal, 10.0, 499.0, sample_rate)

# --- Step 3: MUAP Extraction - Peak Detection Method ---
peaks, _ = find_peaks(filtered_emg_signal, height=0.3, distance=25)

# --- Step 4: MUAP Extraction - CWT Method ---
scales = np.arange(1, 128)
coefficients, frequencies = pywt.cwt(filtered_emg_signal, scales, 'morl', 1 / sample_rate)

# Plotting the CWT result
plt.figure(figsize=(15, 5))
plt.imshow(coefficients, extent=[0, duration, 1, max(frequencies)], cmap='jet', aspect='auto',
           vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
plt.colorbar(label='Magnitude')
plt.title('Continuous Wavelet Transform (CWT) of EMG Signal')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.show()

# --- Extracting and Plotting MUAPs ---
# Peak Detection Method
plt.figure(figsize=(15, 5))
plt.plot(time, filtered_emg_signal, label="Filtered EMG Signal")
plt.plot(time[peaks], filtered_emg_signal[peaks], 'rx', label="Identified MUAPs (Peak Detection)")
plt.title("EMG Signal with Identified MUAPs (Peak Detection)")
plt.xlabel("Time (s)")
plt.ylabel("EMG amplitude")
plt.legend()
plt.grid(True)
plt.show()

# CWT Method (Heuristic Selection of Segments)
selected_intervals = [(2.5, 3), (6.5, 7), (7.5, 8)]
muap_window_size = 25  # Window size for individual MUAPs

extracted_muaps = []
for interval in selected_intervals:
    start_index = int(interval[0] * sample_rate)
    end_index = int(interval[1] * sample_rate)
    segment = filtered_emg_signal[start_index:end_index]

    segment_peaks, _ = find_peaks(segment, height=0.3, distance=25)
    for peak in segment_peaks[:1]:  # First peak in each segment
        muap_start = max(0, peak - muap_window_size // 2)
        muap_end = min(len(segment), peak + muap_window_size // 2)
        muap = segment[muap_start:muap_end]
        extracted_muaps.append((start_index / sample_rate + muap_start / sample_rate, muap))

# Plotting extracted MUAPs (CWT Method)
plt.figure(figsize=(15, 5))
for i, (time_offset, muap) in enumerate(extracted_muaps[:4]):
    muap_time = np.linspace(time_offset, time_offset + (len(muap) - 1) / sample_rate, len(muap))
    plt.subplot(1, 4, i + 1)
    plt.plot(muap_time, muap)
    plt.title(f"MUAP {i + 1} (CWT Method)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
plt.tight_layout()
plt.show()

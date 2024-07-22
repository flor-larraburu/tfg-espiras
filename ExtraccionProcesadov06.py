import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Función para calcular la media móvil y la desviación estándar usando operaciones vectorizadas


def calculate_rolling_stats(signal, window_size):
    signal = np.asarray(signal)
    rolling_mean = np.convolve(signal, np.ones(
        window_size), 'valid') / window_size
    squared_signal = np.square(signal)
    rolling_std = np.sqrt(np.convolve(squared_signal, np.ones(
        window_size), 'valid') / window_size - np.square(rolling_mean))
    return rolling_mean, rolling_std

# Función para leer los datos del archivo


def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    column1 = [float(line.split()[0]) for line in data]
    return column1

# Función para guardar la traza en un archivo JSON junto con su velocidad


def save_trace_to_json(data, index):
    os.makedirs('db', exist_ok=True)
    json_file = os.path.join('db', f'traza{index + 1}.json')
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

# Función para suavizar la señal


def smooth_signal(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Función para detectar picos usando la derivada


def detectar_picos(signal, derivative):
    peaks = []
    for i in range(1, len(derivative)):
        if derivative[i-1] > 0 and derivative[i] <= 0:
            peaks.append((i, signal[i]))
    return peaks

# Función para procesar y visualizar los picos


def process_and_visualize_peaks(signal, t, window_size, peak_window_size, threshold_factor):
    signal = np.asarray(signal, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    counter = 0
    smoothed_signal = smooth_signal(signal, window_size=100)
    rolling_mean, rolling_std = calculate_rolling_stats(
        smoothed_signal, window_size)

    highest_peak_global_index = -1000
    step_size = 1000

    trace_index = 0
    i = 0
    while i < len(signal) - window_size + 1:
        window = signal[i:i + window_size]
        window_mean = rolling_mean[i]
        window_std = rolling_std[i]

        start_index = np.argmax(window > (window_mean * threshold_factor))
        if window[start_index] <= (window_mean * threshold_factor):
            i += step_size
            continue

        if i < highest_peak_global_index + 1000:
            i += step_size
            continue

        end_index = i + start_index + peak_window_size
        if end_index > len(signal):
            end_index = len(signal)
        peak_window = smoothed_signal[i + start_index - 1000:end_index]
        peak_window_t = t[i + start_index - 1000:end_index]

        if len(peak_window) == 0 or len(peak_window_t) == 0:
            i += step_size
            continue

        derivative = np.gradient(peak_window, peak_window_t)
        smoothed_derivative = smooth_signal(derivative, window_size=100)

        peaks = detectar_picos(peak_window, smoothed_derivative)
        if len(peaks) < 2:
            i += step_size
            continue

        sorted_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:2]
        highest_peak_index, highest_peak_value = sorted_peaks[0]
        second_highest_peak_index, second_highest_peak_value = sorted_peaks[1]

        # Verificar la distancia entre los picos
        if not (50 <= abs(highest_peak_index - second_highest_peak_index) <= 1000):
            i += step_size
            continue

        highest_peak_time = peak_window_t[highest_peak_index]
        second_highest_peak_time = peak_window_t[second_highest_peak_index]
        highest_peak_global_index = i + start_index + highest_peak_index

        peak_difference = highest_peak_value - second_highest_peak_value

        init_index = np.argmax(smoothed_derivative > 15000)
        init_time = peak_window_t[init_index - 100]

        distance_km = 0.00032
        time_seconds = highest_peak_time - init_time
        time_hours = time_seconds / 3600
        velocity_kmh = distance_km / time_hours if time_hours > 0 else 0

        post_peak_signal = peak_window[highest_peak_index:]
        if len(post_peak_signal) > 0 and np.any(np.isclose(post_peak_signal, peak_window[init_index], atol=0.1)):
            counter += 1
            data = {
                "numero_de_picos": len(peaks),
                "velocidad_calculada": velocity_kmh,
                "velocidad_real": 0,
                "senal_en_bruto": peak_window.tolist(),
                "senal_suavizada": smoothed_signal.tolist(),
                "senal_derivada": smoothed_derivative.tolist(),
                "pico_mas_alto": highest_peak_value,
                "segundo_pico_mas_alto": second_highest_peak_value,
                "diferencia_de_picos": peak_difference
            }
            save_trace_to_json(data, counter)
            trace_index += 1

            fig, ax1 = plt.subplots(figsize=(12, 8))

            ax1.plot(peak_window_t, peak_window,
                     label='Ventana centrada en el pico', color='b')
            ax1.plot(highest_peak_time, highest_peak_value,
                     'ro', label='Pico más alto')
            ax1.plot(second_highest_peak_time, second_highest_peak_value,
                     'yo', label='Segundo pico más alto')
            ax1.plot(init_time, peak_window[init_index],
                     'go', label='Inicio de la subida')
            ax1.set_xlabel('Tiempo (s)')
            ax1.set_ylabel('Amplitud', color='b')
            ax1.tick_params(axis='y', labelcolor='b')

            ax2 = ax1.twinx()
            ax2.plot(peak_window_t, smoothed_derivative,
                     label='Derivada de la señal', color='r', linestyle='dashed')
            ax2.set_ylabel('Derivada', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            fig.tight_layout()
            fig.suptitle(
                f'Ventana centrada en el pico {highest_peak_index + i}', y=1.02)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax1.grid(True)
            plt.show()

        i = highest_peak_global_index + 500

        if i <= highest_peak_global_index:
            i = highest_peak_global_index + step_size

# Función principal


def main():
    file_path = input("Ingrese la ruta del archivo: ")

    signal = read_data(file_path)

    fs = 10000
    t = np.arange(len(signal)) / fs

    window_size = 4000
    peak_window_size = 1500
    threshold_factor = 1.0025

    process_and_visualize_peaks(
        signal, t, window_size, peak_window_size, threshold_factor)

    plt.figure(figsize=(12, 8))
    plt.plot(t, signal, label='Señal Original')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title('Señal Original')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

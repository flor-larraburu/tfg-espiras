import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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

# Función para guardar la traza en un archivo CSV junto con su velocidad


def save_trace_to_csv(peak_window, peak_index, velocity):
    os.makedirs('db', exist_ok=True)
    csv_file = os.path.join('db', f'traza{peak_index + 1}.csv')
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Escribir la traza
        writer.writerow(peak_window)
        # Escribir la velocidadcd
        writer.writerow([f'Velocidad: {velocity:.2f} km/h'])

# Función para suavizar la señal


def smooth_signal(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')


def process_and_visualize_peaks(signal, t, window_size, peak_window_size, threshold_factor):
    # Asegurarse de que la señal sea float64
    signal = np.asarray(signal, dtype=np.float64)
    # Asegurarse de que el tiempo sea float64
    t = np.asarray(t, dtype=np.float64)

    # Suavizar la señal
    smoothed_signal = smooth_signal(signal, window_size=50)

    rolling_mean, rolling_std = calculate_rolling_stats(
        smoothed_signal, window_size)

    # Inicializar con un valor por debajo de cero menos 1000 muestras
    highest_peak_global_index = -1000
    step_size = 1000  # Tamaño del paso para avanzar en cada iteración

    i = 0
    while i < len(signal) - window_size + 1:

        window = signal[i:i + window_size]
        window_mean = rolling_mean[i]
        window_std = rolling_std[i]

        # Detectar el inicio de la señal cuando se supere la media móvil más la desviación estándar
        start_index = np.argmax(window > (window_mean * threshold_factor))
        if window[start_index] <= (window_mean * threshold_factor):
            i += step_size
            continue

        # Filtrar ventanas que comiencen antes del índice del pico más alto de la última ventana + 1000 muestras
        if i < highest_peak_global_index + 1000:
            i += step_size
            continue

        # Si se detecta un inicio de señal, extraer una ventana  a partir de ese punto
        end_index = i + start_index + peak_window_size
        if end_index > len(signal):
            end_index = len(signal)
        peak_window = smoothed_signal[i + start_index-1000:end_index]
        peak_window_t = t[i + start_index-1000:end_index]

        # Encontrar el pico más alto en la ventana
        peaks, _ = find_peaks(peak_window)
        if len(peaks) == 0:
            i += step_size
            continue

        highest_peak_index = peaks[np.argmax(peak_window[peaks])]
        highest_peak_time = peak_window_t[highest_peak_index]
        highest_peak_global_index = i + start_index + highest_peak_index

        # Derivada de la señal
        derivative = np.gradient(peak_window, peak_window_t)

        # Suavizar la derivada
        smoothed_derivative = smooth_signal(derivative, window_size=100)
        # Detectar el punto donde la derivada comienza a subir hacia el pico
        # Umbral para detectar inicio del incremento
        init_index = np.argmax(smoothed_derivative > 15000)
        init_time = peak_window_t[init_index-100]

        # Calcular la velocidad
        distance_km = 0.00032  # 32 cm in kilometers
        # Δt in seconds desde el inicio de la señal
        time_seconds = highest_peak_time - init_time
        time_hours = time_seconds / 3600  # Convert seconds to hours
        velocity_kmh = distance_km / time_hours if time_hours > 0 else 0
        print(velocity_kmh)

        # Condición para verificar que el nivel de señal después del pico es similar al nivel en init_index en algún momento
        post_peak_signal = peak_window[highest_peak_index:]
        if len(post_peak_signal) > 0 and np.any(np.isclose(post_peak_signal, peak_window[init_index], atol=0.1)):
            fig, ax1 = plt.subplots(figsize=(12, 8))

            # Graficar la señal original
            ax1.plot(peak_window_t, peak_window,
                     label='Ventana centrada en el pico', color='b')
            ax1.plot(highest_peak_time,
                     peak_window[highest_peak_index], 'ro', label='Pico más alto')
            ax1.plot(init_time, peak_window[init_index],
                     'go', label='Inicio de la subida')
            ax1.set_xlabel('Tiempo (s)')
            ax1.set_ylabel('Amplitud', color='b')
            ax1.tick_params(axis='y', labelcolor='b')

            # Crear un segundo eje y para la derivada
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

            # Guardar la traza en un archivo CSV junto con su velocidad
            save_trace_to_csv(
                peak_window, highest_peak_index + i, velocity_kmh)

        # Continuar 500 muestras después del índice del pico más alto
        i = highest_peak_global_index + 500

        # Asegurarse de que el índice no se quede estancado
        if i <= highest_peak_global_index:
            i = highest_peak_global_index + step_size

# Función principal


def main():
    # Solicitar al usuario que ingrese la ruta del archivo
    file_path = input("Ingrese la ruta del archivo: ")

    # Leer los datos del archivo
    signal = read_data(file_path)

    # Configuración de la señal
    fs = 10000  # Frecuencia de muestreo
    t = np.arange(len(signal)) / fs

    # Definir el tamaño de la ventana para calcular la media móvil y la desviación estándar
    window_size = 4000
    peak_window_size = 1500
    threshold_factor = 1.0025  # Factor de umbral

    # Procesar y visualizar los picos
    process_and_visualize_peaks(
        signal, t, window_size, peak_window_size, threshold_factor)

    # Mostrar la señal original
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

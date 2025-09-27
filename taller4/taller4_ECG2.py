import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy import signal
import os

print("TALLER 4: ALGORITMO PAN-TOMPKINS - ECG2")
print("Detección de QRS y análisis de variabilidad cardíaca\n")

# 1. Cargar datos
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_dir, 'ECG1.csv')

data = pd.read_csv(csv_file_path, sep=';')

# Extraer columnas relevantes
tiempo_str = data['Tiempo'].values
ecg_signal = data['ECG 2'].values  # Usamos ECG 2 (señal original, NO filtrada)

print(f"Datos cargados: {len(ecg_signal)} muestras")

# Convertir timestamps a segundos
# Extraer solo la parte de tiempo (sin fecha)
tiempo_parts = [t.split(' ')[-1] for t in tiempo_str]
# Convertir a segundos desde el primer timestamp
tiempo_sec = []
base_time = None

for i, t in enumerate(tiempo_parts):
    h, m, s = t.split(':')
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)

    if i == 0:
        base_time = total_seconds

    tiempo_sec.append(total_seconds - base_time)

tiempo_sec = np.array(tiempo_sec)

# Calcular frecuencia de muestreo real
dt_mean = np.mean(np.diff(tiempo_sec))
fs_real = 1 / dt_mean
print(f"Frecuencia de muestreo calculada: {fs_real:.1f} Hz")
print(f"Duración total: {tiempo_sec[-1]:.1f} segundos")

# Usar Fs=256Hz como especifica el taller, pero mostrar la real para comparación
Fs = 256.0
print(f"Usando Fs = {Fs} Hz para el análisis (según especificación del taller)")

# Crear vector de tiempo uniforme basado en Fs=256Hz
N = len(ecg_signal)
vtime = np.arange(0, N/Fs, 1/Fs)
if len(vtime) > N:
    vtime = vtime[:N]

# Graficar señal original
fig1, ax = plt.subplots(1, 1, figsize=(15, 6))
ax.plot(vtime, ecg_signal, 'b-', linewidth=0.8)
ax.set_title('Señal ECG Original - ECG2')
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Amplitud (μV)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

print("\nIMPLEMENTACIÓN ALGORITMO PAN-TOMPKINS")

# PASO 1: Filtro paso-banda 5-15Hz
lowcut = 5.0
highcut = 15.0
nyquist = Fs / 2
low = lowcut / nyquist
high = highcut / nyquist
order = 4

b, a = butter(order, [low, high], btype='band')
ecg_filtered = filtfilt(b, a, ecg_signal)

# PASO 2: Diferenciación (pendiente de 4 muestras)
# H(z) = 0.1(2 + z^-1 - z^-3 - 2z^-4)
# En forma de coeficientes: b = [0.2, 0.1, 0, -0.1, -0.2], a = [1]
diff_coeffs = np.array([0.2, 0.1, 0, -0.1, -0.2])
ecg_diff = signal.lfilter(diff_coeffs, [1], ecg_filtered)

# PASO 3: Cuadrado
ecg_squared = ecg_diff ** 2

# PASO 4: Integración con ventana móvil
window_samples = int(0.15 * Fs)  # 150ms en muestras
print(f"  - Ventana de integración: {window_samples} muestras ({window_samples/Fs*1000:.0f} ms)")

# Crear ventana de integración (todos unos)
integration_window = np.ones(window_samples) / window_samples
ecg_integrated = signal.lfilter(integration_window, [1], ecg_squared)


# Graficar todos los pasos del algoritmo Pan-Tompkins
fig2, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)

# Señal original
axes[0].plot(vtime, ecg_signal, 'b-', linewidth=0.8)
axes[0].set_title('Señal original')
axes[0].set_ylabel('Amplitud (μV)')
axes[0].grid(True, alpha=0.3)

# Filtro paso-banda
axes[1].plot(vtime, ecg_filtered, 'g-', linewidth=0.8)
axes[1].set_title('Filtro paso-banda\n5-15Hz')
axes[1].set_ylabel('Amplitud (μV)')
axes[1].grid(True, alpha=0.3)

# Diferenciación
axes[2].plot(vtime, ecg_diff, 'r-', linewidth=0.8)
axes[2].set_title('Diferenciación:\npendiente 4 muestras\nH(z) = 0.1(2 + z⁻¹ - z⁻³ - 2z⁻⁴)')
axes[2].set_ylabel('Amplitud')
axes[2].grid(True, alpha=0.3)

# Cuadrado
axes[3].plot(vtime, ecg_squared, 'm-', linewidth=0.8)
axes[3].set_title('Cuadrado')
axes[3].set_ylabel('Amplitud²')
axes[3].grid(True, alpha=0.3)

# Integración
axes[4].plot(vtime, ecg_integrated, 'c-', linewidth=0.8)
axes[4].set_title('Integración\nventana de 150ms')
axes[4].set_xlabel('Tiempo (s)')
axes[4].set_ylabel('Amplitud integrada')
axes[4].grid(True, alpha=0.3)

plt.suptitle('Algoritmo Pan-Tompkins - ECG2 (Todos los pasos)', fontsize=14)
plt.tight_layout()
plt.show(block=False)

# PASO 5: Umbralización estadística y detección de picos R
print("\nDETECCIÓN DE PICOS R")

# Calcular umbral adaptativo
mean_integrated = np.mean(ecg_integrated)
std_integrated = np.std(ecg_integrated)

# Umbral adaptativo más sofisticado
# Usar percentil en lugar de máximo para evitar outliers
threshold_factor = 0.5  # Factor para el umbral
percentile_95 = np.percentile(ecg_integrated, 95)
threshold = threshold_factor * percentile_95

# Umbral alternativo basado en estadísticas
threshold_stat = mean_integrated + 3 * std_integrated

# Usar el menor de los dos umbrales para ser más sensible
threshold = min(threshold, threshold_stat)

print(f"Estadísticas de la señal integrada:")
print(f"  - Media: {mean_integrated:.4f}")
print(f"  - Desviación estándar: {std_integrated:.4f}")
print(f"  - Percentil 95: {percentile_95:.4f}")
print(f"Umbral usado: {threshold:.4f}")
print(f"  - Umbral por percentil: {threshold_factor * percentile_95:.4f}")
print(f"  - Umbral estadístico: {threshold_stat:.4f}")

# Detectar picos R usando find_peaks
# Distancia mínima entre picos R: ~0.4s (150 bpm máximo) = 0.4 * Fs muestras
min_distance = int(0.4 * Fs)  # Mínimo 400ms entre picos R

peaks_R, properties = find_peaks(ecg_integrated,
                                height=threshold,
                                distance=min_distance,
                                prominence=std_integrated)

print(f"Picos R detectados: {len(peaks_R)}")
print(f"Distancia mínima entre picos: {min_distance} muestras ({min_distance/Fs:.2f} s)")

# Calcular intervalos RR (en segundos)
if len(peaks_R) > 1:
    RR_intervals = np.diff(peaks_R) / Fs  # Convertir a segundos
    print(f"Intervalos RR calculados: {len(RR_intervals)} intervalos")
    print(f"  - RR promedio: {np.mean(RR_intervals):.3f} s")
    print(f"  - RR mínimo: {np.min(RR_intervals):.3f} s")
    print(f"  - RR máximo: {np.max(RR_intervals):.3f} s")
else:
    RR_intervals = np.array([])
    print("No se detectaron suficientes picos R para calcular intervalos")

# Graficar detección de picos R
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Señal integrada con picos R detectados
ax1.plot(vtime, ecg_integrated, 'c-', linewidth=1, label='Señal integrada')
ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Umbral ({threshold:.3f})')
ax1.plot(vtime[peaks_R], ecg_integrated[peaks_R], 'ro', markersize=8,
         label=f'Picos R detectados ({len(peaks_R)})')
ax1.set_title('Detección de Picos R - Señal Integrada')
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('Amplitud integrada')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Señal ECG original con picos R marcados
ax2.plot(vtime, ecg_signal, 'b-', linewidth=0.8, alpha=0.7, label='ECG original')
ax2.plot(vtime[peaks_R], ecg_signal[peaks_R], 'ro', markersize=6,
         label=f'Picos R ({len(peaks_R)})')
ax2.set_title('Picos R en Señal ECG Original')
ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('Amplitud (μV)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=False)

# PASO 6: Detectar ondas P, Q, S, T
print("\nDETECCIÓN DE ONDAS P, Q, S, T")

# Ventana de búsqueda: ±400ms alrededor del pico R
window_ms = 400  # milisegundos
window_samples = int(window_ms * Fs / 1000)
print(f"Ventana de búsqueda: ±{window_ms} ms = ±{window_samples} muestras")

# Listas para almacenar las detecciones
P_peaks = []
Q_peaks = []
S_peaks = []
T_peaks = []

for i, R_peak in enumerate(peaks_R):
    # Definir ventanas de búsqueda
    start_idx = max(0, R_peak - window_samples)
    end_idx = min(len(ecg_signal), R_peak + window_samples)

    # Extraer segmento ECG alrededor del pico R
    segment = ecg_signal[start_idx:end_idx]
    segment_indices = np.arange(start_idx, end_idx)

    # Buscar P: antes del R (máximo en la primera mitad)
    pre_R_end = R_peak - start_idx
    if pre_R_end > 10:  # Si hay suficientes muestras antes del R
        pre_R_segment = segment[:pre_R_end]
        pre_R_indices = segment_indices[:pre_R_end]

        # P es un máximo antes de R
        if len(pre_R_segment) > 0:
            P_idx_local = np.argmax(pre_R_segment[:-10])  # Evitar estar muy cerca de R
            P_idx_global = pre_R_indices[P_idx_local]
            P_peaks.append(P_idx_global)

        # Q es un mínimo justo antes de R
        Q_search_start = max(0, pre_R_end - 20)  # Buscar en los últimos 20 puntos antes de R
        Q_segment = segment[Q_search_start:pre_R_end]
        if len(Q_segment) > 0:
            Q_idx_local = np.argmin(Q_segment)
            Q_idx_global = segment_indices[Q_search_start + Q_idx_local]
            Q_peaks.append(Q_idx_global)

    # Buscar S y T: después del R
    post_R_start = R_peak - start_idx
    if post_R_start < len(segment) - 10:
        post_R_segment = segment[post_R_start:]
        post_R_indices = segment_indices[post_R_start:]

        # S es un mínimo justo después de R
        S_search_end = min(len(post_R_segment), 20)  # Buscar en los primeros 20 puntos después de R
        S_segment = post_R_segment[1:S_search_end]  # Empezar después del pico R
        if len(S_segment) > 0:
            S_idx_local = np.argmin(S_segment)
            S_idx_global = post_R_indices[1 + S_idx_local]
            S_peaks.append(S_idx_global)

        # T es un máximo después de S (en la segunda mitad)
        if len(post_R_segment) > 20:
            T_segment = post_R_segment[20:]  # Buscar después de donde podría estar S
            T_indices = post_R_indices[20:]
            if len(T_segment) > 0:
                T_idx_local = np.argmax(T_segment)
                T_idx_global = T_indices[T_idx_local]
                T_peaks.append(T_idx_global)

print(f"Ondas detectadas:")
print(f"  - Ondas P: {len(P_peaks)}")
print(f"  - Ondas Q: {len(Q_peaks)}")
print(f"  - Ondas R: {len(peaks_R)}")
print(f"  - Ondas S: {len(S_peaks)}")
print(f"  - Ondas T: {len(T_peaks)}")

# Graficar detección de todas las ondas
fig4, ax = plt.subplots(1, 1, figsize=(16, 8))

# Mostrar solo un segmento representativo para mejor visualización
start_time = 10  # segundos
end_time = 20    # segundos
start_idx = int(start_time * Fs)
end_idx = int(end_time * Fs)

time_segment = vtime[start_idx:end_idx]
ecg_segment = ecg_signal[start_idx:end_idx]

ax.plot(time_segment, ecg_segment, 'b-', linewidth=1.5, label='ECG')

# Filtrar picos que están en el segmento mostrado
def plot_peaks_in_segment(peaks, color, marker, label):
    segment_peaks = [p for p in peaks if start_idx <= p < end_idx]
    if segment_peaks:
        ax.plot(vtime[segment_peaks], ecg_signal[segment_peaks],
               marker, color=color, markersize=8, label=f'{label} ({len(segment_peaks)})')

plot_peaks_in_segment(P_peaks, 'green', 'o', 'P')
plot_peaks_in_segment(Q_peaks, 'orange', 's', 'Q')
plot_peaks_in_segment(peaks_R, 'red', '^', 'R')
plot_peaks_in_segment(S_peaks, 'purple', 'v', 'S')
plot_peaks_in_segment(T_peaks, 'cyan', 'd', 'T')

ax.set_title(f'Detección de Ondas PQRST - ECG2 (Segmento {start_time}-{end_time}s)')
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Amplitud (μV)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

# PASO 7: Análisis de Variabilidad Cardíaca (HRV) y Frecuencia Cardíaca
print("\nANÁLISIS DE VARIABILIDAD CARDÍACA (HRV)")

if len(RR_intervals) > 0:
    # Calcular frecuencia cardíaca instantánea
    HR_instantaneous = 60 / RR_intervals  # bpm

    # Calcular estadísticas
    HR_mean = np.mean(HR_instantaneous)
    HR_std = np.std(HR_instantaneous)
    HR_min = np.min(HR_instantaneous)
    HR_max = np.max(HR_instantaneous)

    # Frecuencia cardíaca global
    total_time = vtime[-1]  # Tiempo total en segundos
    total_beats = len(peaks_R)
    HR_global = (total_beats / total_time) * 60  # bpm

    print(f"Análisis de Frecuencia Cardíaca:")
    print(f"  - Frecuencia cardíaca global: {HR_global:.1f} bpm")
    print(f"  - FC promedio (de intervalos RR): {HR_mean:.1f} bpm")
    print(f"  - FC mínima: {HR_min:.1f} bpm")
    print(f"  - FC máxima: {HR_max:.1f} bpm")
    print(f"  - Desviación estándar FC: {HR_std:.1f} bpm")

    print(f"Análisis de Intervalos RR:")
    print(f"  - RMSSD: {np.sqrt(np.mean(np.diff(RR_intervals)**2))*1000:.1f} ms")
    print(f"  - SDNN: {np.std(RR_intervals)*1000:.1f} ms")

    # Crear tacograma (HRV)
    fig5, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Intervalos RR
    RR_times = vtime[peaks_R[1:]]  # Tiempo de cada intervalo RR
    ax1.plot(RR_times, RR_intervals*1000, 'bo-', linewidth=1, markersize=4)
    ax1.set_title('Tacograma - Intervalos RR')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Intervalo RR (ms)')
    ax1.grid(True, alpha=0.3)

    # Frecuencia cardíaca instantánea
    ax2.plot(RR_times, HR_instantaneous, 'ro-', linewidth=1, markersize=4)
    ax2.axhline(y=HR_global, color='g', linestyle='--', linewidth=2,
               label=f'FC Global: {HR_global:.1f} bpm')
    ax2.set_title('Frecuencia Cardíaca Instantánea')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Frecuencia Cardíaca (bpm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Análisis de Variabilidad Cardíaca (HRV) - ECG2')
    plt.tight_layout()
    plt.show(block=False)

else:
    print("No se pudieron calcular métricas de HRV - insuficientes picos R detectados")

print("\nRESUMEN FINAL")
print(f"Señal procesada: ECG2")
print(f"Duración total: {vtime[-1]:.1f} segundos")
print(f"Picos R detectados: {len(peaks_R)}")
if len(RR_intervals) > 0:
    print(f"Frecuencia cardíaca global: {HR_global:.1f} bpm")
    print(f"Intervalos RR analizados: {len(RR_intervals)}")

print("\nAnálisis Pan-Tompkins completado para ECG2")
plt.show()
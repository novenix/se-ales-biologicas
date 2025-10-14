#!/usr/bin/env python3
"""
TALLER 6 - ANALISIS DE SENALES EEG
Analisis de senales de EEG en dominio del tiempo y frecuencia (FFT)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import fft
import os

print("=" * 60)
print("TALLER 6: ANALISIS DE SENALES EEG")
print("=" * 60)

# Cargar datos
current_dir = os.path.dirname(os.path.abspath(__file__))
file1_path = os.path.join(current_dir, 'FileEEG.mat')
file2_path = os.path.join(current_dir, 'sEEG.mat')

print("\n1. CARGANDO ARCHIVOS .MAT")
print("-" * 60)

# Cargar FileEEG.mat
print(f"Cargando: {file1_path}")
data1 = loadmat(file1_path)
print(f"  Variables disponibles: {[k for k in data1.keys() if not k.startswith('__')]}")

# Cargar sEEG.mat
print(f"Cargando: {file2_path}")
data2 = loadmat(file2_path)
print(f"  Variables disponibles: {[k for k in data2.keys() if not k.startswith('__')]}")

# Extraer senales - buscar claves especificas primero
def extract_signal(data, filename):
    """Extrae la senal del diccionario de datos .mat"""
    # Buscar claves tipicas de senales EEG
    signal_keys = ['SigEEG', 'm_Sig', 'data', 'signal', 'eeg']

    for key in signal_keys:
        if key in data:
            signal = data[key]
            if isinstance(signal, np.ndarray) and signal.size > 100:
                # Aplanar si es necesario
                if signal.ndim > 1:
                    if signal.shape[0] == 1:
                        signal = signal.flatten()
                    elif signal.shape[1] == 1:
                        signal = signal.flatten()
                    else:
                        signal = signal[:, 0]
                print(f"  -> Senal '{key}' en {filename}: shape={signal.shape}, dtype={signal.dtype}")
                return signal, key

    # Si no encontro nada, buscar cualquier array grande
    for key in data.keys():
        if not key.startswith('__'):
            signal = data[key]
            if isinstance(signal, np.ndarray) and signal.size > 100:
                if signal.ndim > 1:
                    if signal.shape[0] == 1:
                        signal = signal.flatten()
                    elif signal.shape[1] == 1:
                        signal = signal.flatten()
                    else:
                        signal = signal[:, 0]
                print(f"  -> Senal '{key}' en {filename}: shape={signal.shape}, dtype={signal.dtype}")
                return signal, key
    return None, None

# Extraer Fs (frecuencia de muestreo)
Fs1 = data1['Fs'][0, 0] if 'Fs' in data1 else 256
Fs2 = data2['Fs'][0, 0] if 'Fs' in data2 else 256

signal1, name1 = extract_signal(data1, 'FileEEG.mat')
signal2, name2 = extract_signal(data2, 'sEEG.mat')

if signal1 is None or signal2 is None:
    print("\nError: No se pudo extraer la senal de los archivos")
    exit(1)

L1 = len(signal1)
L2 = len(signal2)
time1 = np.arange(L1) / Fs1
time2 = np.arange(L2) / Fs2

print(f"\n2. INFORMACION DE LAS SENALES")
print("-" * 60)
print(f"FileEEG.mat ({name1}):")
print(f"  - Muestras originales: {L1}")
print(f"  - Duracion: {L1/Fs1:.2f} s")
print(f"  - Fs: {Fs1} Hz")
print(f"  - Rango: [{np.min(signal1):.4f}, {np.max(signal1):.4f}]")
print(f"  - Media: {np.mean(signal1):.4f}")
print(f"  - Desv. Est: {np.std(signal1):.4f}")

print(f"\nsEEG.mat ({name2}):")
print(f"  - Muestras originales: {L2}")
print(f"  - Duracion: {L2/Fs2:.2f} s")
print(f"  - Fs: {Fs2} Hz")
print(f"  - Rango: [{np.min(signal2):.4f}, {np.max(signal2):.4f}]")
print(f"  - Media: {np.mean(signal2):.4f}")
print(f"  - Desv. Est: {np.std(signal2):.4f}")

# Diezmado para graficas mas ligeras (reducir a ~200 puntos)
downsample_factor1 = max(1, L1 // 200)
downsample_factor2 = max(1, L2 // 200)

signal1_decimated = signal1[::downsample_factor1]
signal2_decimated = signal2[::downsample_factor2]
time1_decimated = time1[::downsample_factor1]
time2_decimated = time2[::downsample_factor2]

print(f"\nDiezmado para graficacion:")
print(f"  - FileEEG.mat: {L1} -> {len(signal1_decimated)} puntos (factor: {downsample_factor1})")
print(f"  - sEEG.mat: {L2} -> {len(signal2_decimated)} puntos (factor: {downsample_factor2})")

# ============================================================================
# 3. ANALISIS FFT - Siguiendo el ejemplo de Matlab del taller.txt
# ============================================================================

def compute_single_sided_spectrum(signal, Fs):
    """
    Calcula el espectro de amplitud de un solo lado (single-sided)
    segun el ejemplo de Matlab del taller.txt
    """
    L = len(signal)

    # FFT
    Y = fft(signal)

    # Normalizar
    P2 = np.abs(Y / L)

    # Tomar la mitad del espectro
    P1 = P2[0:L//2 + 1]

    # Multiplicar por 2 la mitad del espectro (excepto DC y Nyquist)
    P1[1:-1] = 2 * P1[1:-1]

    # Vector de frecuencias
    f = Fs * np.arange(0, L//2 + 1) / L

    return f, P1

print("\n3. CALCULANDO FFT")
print("-" * 60)

# Calcular FFT para ambas senales
f1, P1 = compute_single_sided_spectrum(signal1, Fs1)
f2, P2 = compute_single_sided_spectrum(signal2, Fs2)

print(f"FFT calculada para FileEEG.mat")
print(f"  - Frecuencias: 0 - {f1[-1]:.2f} Hz")
print(f"  - Resolucion: {f1[1] - f1[0]:.4f} Hz")

print(f"FFT calculada para sEEG.mat")
print(f"  - Frecuencias: 0 - {f2[-1]:.2f} Hz")
print(f"  - Resolucion: {f2[1] - f2[0]:.4f} Hz")

# Identificar frecuencias dominantes
def find_dominant_frequencies(f, P, n_peaks=5, min_freq=0.5):
    """Encuentra las frecuencias dominantes en el espectro"""
    # Ignorar DC y frecuencias muy bajas
    mask = f >= min_freq
    f_filtered = f[mask]
    P_filtered = P[mask]

    # Encontrar los picos mas altos
    sorted_indices = np.argsort(P_filtered)[::-1]
    top_freqs = f_filtered[sorted_indices[:n_peaks]]
    top_amps = P_filtered[sorted_indices[:n_peaks]]

    return top_freqs, top_amps

print("\n4. FRECUENCIAS DOMINANTES")
print("-" * 60)

top_freqs1, top_amps1 = find_dominant_frequencies(f1, P1)
print(f"FileEEG.mat - Top 5 frecuencias:")
for i, (freq, amp) in enumerate(zip(top_freqs1, top_amps1), 1):
    print(f"  {i}. {freq:.2f} Hz -> amplitud: {amp:.6f}")

top_freqs2, top_amps2 = find_dominant_frequencies(f2, P2)
print(f"\nsEEG.mat - Top 5 frecuencias:")
for i, (freq, amp) in enumerate(zip(top_freqs2, top_amps2), 1):
    print(f"  {i}. {freq:.2f} Hz -> amplitud: {amp:.6f}")

# ============================================================================
# 5. VISUALIZACION
# ============================================================================

print("\n5. GENERANDO GRAFICAS")
print("-" * 60)

# ============================================================================
# FIGURA 1: Todos los puntos
# ============================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Analisis de Senales EEG - TODOS LOS PUNTOS',
              fontsize=14, fontweight='bold')

# ---- FileEEG.mat ----
# Dominio del tiempo (todos los puntos)
axes1[0, 0].plot(time1, signal1, 'b-', linewidth=0.5)
axes1[0, 0].set_title(f'FileEEG.mat - Dominio del Tiempo ({L1} puntos)\n({name1})')
axes1[0, 0].set_xlabel('Tiempo (s)')
axes1[0, 0].set_ylabel('Amplitud')
axes1[0, 0].grid(True, alpha=0.3)

# Dominio de la frecuencia (FFT)
axes1[0, 1].plot(f1, P1, 'b-', linewidth=1)
axes1[0, 1].set_title(f'FileEEG.mat - Espectro de Frecuencia (FFT)\n({name1})')
axes1[0, 1].set_xlabel('Frecuencia (Hz)')
axes1[0, 1].set_ylabel('|P1(f)|')
axes1[0, 1].grid(True, alpha=0.3)
axes1[0, 1].set_xlim([0, 60])

# Marcar frecuencias dominantes
for freq in top_freqs1[:3]:
    axes1[0, 1].axvline(x=freq, color='r', linestyle='--', alpha=0.5, linewidth=1)

# ---- sEEG.mat ----
# Dominio del tiempo (todos los puntos)
axes1[1, 0].plot(time2, signal2, 'g-', linewidth=0.5)
axes1[1, 0].set_title(f'sEEG.mat - Dominio del Tiempo ({L2} puntos)\n({name2})')
axes1[1, 0].set_xlabel('Tiempo (s)')
axes1[1, 0].set_ylabel('Amplitud')
axes1[1, 0].grid(True, alpha=0.3)

# Dominio de la frecuencia (FFT)
axes1[1, 1].plot(f2, P2, 'g-', linewidth=1)
axes1[1, 1].set_title(f'sEEG.mat - Espectro de Frecuencia (FFT)\n({name2})')
axes1[1, 1].set_xlabel('Frecuencia (Hz)')
axes1[1, 1].set_ylabel('|P1(f)|')
axes1[1, 1].grid(True, alpha=0.3)
axes1[1, 1].set_xlim([0, 60])

# Marcar frecuencias dominantes
for freq in top_freqs2[:3]:
    axes1[1, 1].axvline(x=freq, color='r', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()

# Guardar figura 1
output_path1 = os.path.join(current_dir, 'taller6_analisis_eeg_completo.png')
fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"Grafica (todos los puntos) guardada: {output_path1}")

# ============================================================================
# FIGURA 2: 200 puntos diezmados
# ============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Analisis de Senales EEG - DIEZMADO (~200 PUNTOS)',
              fontsize=14, fontweight='bold')

# ---- FileEEG.mat ----
# Dominio del tiempo (diezmado)
axes2[0, 0].plot(time1_decimated, signal1_decimated, 'b-', linewidth=0.5, marker='o', markersize=2)
axes2[0, 0].set_title(f'FileEEG.mat - Dominio del Tiempo ({len(signal1_decimated)} puntos)\n({name1})')
axes2[0, 0].set_xlabel('Tiempo (s)')
axes2[0, 0].set_ylabel('Amplitud')
axes2[0, 0].grid(True, alpha=0.3)

# Dominio de la frecuencia (FFT)
axes2[0, 1].plot(f1, P1, 'b-', linewidth=1)
axes2[0, 1].set_title(f'FileEEG.mat - Espectro de Frecuencia (FFT)\n({name1})')
axes2[0, 1].set_xlabel('Frecuencia (Hz)')
axes2[0, 1].set_ylabel('|P1(f)|')
axes2[0, 1].grid(True, alpha=0.3)
axes2[0, 1].set_xlim([0, 60])

# Marcar frecuencias dominantes
for freq in top_freqs1[:3]:
    axes2[0, 1].axvline(x=freq, color='r', linestyle='--', alpha=0.5, linewidth=1)

# ---- sEEG.mat ----
# Dominio del tiempo (diezmado)
axes2[1, 0].plot(time2_decimated, signal2_decimated, 'g-', linewidth=0.5, marker='o', markersize=2)
axes2[1, 0].set_title(f'sEEG.mat - Dominio del Tiempo ({len(signal2_decimated)} puntos)\n({name2})')
axes2[1, 0].set_xlabel('Tiempo (s)')
axes2[1, 0].set_ylabel('Amplitud')
axes2[1, 0].grid(True, alpha=0.3)

# Dominio de la frecuencia (FFT)
axes2[1, 1].plot(f2, P2, 'g-', linewidth=1)
axes2[1, 1].set_title(f'sEEG.mat - Espectro de Frecuencia (FFT)\n({name2})')
axes2[1, 1].set_xlabel('Frecuencia (Hz)')
axes2[1, 1].set_ylabel('|P1(f)|')
axes2[1, 1].grid(True, alpha=0.3)
axes2[1, 1].set_xlim([0, 60])

# Marcar frecuencias dominantes
for freq in top_freqs2[:3]:
    axes2[1, 1].axvline(x=freq, color='r', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()

# Guardar figura 2
output_path2 = os.path.join(current_dir, 'taller6_analisis_eeg_diezmado.png')
fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Grafica (diezmado) guardada: {output_path2}")

plt.show()

# ============================================================================
# 6. ANALISIS DE BANDAS DE FRECUENCIA EEG
# ============================================================================

print("\n6. ANALISIS POR BANDAS DE FRECUENCIA EEG")
print("-" * 60)

# Bandas tipicas de EEG
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 60)
}

def analyze_frequency_bands(f, P, bands):
    """Calcula la potencia en cada banda de frecuencia"""
    band_powers = {}
    for band_name, (low, high) in bands.items():
        mask = (f >= low) & (f <= high)
        power = np.sum(P[mask] ** 2)  # Potencia como suma de amplitudes al cuadrado
        band_powers[band_name] = power
    return band_powers

# Analizar FileEEG.mat
print("FileEEG.mat - Potencia por banda:")
band_powers1 = analyze_frequency_bands(f1, P1, bands)
total_power1 = sum(band_powers1.values())
for band, power in band_powers1.items():
    percentage = (power / total_power1) * 100
    print(f"  {band:6s}: {power:.6f} ({percentage:.2f}%)")

print("\nsEEG.mat - Potencia por banda:")
band_powers2 = analyze_frequency_bands(f2, P2, bands)
total_power2 = sum(band_powers2.values())
for band, power in band_powers2.items():
    percentage = (power / total_power2) * 100
    print(f"  {band:6s}: {power:.6f} ({percentage:.2f}%)")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 60)
print("ANALISIS COMPLETADO")
print("=" * 60)
print("* Senales cargadas y graficadas")
print("* FFT calculada segun ejemplo de Matlab")
print("* Frecuencias dominantes identificadas")
print("* Analisis por bandas EEG completado")
print("* Grafica guardada en: taller6_analisis_eeg.png")
print("\nLas graficas muestran:")
print("  - Senales originales en dominio del tiempo")
print("  - Espectros de frecuencia (single-sided) con FFT")
print("  - Frecuencias dominantes marcadas en rojo")

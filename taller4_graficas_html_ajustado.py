#!/usr/bin/env python3
"""
VERSIÓN AJUSTADA - GENERADOR DE GRÁFICAS HTML PARA TALLER 4
Mejoras en la detección de ondas PQRST
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

print("GENERANDO GRÁFICAS HTML PARA TALLER 4 - VERSIÓN AJUSTADA")
print("=========================================================\n")

# 1. CARGAR DATOS
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_dir, 'taller4/ECG1.csv')

data = pd.read_csv(csv_file_path, sep=';')
tiempo_str = data['Tiempo'].values
ecg_signal = data['ECG 2'].values

# Convertir timestamps a segundos
tiempo_parts = [t.split(' ')[-1] for t in tiempo_str]
tiempo_sec = []
base_time = None

for i, t in enumerate(tiempo_parts):
    h, m, s = t.split(':')
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    if i == 0:
        base_time = total_seconds
    tiempo_sec.append(total_seconds - base_time)

tiempo_sec = np.array(tiempo_sec)

# Parámetros
Fs = 256.0
N = len(ecg_signal)
vtime = np.arange(0, N/Fs, 1/Fs)
if len(vtime) > N:
    vtime = vtime[:N]

duracion_total = vtime[-1]
print(f"Datos cargados: {len(ecg_signal)} muestras")
print(f"Duración: {duracion_total:.1f} segundos")
print(f"Frecuencia de muestreo: {Fs} Hz")

# 2. PREPROCESAMIENTO - Eliminar tendencia
print("\nPreprocesando señal...")
# Eliminar offset DC
ecg_signal = ecg_signal - np.mean(ecg_signal)

# Filtro paso alto para eliminar deriva baseline
b_hp, a_hp = butter(2, 0.5/Fs*2, btype='highpass')
ecg_signal = filtfilt(b_hp, a_hp, ecg_signal)

# 3. ALGORITMO PAN-TOMPKINS AJUSTADO
print("Aplicando algoritmo Pan-Tompkins ajustado...")

# PASO 1: Filtro paso-banda más conservador
lowcut = 5.0
highcut = 15.0  # Reducido de 35 a 15 Hz para menos ruido
nyquist = Fs / 2
low = lowcut / nyquist
high = highcut / nyquist
order = 2  # Reducido de 4 a 2 para menos atenuación

b, a = butter(order, [low, high], btype='band')
ecg_filtered = filtfilt(b, a, ecg_signal)

# PASO 2: Diferenciación
diff_coeffs = np.array([0.2, 0.1, 0, -0.1, -0.2])
ecg_diff = signal.lfilter(diff_coeffs, [1], ecg_filtered)

# PASO 3: Cuadrado
ecg_squared = ecg_diff ** 2

# PASO 4: Integración - ventana más grande
window_ms = 150  # Aumentado de 80ms a 150ms
window_samples = int(window_ms * Fs / 1000)
integration_window = np.ones(window_samples) / window_samples
ecg_integrated = signal.lfilter(integration_window, [1], ecg_squared)

print(f"  - Filtro paso-banda: {lowcut}-{highcut} Hz")
print(f"  - Ventana de integración: {window_ms} ms")

# 4. DETECCIÓN DE PICOS R MEJORADA
print("\nDetectando picos R con umbral adaptativo...")

# Normalizar señal integrada
ecg_integrated_norm = ecg_integrated / np.max(ecg_integrated)

# Calcular umbral adaptativo mejorado
mean_integrated = np.mean(ecg_integrated_norm)
std_integrated = np.std(ecg_integrated_norm)

# Usar percentil más alto para ser más selectivo
percentile_threshold = np.percentile(ecg_integrated_norm, 99)  # Aumentado a 99
threshold = 0.3 * percentile_threshold  # Aumentado de 0.05 a 0.3

# Umbral mínimo basado en estadísticas
threshold_stat = mean_integrated + 1.5 * std_integrated  # Aumentado de 0.5 a 1.5

# Usar el MAYOR de los dos umbrales (más restrictivo)
threshold = max(threshold, threshold_stat)

print(f"  - Umbral calculado: {threshold:.3f}")
print(f"  - Media normalizada: {mean_integrated:.3f}")
print(f"  - Desv. estándar normalizada: {std_integrated:.3f}")

# Detectar picos R con parámetros más estrictos
min_distance = int(0.6 * Fs)  # Aumentado de 0.35s a 0.6s (100 bpm máximo)

peaks_R, properties = find_peaks(ecg_integrated_norm,
                                height=threshold,
                                distance=min_distance,
                                prominence=threshold/2)  # Prominencia más estricta

# Filtrar picos por altura relativa
if len(peaks_R) > 0:
    peak_heights = ecg_integrated_norm[peaks_R]
    mean_height = np.mean(peak_heights)

    # Mantener solo picos que sean al menos 50% de la altura media
    valid_peaks = peaks_R[peak_heights > 0.5 * mean_height]
    peaks_R = valid_peaks

print(f"Picos R detectados en señal integrada: {len(peaks_R)}")

# CORRECCIÓN: Ajustar picos R a los máximos reales en ecg_filtered
# Los picos en ecg_integrated_norm están desplazados por el procesamiento
peaks_R_adjusted = []
search_window = int(0.08 * Fs)  # Buscar ±80ms alrededor

for peak in peaks_R:
    start_idx = max(0, peak - search_window)
    end_idx = min(len(ecg_filtered), peak + search_window)

    # Buscar el máximo REAL en la señal filtrada
    segment = ecg_filtered[start_idx:end_idx]
    max_idx_local = np.argmax(segment)
    max_idx_global = start_idx + max_idx_local

    peaks_R_adjusted.append(max_idx_global)

peaks_R_adjusted = np.array(peaks_R_adjusted)
peaks_R_original = peaks_R.copy()  # Guardar originales para gráfica de integrada
peaks_R = peaks_R_adjusted  # Usar ajustados para el resto

print(f"✓ Picos R ajustados a máximos reales: {len(peaks_R)}")

# Calcular frecuencia cardíaca esperada
if len(peaks_R) > 1:
    RR_intervals = np.diff(peaks_R) / Fs
    HR_mean = 60 / np.mean(RR_intervals)
    print(f"Frecuencia cardíaca promedio: {HR_mean:.1f} bpm")

    # Verificación de sentido
    if HR_mean < 40 or HR_mean > 200:
        print(f"⚠️  ADVERTENCIA: FC fuera de rango normal ({HR_mean:.1f} bpm)")
else:
    RR_intervals = np.array([])
    HR_mean = 0

# 5. DETECCIÓN DE ONDAS PQST MEJORADA
print("\nDetectando ondas PQST...")

P_peaks = []
Q_peaks = []
S_peaks = []
T_peaks = []

# Solo detectar PQST si tenemos una cantidad razonable de picos R
if 10 < len(peaks_R) < 500:  # Rango razonable de latidos

    for i, R_peak in enumerate(peaks_R):
        # Ventanas de búsqueda más específicas

        # Ventana para P: 50-200ms antes de R
        P_start = max(0, R_peak - int(0.2 * Fs))
        P_end = max(0, R_peak - int(0.05 * Fs))

        if P_start < P_end:
            P_segment = ecg_filtered[P_start:P_end]
            if len(P_segment) > 0:
                # P es un máximo local
                P_idx = np.argmax(P_segment)
                if P_segment[P_idx] > 0:  # Solo si es positivo
                    P_peaks.append(P_start + P_idx)

        # Ventana para Q: 20-50ms antes de R
        Q_start = max(0, R_peak - int(0.05 * Fs))
        Q_end = R_peak

        if Q_start < Q_end:
            Q_segment = ecg_filtered[Q_start:Q_end]
            if len(Q_segment) > 0:
                # Q es un mínimo local justo antes de R
                Q_idx = np.argmin(Q_segment)
                if Q_segment[Q_idx] < ecg_filtered[R_peak] * 0.5:  # Solo si es menor que R
                    Q_peaks.append(Q_start + Q_idx)

        # Ventana para S: 20-50ms después de R
        S_start = R_peak
        S_end = min(len(ecg_filtered), R_peak + int(0.05 * Fs))

        if S_start < S_end:
            S_segment = ecg_filtered[S_start:S_end]
            if len(S_segment) > 0:
                # S es un mínimo local justo después de R
                S_idx = np.argmin(S_segment)
                if S_segment[S_idx] < ecg_filtered[R_peak] * 0.5:  # Solo si es menor que R
                    S_peaks.append(S_start + S_idx)

        # Ventana para T: 100-300ms después de R
        T_start = R_peak + int(0.1 * Fs)
        T_end = min(len(ecg_filtered), R_peak + int(0.3 * Fs))

        if T_start < T_end:
            T_segment = ecg_filtered[T_start:T_end]
            if len(T_segment) > 0:
                # T es un máximo local
                T_idx = np.argmax(T_segment)
                if T_segment[T_idx] > 0:  # Solo si es positivo
                    T_peaks.append(T_start + T_idx)

print(f"Ondas detectadas:")
print(f"  - P: {len(P_peaks)}")
print(f"  - Q: {len(Q_peaks)}")
print(f"  - R: {len(peaks_R)}")
print(f"  - S: {len(S_peaks)}")
print(f"  - T: {len(T_peaks)}")

# 6. CREAR GRÁFICAS HTML

# GRÁFICA 1: Comparación de señal original vs filtrada
print("\nCreando gráfica 1: Comparación de preprocesamiento...")
fig1 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    subplot_titles=['Señal Original', 'Señal Filtrada (5-15 Hz)']
)

# Mostrar solo primeros 10 segundos para mejor visualización
show_samples = int(10 * Fs)
fig1.add_trace(
    go.Scatter(x=vtime[:show_samples], y=ecg_signal[:show_samples],
               mode='lines', name='Original', line=dict(color='blue', width=0.8)),
    row=1, col=1
)

fig1.add_trace(
    go.Scatter(x=vtime[:show_samples], y=ecg_filtered[:show_samples],
               mode='lines', name='Filtrada', line=dict(color='green', width=0.8)),
    row=2, col=1
)

fig1.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
fig1.update_yaxes(title_text="Amplitud", row=1, col=1)
fig1.update_yaxes(title_text="Amplitud", row=2, col=1)

fig1.update_layout(
    title_text='Preprocesamiento de Señal ECG',
    height=800,
    width=1400,
    template='plotly_white'
)

fig1.write_html('taller4_ajustado_1_preprocesamiento.html')
print("✅ Guardado: taller4_ajustado_1_preprocesamiento.html")

# GRÁFICA 2: Detección de picos R
print("Creando gráfica 2: Detección de picos R ajustada...")
fig2 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    subplot_titles=['Señal Integrada Normalizada con Umbral', 'ECG Filtrado con Picos R']
)

# Señal integrada con umbral
fig2.add_trace(
    go.Scatter(x=vtime, y=ecg_integrated_norm,
               mode='lines', name='Integrada', line=dict(color='cyan', width=0.8)),
    row=1, col=1
)

# Línea de umbral
fig2.add_trace(
    go.Scatter(x=[vtime[0], vtime[-1]], y=[threshold, threshold],
               mode='lines', name=f'Umbral ({threshold:.3f})',
               line=dict(color='red', width=2, dash='dash')),
    row=1, col=1
)

# Picos R detectados (usar originales para señal integrada)
if len(peaks_R_original) > 0:
    fig2.add_trace(
        go.Scatter(x=vtime[peaks_R_original], y=ecg_integrated_norm[peaks_R_original],
                   mode='markers', name=f'Picos R ({len(peaks_R_original)})',
                   marker=dict(color='red', size=8)),
        row=1, col=1
    )

# ECG filtrado con picos R
fig2.add_trace(
    go.Scatter(x=vtime, y=ecg_filtered,
               mode='lines', name='ECG Filtrado', line=dict(color='blue', width=0.8)),
    row=2, col=1
)

if len(peaks_R) > 0:
    fig2.add_trace(
        go.Scatter(x=vtime[peaks_R], y=ecg_filtered[peaks_R],
                   mode='markers', name='Picos R',
                   marker=dict(color='red', size=8, symbol='triangle-up')),
        row=2, col=1
    )

fig2.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
fig2.update_yaxes(title_text="Amplitud Normalizada", row=1, col=1)
fig2.update_yaxes(title_text="Amplitud", row=2, col=1)

fig2.update_layout(
    title_text=f'Detección de Picos R Ajustada - {len(peaks_R)} latidos detectados',
    height=800,
    width=1400,
    template='plotly_white'
)

fig2.write_html('taller4_ajustado_2_picos_R.html')
print("✅ Guardado: taller4_ajustado_2_picos_R.html")

# GRÁFICA 3: Ondas PQRST en segmento detallado
print("Creando gráfica 3: Detección PQRST en segmento...")

# Buscar un segmento con buenos complejos QRS (5-15 segundos)
start_time = 5
end_time = 15
start_idx = int(start_time * Fs)
end_idx = int(end_time * Fs)

fig3 = go.Figure()

# ECG filtrado en el segmento
fig3.add_trace(go.Scatter(
    x=vtime[start_idx:end_idx],
    y=ecg_filtered[start_idx:end_idx],
    mode='lines',
    name='ECG Filtrado',
    line=dict(color='blue', width=1.5)
))

# Añadir cada tipo de onda si está en el segmento
def add_peaks_to_plot(peaks, color, symbol, name, size=10):
    peaks_in_segment = [p for p in peaks if start_idx <= p < end_idx]
    if peaks_in_segment:
        fig3.add_trace(go.Scatter(
            x=vtime[peaks_in_segment],
            y=ecg_filtered[peaks_in_segment],
            mode='markers',
            name=f'{name} ({len(peaks_in_segment)})',
            marker=dict(color=color, size=size, symbol=symbol)
        ))
    return len(peaks_in_segment)

p_count = add_peaks_to_plot(P_peaks, 'green', 'circle', 'P', 8)
q_count = add_peaks_to_plot(Q_peaks, 'orange', 'square', 'Q', 8)
r_count = add_peaks_to_plot(peaks_R, 'red', 'triangle-up', 'R', 12)
s_count = add_peaks_to_plot(S_peaks, 'purple', 'triangle-down', 'S', 8)
t_count = add_peaks_to_plot(T_peaks, 'cyan', 'diamond', 'T', 8)

fig3.update_layout(
    title=f'Detección PQRST Ajustada - Segmento {start_time}-{end_time}s (P:{p_count} Q:{q_count} R:{r_count} S:{s_count} T:{t_count})',
    xaxis_title='Tiempo (s)',
    yaxis_title='Amplitud',
    template='plotly_white',
    width=1400,
    height=600,
    hovermode='x unified'
)

fig3.write_html('taller4_ajustado_3_PQRST.html')
print("✅ Guardado: taller4_ajustado_3_PQRST.html")

# GRÁFICA 4: Vista de un solo complejo QRS
if len(peaks_R) > 5:
    print("Creando gráfica 4: Complejo QRS individual...")

    # Seleccionar un pico R en medio de la señal
    selected_R_idx = len(peaks_R) // 2
    selected_R = peaks_R[selected_R_idx]

    # Ventana de 800ms centrada en R
    window_ms = 400
    window_samples = int(window_ms * Fs / 1000)

    qrs_start = max(0, selected_R - window_samples)
    qrs_end = min(len(ecg_filtered), selected_R + window_samples)

    fig4 = go.Figure()

    # ECG en la ventana
    fig4.add_trace(go.Scatter(
        x=vtime[qrs_start:qrs_end],
        y=ecg_filtered[qrs_start:qrs_end],
        mode='lines',
        name='ECG',
        line=dict(color='blue', width=2)
    ))

    # Marcar el pico R principal
    fig4.add_trace(go.Scatter(
        x=[vtime[selected_R]],
        y=[ecg_filtered[selected_R]],
        mode='markers+text',
        name='R',
        marker=dict(color='red', size=15, symbol='triangle-up'),
        text=['R'],
        textposition="top center"
    ))

    # Buscar y marcar ondas asociadas a este R específico
    for peaks, color, symbol, name, search_range in [
        (P_peaks, 'green', 'circle', 'P', (-200, -50)),
        (Q_peaks, 'orange', 'square', 'Q', (-50, 0)),
        (S_peaks, 'purple', 'triangle-down', 'S', (0, 50)),
        (T_peaks, 'cyan', 'diamond', 'T', (100, 300))
    ]:
        # Buscar la onda más cercana en el rango especificado (en ms)
        search_start = selected_R + int(search_range[0] * Fs / 1000)
        search_end = selected_R + int(search_range[1] * Fs / 1000)

        peaks_in_range = [p for p in peaks if search_start <= p <= search_end]
        if peaks_in_range:
            # Tomar el más cercano al R
            closest_peak = min(peaks_in_range, key=lambda x: abs(x - selected_R))

            fig4.add_trace(go.Scatter(
                x=[vtime[closest_peak]],
                y=[ecg_filtered[closest_peak]],
                mode='markers+text',
                name=name,
                marker=dict(color=color, size=12, symbol=symbol),
                text=[name],
                textposition="top center"
            ))

    # Añadir líneas verticales de referencia
    fig4.add_vline(x=vtime[selected_R], line_dash="dot", line_color="red", opacity=0.3)

    # Añadir anotación con intervalos
    if selected_R_idx > 0 and selected_R_idx < len(peaks_R) - 1:
        RR_prev = (peaks_R[selected_R_idx] - peaks_R[selected_R_idx-1]) / Fs * 1000  # ms
        RR_next = (peaks_R[selected_R_idx+1] - peaks_R[selected_R_idx]) / Fs * 1000  # ms
        HR_instant = 60000 / ((RR_prev + RR_next) / 2)  # bpm

        fig4.add_annotation(
            text=f"RR previo: {RR_prev:.0f}ms | RR siguiente: {RR_next:.0f}ms | FC: {HR_instant:.0f}bpm",
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

    fig4.update_layout(
        title='Complejo QRS Individual - Vista Detallada',
        xaxis_title='Tiempo (s)',
        yaxis_title='Amplitud',
        template='plotly_white',
        width=1200,
        height=600,
        showlegend=True
    )

    fig4.write_html('taller4_ajustado_4_complejo_QRS.html')
    print("✅ Guardado: taller4_ajustado_4_complejo_QRS.html")

# RESUMEN
print("\n" + "="*60)
print("ANÁLISIS COMPLETADO - VERSIÓN AJUSTADA")
print("="*60)
print(f"Señal: {duracion_total:.1f} segundos")
print(f"Picos R detectados: {len(peaks_R)}")

if len(peaks_R) > 1:
    latidos_esperados = duracion_total * HR_mean / 60
    print(f"Frecuencia cardíaca: {HR_mean:.1f} bpm")
    print(f"Latidos esperados: ~{latidos_esperados:.0f}")

    # Validación
    if abs(len(peaks_R) - latidos_esperados) / latidos_esperados > 0.2:
        print("⚠️  La cantidad de picos detectados difiere significativamente de lo esperado")
        print("   Considera ajustar los parámetros de detección")

print(f"\nOndas PQST detectadas:")
print(f"  P: {len(P_peaks)} | Q: {len(Q_peaks)} | S: {len(S_peaks)} | T: {len(T_peaks)}")

print("\nGRÁFICAS HTML GENERADAS:")
print("1. taller4_ajustado_1_preprocesamiento.html")
print("2. taller4_ajustado_2_picos_R.html")
print("3. taller4_ajustado_3_PQRST.html")
if len(peaks_R) > 5:
    print("4. taller4_ajustado_4_complejo_QRS.html")

print("\n✅ Abre los archivos HTML en tu navegador para explorar las gráficas")
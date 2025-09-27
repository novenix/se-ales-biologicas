#!/usr/bin/env python3
"""
GENERADOR DE GRÁFICAS HTML PARA TALLER 4
Convierte las gráficas del análisis Pan-Tompkins a versiones HTML interactivas
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Importar función de generarhtmls.py si necesitas alguna
# from generarhtmls import crear_grafica_html

print("GENERANDO GRÁFICAS HTML PARA TALLER 4")
print("=====================================\n")

# 1. CARGAR Y PROCESAR DATOS (igual que taller4_ECG2.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_dir, 'taller4/ECG1.csv')

data = pd.read_csv(csv_file_path, sep=';')
tiempo_str = data['Tiempo'].values
ecg_signal = data['ECG 2'].values  # ECG 2 como en el taller

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

print(f"Datos cargados: {len(ecg_signal)} muestras")
print(f"Procesando algoritmo Pan-Tompkins...")

# 2. ALGORITMO PAN-TOMPKINS (igual que taller4)
# Filtro paso-banda
lowcut = 5.0
highcut = 35.0
nyquist = Fs / 2
low = lowcut / nyquist
high = highcut / nyquist
order = 4

b, a = butter(order, [low, high], btype='band')
ecg_filtered = filtfilt(b, a, ecg_signal)

# Diferenciación
diff_coeffs = np.array([0.2, 0.1, 0, -0.1, -0.2])
ecg_diff = signal.lfilter(diff_coeffs, [1], ecg_filtered)

# Cuadrado
ecg_squared = ecg_diff ** 2

# Integración
window_samples = int(0.08 * Fs)
integration_window = np.ones(window_samples) / window_samples
ecg_integrated = signal.lfilter(integration_window, [1], ecg_squared)

# Detección de picos R
mean_integrated = np.mean(ecg_integrated)
std_integrated = np.std(ecg_integrated)
threshold_factor = 0.15  # Aumentado de 0.05 a 0.15 (3x más alto)
percentile_95 = np.percentile(ecg_integrated, 70)  # Aumentado de 60 a 70
threshold = threshold_factor * percentile_95
threshold_stat = mean_integrated + 0.8 * std_integrated  # Aumentado de 0.5 a 0.8
threshold = min(threshold, threshold_stat)

print(f"Umbral calculado: {threshold:.6f}")
print(f"  - Por percentil: {threshold_factor * percentile_95:.6f}")
print(f"  - Por estadísticas: {threshold_stat:.6f}")

min_distance = int(0.35 * Fs)
peaks_R, properties = find_peaks(ecg_integrated,
                                height=threshold,
                                distance=min_distance,
                                prominence=std_integrated*0.1)

print(f"Picos R detectados: {len(peaks_R)}")

# Calcular intervalos RR
if len(peaks_R) > 1:
    RR_intervals = np.diff(peaks_R) / Fs
    HR_instantaneous = 60 / RR_intervals
    HR_global = (len(peaks_R) / vtime[-1]) * 60
else:
    RR_intervals = np.array([])
    HR_instantaneous = np.array([])
    HR_global = 0

# DETECCIÓN DE ONDAS P, Q, S, T
print("\nDetectando ondas PQRST...")
window_ms = 400
window_samples = int(window_ms * Fs / 1000)

P_peaks = []
Q_peaks = []
S_peaks = []
T_peaks = []

for i, R_peak in enumerate(peaks_R):
    start_idx = max(0, R_peak - window_samples)
    end_idx = min(len(ecg_signal), R_peak + window_samples)

    segment = ecg_signal[start_idx:end_idx]
    segment_indices = np.arange(start_idx, end_idx)

    # Buscar P: antes del R
    pre_R_end = R_peak - start_idx
    if pre_R_end > 10:
        pre_R_segment = segment[:pre_R_end]
        pre_R_indices = segment_indices[:pre_R_end]

        # P es un máximo antes de R
        if len(pre_R_segment) > 0:
            P_idx_local = np.argmax(pre_R_segment[:-10])
            P_idx_global = pre_R_indices[P_idx_local]
            P_peaks.append(P_idx_global)

        # Q es un mínimo justo antes de R
        Q_search_start = max(0, pre_R_end - 20)
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
        S_search_end = min(len(post_R_segment), 20)
        S_segment = post_R_segment[1:S_search_end]
        if len(S_segment) > 0:
            S_idx_local = np.argmin(S_segment)
            S_idx_global = post_R_indices[1 + S_idx_local]
            S_peaks.append(S_idx_global)

        # T es un máximo después de S
        if len(post_R_segment) > 20:
            T_segment = post_R_segment[20:]
            T_indices = post_R_indices[20:]
            if len(T_segment) > 0:
                T_idx_local = np.argmax(T_segment)
                T_idx_global = T_indices[T_idx_local]
                T_peaks.append(T_idx_global)

print(f"Ondas detectadas: P={len(P_peaks)}, Q={len(Q_peaks)}, R={len(peaks_R)}, S={len(S_peaks)}, T={len(T_peaks)}")

# 3. CREAR GRÁFICAS HTML INTERACTIVAS

# GRÁFICA 1: Señal ECG Original
print("\nCreando gráfica 1: Señal ECG Original...")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=vtime,
    y=ecg_signal,
    mode='lines',
    name='ECG Original',
    line=dict(color='blue', width=1)
))

fig1.update_layout(
    title='Señal ECG Original - ECG2',
    xaxis_title='Tiempo (s)',
    yaxis_title='Amplitud (μV)',
    template='plotly_white',
    width=1400,
    height=500,
    hovermode='x unified'
)

fig1.write_html('taller4_1_ecg_original.html')
print("✅ Guardado: taller4_1_ecg_original.html")

# GRÁFICA 2: Todos los pasos del algoritmo Pan-Tompkins
print("Creando gráfica 2: Algoritmo Pan-Tompkins completo...")
fig2 = make_subplots(
    rows=5, cols=1,
    shared_xaxes=True,
    subplot_titles=['Señal Original',
                    'Filtro Paso-Banda (5-35Hz)',
                    'Diferenciación',
                    'Cuadrado',
                    'Integración (ventana 80ms)'],
    vertical_spacing=0.05
)

# Añadir cada paso
fig2.add_trace(
    go.Scatter(x=vtime, y=ecg_signal, mode='lines', name='Original',
               line=dict(color='blue', width=0.8)),
    row=1, col=1
)

fig2.add_trace(
    go.Scatter(x=vtime, y=ecg_filtered, mode='lines', name='Filtrado',
               line=dict(color='green', width=0.8)),
    row=2, col=1
)

fig2.add_trace(
    go.Scatter(x=vtime, y=ecg_diff, mode='lines', name='Diferenciado',
               line=dict(color='red', width=0.8)),
    row=3, col=1
)

fig2.add_trace(
    go.Scatter(x=vtime, y=ecg_squared, mode='lines', name='Cuadrado',
               line=dict(color='magenta', width=0.8)),
    row=4, col=1
)

fig2.add_trace(
    go.Scatter(x=vtime, y=ecg_integrated, mode='lines', name='Integrado',
               line=dict(color='cyan', width=0.8)),
    row=5, col=1
)

# Actualizar diseño
fig2.update_xaxes(title_text="Tiempo (s)", row=5, col=1)
fig2.update_yaxes(title_text="Amplitud", row=1, col=1)
fig2.update_yaxes(title_text="Amplitud", row=2, col=1)
fig2.update_yaxes(title_text="Amplitud", row=3, col=1)
fig2.update_yaxes(title_text="Amplitud²", row=4, col=1)
fig2.update_yaxes(title_text="Integrada", row=5, col=1)

fig2.update_layout(
    title_text='Algoritmo Pan-Tompkins - Todos los Pasos',
    height=1200,
    width=1400,
    template='plotly_white',
    showlegend=False
)

fig2.write_html('taller4_2_pan_tompkins_pasos.html')
print("✅ Guardado: taller4_2_pan_tompkins_pasos.html")

# GRÁFICA 3: Detección de Picos R
print("Creando gráfica 3: Detección de Picos R...")
fig3 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    subplot_titles=['Señal Integrada con Picos R', 'ECG Original con Picos R'],
    vertical_spacing=0.1
)

# Señal integrada
fig3.add_trace(
    go.Scatter(x=vtime, y=ecg_integrated, mode='lines',
               name='Señal Integrada', line=dict(color='cyan', width=1)),
    row=1, col=1
)

# Línea de umbral
fig3.add_trace(
    go.Scatter(x=[vtime[0], vtime[-1]], y=[threshold, threshold],
               mode='lines', name=f'Umbral ({threshold:.3f})',
               line=dict(color='red', width=2, dash='dash')),
    row=1, col=1
)

# Picos R en señal integrada
if len(peaks_R) > 0:
    fig3.add_trace(
        go.Scatter(x=vtime[peaks_R], y=ecg_integrated[peaks_R],
                   mode='markers', name=f'Picos R ({len(peaks_R)})',
                   marker=dict(color='red', size=8)),
        row=1, col=1
    )

# ECG original
fig3.add_trace(
    go.Scatter(x=vtime, y=ecg_signal, mode='lines',
               name='ECG Original', line=dict(color='blue', width=0.8)),
    row=2, col=1
)

# Picos R en ECG original
if len(peaks_R) > 0:
    fig3.add_trace(
        go.Scatter(x=vtime[peaks_R], y=ecg_signal[peaks_R],
                   mode='markers', name='Picos R',
                   marker=dict(color='red', size=6)),
        row=2, col=1
    )

fig3.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
fig3.update_yaxes(title_text="Amplitud Integrada", row=1, col=1)
fig3.update_yaxes(title_text="Amplitud (μV)", row=2, col=1)

fig3.update_layout(
    title_text='Detección de Picos R',
    height=800,
    width=1400,
    template='plotly_white'
)

fig3.write_html('taller4_3_deteccion_picos_R.html')
print("✅ Guardado: taller4_3_deteccion_picos_R.html")

# GRÁFICA 4: Análisis HRV (si hay suficientes picos)
if len(RR_intervals) > 0:
    print("Creando gráfica 4: Análisis de Variabilidad Cardíaca...")

    fig4 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=['Tacograma - Intervalos RR', 'Frecuencia Cardíaca Instantánea'],
        vertical_spacing=0.1
    )

    # Intervalos RR
    RR_times = vtime[peaks_R[1:]]
    fig4.add_trace(
        go.Scatter(x=RR_times, y=RR_intervals*1000,
                   mode='lines+markers', name='Intervalos RR',
                   line=dict(color='blue', width=1),
                   marker=dict(size=4)),
        row=1, col=1
    )

    # Frecuencia cardíaca instantánea
    fig4.add_trace(
        go.Scatter(x=RR_times, y=HR_instantaneous,
                   mode='lines+markers', name='FC Instantánea',
                   line=dict(color='red', width=1),
                   marker=dict(size=4)),
        row=2, col=1
    )

    # Línea de FC global
    fig4.add_trace(
        go.Scatter(x=[RR_times[0], RR_times[-1]], y=[HR_global, HR_global],
                   mode='lines', name=f'FC Global: {HR_global:.1f} bpm',
                   line=dict(color='green', width=2, dash='dash')),
        row=2, col=1
    )

    fig4.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
    fig4.update_yaxes(title_text="Intervalo RR (ms)", row=1, col=1)
    fig4.update_yaxes(title_text="FC (bpm)", row=2, col=1)

    fig4.update_layout(
        title_text='Análisis de Variabilidad Cardíaca (HRV)',
        height=800,
        width=1400,
        template='plotly_white'
    )

    fig4.write_html('taller4_4_analisis_HRV.html')
    print("✅ Guardado: taller4_4_analisis_HRV.html")

# GRÁFICA 5: Detección de ondas PQRST
print("Creando gráfica 5: Detección completa de ondas PQRST...")

# Crear dos subgráficas: completa y zoom
fig5 = make_subplots(
    rows=2, cols=1,
    subplot_titles=['Vista Completa con PQRST', 'Segmento Detallado (10-20s)'],
    vertical_spacing=0.15,
    row_heights=[0.5, 0.5]
)

# Vista completa
fig5.add_trace(
    go.Scatter(x=vtime, y=ecg_signal, mode='lines',
               name='ECG', line=dict(color='blue', width=0.8)),
    row=1, col=1
)

# Añadir todas las ondas detectadas
if P_peaks:
    fig5.add_trace(
        go.Scatter(x=vtime[P_peaks], y=ecg_signal[P_peaks],
                   mode='markers', name=f'P ({len(P_peaks)})',
                   marker=dict(color='green', size=6, symbol='circle')),
        row=1, col=1
    )

if Q_peaks:
    fig5.add_trace(
        go.Scatter(x=vtime[Q_peaks], y=ecg_signal[Q_peaks],
                   mode='markers', name=f'Q ({len(Q_peaks)})',
                   marker=dict(color='orange', size=6, symbol='square')),
        row=1, col=1
    )

if len(peaks_R) > 0:
    fig5.add_trace(
        go.Scatter(x=vtime[peaks_R], y=ecg_signal[peaks_R],
                   mode='markers', name=f'R ({len(peaks_R)})',
                   marker=dict(color='red', size=8, symbol='triangle-up')),
        row=1, col=1
    )

if S_peaks:
    fig5.add_trace(
        go.Scatter(x=vtime[S_peaks], y=ecg_signal[S_peaks],
                   mode='markers', name=f'S ({len(S_peaks)})',
                   marker=dict(color='purple', size=6, symbol='triangle-down')),
        row=1, col=1
    )

if T_peaks:
    fig5.add_trace(
        go.Scatter(x=vtime[T_peaks], y=ecg_signal[T_peaks],
                   mode='markers', name=f'T ({len(T_peaks)})',
                   marker=dict(color='cyan', size=6, symbol='diamond')),
        row=1, col=1
    )

# Segmento detallado 10-20s
start_time = 10
end_time = 20
start_idx = int(start_time * Fs)
end_idx = int(end_time * Fs)

fig5.add_trace(
    go.Scatter(x=vtime[start_idx:end_idx], y=ecg_signal[start_idx:end_idx],
               mode='lines', name='ECG zoom', line=dict(color='blue', width=1.5),
               showlegend=False),
    row=2, col=1
)

# Añadir ondas PQRST en el segmento
def add_peaks_in_segment(peaks, color, symbol, name):
    segment_peaks = [p for p in peaks if start_idx <= p < end_idx]
    if segment_peaks:
        fig5.add_trace(
            go.Scatter(x=vtime[segment_peaks], y=ecg_signal[segment_peaks],
                       mode='markers', name=name,
                       marker=dict(color=color, size=10, symbol=symbol),
                       showlegend=False),
            row=2, col=1
        )
    return len(segment_peaks)

p_count = add_peaks_in_segment(P_peaks, 'green', 'circle', 'P')
q_count = add_peaks_in_segment(Q_peaks, 'orange', 'square', 'Q')
r_count = add_peaks_in_segment(peaks_R, 'red', 'triangle-up', 'R')
s_count = add_peaks_in_segment(S_peaks, 'purple', 'triangle-down', 'S')
t_count = add_peaks_in_segment(T_peaks, 'cyan', 'diamond', 'T')

# Añadir anotaciones con conteo en el segmento
fig5.add_annotation(
    text=f"Ondas en segmento: P={p_count}, Q={q_count}, R={r_count}, S={s_count}, T={t_count}",
    xref="x2", yref="y2",
    x=start_time + (end_time-start_time)/2,
    y=max(ecg_signal[start_idx:end_idx]) * 1.1,
    showarrow=False,
    bgcolor="white",
    bordercolor="black",
    borderwidth=1
)

fig5.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
fig5.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
fig5.update_yaxes(title_text="Amplitud (μV)", row=1, col=1)
fig5.update_yaxes(title_text="Amplitud (μV)", row=2, col=1)

fig5.update_layout(
    title_text='Detección de Ondas PQRST - Algoritmo Pan-Tompkins',
    height=1000,
    width=1400,
    template='plotly_white',
    hovermode='x unified'
)

fig5.write_html('taller4_5_ondas_PQRST.html')
print("✅ Guardado: taller4_5_ondas_PQRST.html")

# GRÁFICA 6: Segmento ultra-detallado con un solo complejo QRS
print("Creando gráfica 6: Complejo QRS individual...")

# Buscar un buen complejo QRS para mostrar (el 5to R si existe)
if len(peaks_R) > 5:
    selected_R = peaks_R[5]

    # Ventana de 600ms centrada en R
    window_ms = 300  # ±300ms
    window_samples = int(window_ms * Fs / 1000)

    qrs_start = max(0, selected_R - window_samples)
    qrs_end = min(len(ecg_signal), selected_R + window_samples)

    fig6 = go.Figure()

    # ECG en la ventana
    fig6.add_trace(go.Scatter(
        x=vtime[qrs_start:qrs_end],
        y=ecg_signal[qrs_start:qrs_end],
        mode='lines',
        name='ECG',
        line=dict(color='blue', width=2)
    ))

    # Marcar cada onda si está en el rango
    for peaks, color, symbol, name in [
        (P_peaks, 'green', 'circle', 'P'),
        (Q_peaks, 'orange', 'square', 'Q'),
        ([selected_R], 'red', 'triangle-up', 'R'),
        (S_peaks, 'purple', 'triangle-down', 'S'),
        (T_peaks, 'cyan', 'diamond', 'T')
    ]:
        peaks_in_window = [p for p in peaks if qrs_start <= p < qrs_end]
        if peaks_in_window:
            fig6.add_trace(go.Scatter(
                x=vtime[peaks_in_window],
                y=ecg_signal[peaks_in_window],
                mode='markers+text',
                name=name,
                marker=dict(color=color, size=12, symbol=symbol),
                text=[name] * len(peaks_in_window),
                textposition="top center",
                textfont=dict(size=14, color=color)
            ))

    fig6.update_layout(
        title='Complejo QRS Individual - Vista Detallada',
        xaxis_title='Tiempo (s)',
        yaxis_title='Amplitud (μV)',
        template='plotly_white',
        width=1000,
        height=600,
        hovermode='x unified',
        showlegend=True
    )

    # Añadir líneas verticales punteadas para cada onda
    for peaks, color, name in [
        (P_peaks, 'green', 'P'),
        (Q_peaks, 'orange', 'Q'),
        ([selected_R], 'red', 'R'),
        (S_peaks, 'purple', 'S'),
        (T_peaks, 'cyan', 'T')
    ]:
        peaks_in_window = [p for p in peaks if qrs_start <= p < qrs_end]
        for peak in peaks_in_window:
            fig6.add_vline(x=vtime[peak], line_dash="dot",
                          line_color=color, opacity=0.3)

    fig6.write_html('taller4_6_complejo_QRS.html')
    print("✅ Guardado: taller4_6_complejo_QRS.html")

# RESUMEN
print("\n" + "="*50)
print("GRÁFICAS HTML GENERADAS EXITOSAMENTE:")
print("="*50)
print("1. taller4_1_ecg_original.html - Señal ECG completa")
print("2. taller4_2_pan_tompkins_pasos.html - Todos los pasos del algoritmo")
print("3. taller4_3_deteccion_picos_R.html - Detección de picos R")
if len(RR_intervals) > 0:
    print("4. taller4_4_analisis_HRV.html - Análisis de variabilidad cardíaca")
print("5. taller4_5_ondas_PQRST.html - Detección completa de ondas PQRST")
if len(peaks_R) > 5:
    print("6. taller4_6_complejo_QRS.html - Complejo QRS individual detallado")

print("\n💡 CARACTERÍSTICAS DE LAS GRÁFICAS HTML:")
print("  ✓ Interactivas: zoom, pan, hover para ver valores")
print("  ✓ Exportables: botón de cámara para guardar como imagen")
print("  ✓ Responsivas: se adaptan al tamaño de la ventana")
print("  ✓ Leyendas clickeables: ocultar/mostrar series")

print("\n📊 ESTADÍSTICAS DEL ANÁLISIS:")
print(f"  - Picos R detectados: {len(peaks_R)}")
if len(RR_intervals) > 0:
    print(f"  - Frecuencia cardíaca global: {HR_global:.1f} bpm")
    print(f"  - RR promedio: {np.mean(RR_intervals)*1000:.1f} ms")
    print(f"  - RMSSD: {np.sqrt(np.mean(np.diff(RR_intervals)**2))*1000:.1f} ms")

print("\n✅ Abre los archivos HTML en tu navegador para explorar las gráficas")
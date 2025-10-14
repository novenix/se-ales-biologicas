import scipy.io as sio
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import FastICA
import os

# Cargar datos
ruta = os.path.join(os.path.dirname(__file__), 'Extra1.mat')
data = sio.loadmat(ruta)
senales = data['Extracell']
header = data['Header'][0,0]
fs = header['SampleRate'][0,0]

# Info basica
n_canales = senales.shape[0]
n_muestras = senales.shape[1]
duracion = n_muestras / fs
tiempo = np.linspace(0, duracion, n_muestras)

print("=== Datos del archivo .mat ===")
print(f"Frecuencia de muestreo: {fs/1000:.2f} kHz")
print(f"Duracion: {duracion:.2f} s")
print(f"Canales: {n_canales}")
print(f"Muestras por canal: {n_muestras:,}")
print(f"Unidades: {header['Units'][0]}\n")

print("=== Estadisticas por electrodo ===")
for i in range(n_canales):
    canal = senales[i,:]
    print(f"Electrodo {i+1}:")
    print(f"  Rango: [{canal.min():.2f}, {canal.max():.2f}] uV")
    print(f"  Media: {canal.mean():.2f} uV")
    print(f"  Std: {canal.std():.2f} uV")
print()

# Punto 1: Graficar las 3 senales 

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=['Electrodo 1', 'Electrodo 2', 'Electrodo 3'],
    vertical_spacing=0.05
)

# bajar de 4 millones de puntos a40 k puntos para que no pese tanto
factor_diezmado = 100
tiempo_plot = tiempo[::factor_diezmado]

for i in range(n_canales):
    senales_plot = senales[i, ::factor_diezmado]
    fig.add_trace(
        go.Scatter(x=tiempo_plot, y=senales_plot,
                   mode='lines', name=f'Electrodo {i+1}',
                   line=dict(width=0.8, color=['blue', 'green', 'red'][i])),
        row=i+1, col=1
    )

fig.update_xaxes(title_text="Tiempo (s)", row=3, col=1)
fig.update_yaxes(title_text="Amplitud (uV)", row=1, col=1)
fig.update_yaxes(title_text="Amplitud (uV)", row=2, col=1)
fig.update_yaxes(title_text="Amplitud (uV)", row=3, col=1)

fig.update_layout(
    title_text='Registros Extracelulares - 3 Electrodos',
    height=1000,
    width=1400,
    template='plotly_white',
    showlegend=False
)

fig.write_html('punto1_registros_extracelulares.html')
print("Guardado: punto1_registros_extracelulares.html\n")


# Punto 2


# traspuesta
senales_T = senales.T


ica = FastICA(n_components=3, random_state=42, max_iter=500)
fuentes = ica.fit_transform(senales_T)  # (n_muestras, 3)
mixing_matrix = ica.mixing_  # matriz de mezcla

# Volver a formato (canales, muestras)
fuentes = fuentes.T

print(f"punto 2 fuentes independientes encontradas: {fuentes.shape}")
print(f"Matriz de mezcla: {mixing_matrix.shape}\n")

# Graficar fuentes ICA
fig_ica = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=['Fuente ICA 1', 'Fuente ICA 2', 'Fuente ICA 3'],
    vertical_spacing=0.05
)

for i in range(3):
    fuentes_plot = fuentes[i, ::factor_diezmado]
    fig_ica.add_trace(
        go.Scatter(x=tiempo_plot, y=fuentes_plot,
                   mode='lines', name=f'Fuente {i+1}',
                   line=dict(width=0.8, color=['purple', 'orange', 'cyan'][i])),
        row=i+1, col=1
    )

fig_ica.update_xaxes(title_text="Tiempo (s)", row=3, col=1)
fig_ica.update_yaxes(title_text="Amplitud (u.a.)", row=1, col=1)
fig_ica.update_yaxes(title_text="Amplitud (u.a.)", row=2, col=1)
fig_ica.update_yaxes(title_text="Amplitud (u.a.)", row=3, col=1)

fig_ica.update_layout(
    title_text='Fuentes Independientes (ICA)',
    height=1000,
    width=1400,
    template='plotly_white',
    showlegend=False
)

fig_ica.write_html('punto2_fuentes_ICA.html')

# varianza de cada fuente
print("\n=== Analisis de fuentes ===")
for i in range(3):
    std = np.std(fuentes[i,:])
    rango = np.max(fuentes[i,:]) - np.min(fuentes[i,:])
    print(f"Fuente {i+1}: Std={std:.2f}, Rango={rango:.2f}")
###  Identifique en qué fuente hay potenciales
# de acción y en cuál no.

#se ven espaiks nitidos en los rangos del 0 al 20 de la fuente 1, luego
# en el spike 26, luego en el spike despues del 40 tambien se ven nitidos

#en la fuente ica 2 e ica 3 se ve mucho ruido y no se pueden ver spikes nitidos

# ¿A qué corresponde la señal que no tiene AP?

#para mi la fuente 3 es la que menos tiene un ap porque no se ven spikes nitidos
#tambien la fuente 2 no tiene spiles nitidos entonces no le veo potenciales de accion





# Punto 3: Detectar spikes y graficar spike train
print("\n=== Punto 3: Deteccion de spikes ===")

# Usar fuente 1 que tiene los AP
senal_spikes = fuentes[0,:]

# Detectar picos con umbral simple
umbral = 3 * np.std(senal_spikes)
from scipy.signal import find_peaks

# Encontrar picos que superen el umbral
picos, propiedades = find_peaks(senal_spikes, height=umbral, distance=int(0.002*fs))

print(f"Spikes detectados: {len(picos)}")
print(f"Umbral usado: {umbral:.2f}")

# Calcular tiempos de los spikes
tiempos_spikes = picos / fs

# Graficar spike train (raster plot)
fig_spike = go.Figure()

# Graficar como puntos verticales
fig_spike.add_trace(go.Scatter(
    x=tiempos_spikes,
    y=np.ones(len(tiempos_spikes)),
    mode='markers',
    marker=dict(symbol='line-ns-open', size=12, line=dict(width=2, color='black')),
    name='Spikes'
))

fig_spike.update_layout(
    title='Spike Train - Fuente ICA 1',
    xaxis_title='Tiempo (s)',
    yaxis_title='Neurona',
    height=300,
    width=1400,
    template='plotly_white',
    yaxis=dict(showticklabels=False, range=[0.5, 1.5])
)

fig_spike.write_html('punto3_spike_train.html')
print("Guardado: punto3_spike_train.html")

# Graficar senal con spikes marcados
fig_senal_spikes = go.Figure()

# Diezmar senal para graficar
senal_plot = senal_spikes[::factor_diezmado]
tiempo_senal = tiempo[::factor_diezmado]

fig_senal_spikes.add_trace(go.Scatter(
    x=tiempo_senal, y=senal_plot,
    mode='lines', name='Señal',
    line=dict(width=0.8, color='blue')
))

# Marcar spikes detectados
fig_senal_spikes.add_trace(go.Scatter(
    x=tiempos_spikes,
    y=senal_spikes[picos],
    mode='markers',
    name='Spikes',
    marker=dict(size=8, color='red', symbol='x')
))

fig_senal_spikes.update_layout(
    title='Deteccion de Spikes en Fuente ICA 1',
    xaxis_title='Tiempo (s)',
    yaxis_title='Amplitud (u.a.)',
    height=500,
    width=1400,
    template='plotly_white'
)

fig_senal_spikes.write_html('punto3_deteccion_spikes.html')
print("Guardado: punto3_deteccion_spikes.html")



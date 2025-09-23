import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import os

# Configurar matplotlib para gráficas interactivas
plt.ion()

print("=== TALLER 3: SPIKE SORTING Y CLUSTERING ===")
print("Basado en el procesamiento de señales extracelulares del Taller 2\n")

# 1. Cargar y procesar datos (reutilizando código del Taller 2)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Usar los datos del taller2
mat_file_path = os.path.join(os.path.dirname(current_dir), 'taller2', 'IntraExtracelular_py.mat')
data = scipy.io.loadmat(mat_file_path)

print("✓ Cargando datos del Taller 2...")

# Extraer variables
variable1 = data['variable1']
extracelular = variable1[1, :]  # Segundo canal (extracelular)
Fs = float(data['SamplingRateHz'][0][0])

print(f"✓ Frecuencia de muestreo: {Fs} Hz")
print(f"✓ Señal extracelular: {len(extracelular)} muestras")

# Aplicar el mismo filtrado del Taller 2
lowcut = 300.0
highcut = 3000.0
nyquist = Fs / 2
low = lowcut / nyquist
high = highcut / nyquist
order = 4
b, a = butter(order, [low, high], btype='band')
extracelular_filtrado = filtfilt(b, a, extracelular)

print(f"✓ Filtro aplicado: {lowcut}-{highcut} Hz")

# Detectar spikes (misma lógica del Taller 2)
extracelular_invertido = -extracelular_filtrado
mean_extra = np.mean(extracelular_invertido)
std_extra = np.std(extracelular_invertido)
threshold_extra = mean_extra + 2.5 * std_extra

peaks_extra, properties = find_peaks(extracelular_invertido,
                                   height=threshold_extra,
                                   distance=int(0.001*Fs),  # 1ms mínimo
                                   prominence=std_extra)

print(f"✓ Picos detectados: {len(peaks_extra)}")
print(f"✓ Umbral usado: {threshold_extra:.4f} μV")

# Crear vector de tiempo
N = len(extracelular)
Ts = 1/Fs
vtime = np.arange(0, N * Ts, Ts)
if len(vtime) > N:
    vtime = vtime[:N]

# Graficar señal con spikes detectados
fig1, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.plot(vtime, extracelular_filtrado, 'b-', linewidth=0.8, label='Señal Extracelular Filtrada')
ax.plot(vtime[peaks_extra], extracelular_filtrado[peaks_extra], 'ro', markersize=4,
        label=f'Spikes detectados ({len(peaks_extra)})')
ax.set_title('Detección de Spikes - Señal Extracelular')
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Amplitud (μV)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show(block=True)

# 2. Construir matriz de eventos (1ms antes y después del pico)
print("\n=== CONSTRUCCIÓN DE MATRIZ DE EVENTOS ===")

# Calcular ventana de 1ms en muestras
window_ms = 0.001  # 1ms
window_samples = int(window_ms * Fs)
print(f"✓ Ventana: ±{window_ms*1000} ms = ±{window_samples} muestras")

# Filtrar picos que están muy cerca de los bordes
valid_peaks = []
for peak in peaks_extra:
    if peak >= window_samples and peak < len(extracelular_filtrado) - window_samples:
        valid_peaks.append(peak)

valid_peaks = np.array(valid_peaks)
print(f"✓ Picos válidos (lejos de bordes): {len(valid_peaks)}")

# Crear matriz de eventos
# Filas = spikes, Columnas = tiempo (2ms total: 1ms antes + 1ms después)
eventos = np.zeros((len(valid_peaks), 2 * window_samples + 1))

for i, peak in enumerate(valid_peaks):
    start_idx = peak - window_samples
    end_idx = peak + window_samples + 1
    eventos[i, :] = extracelular_filtrado[start_idx:end_idx]

print(f"✓ Matriz de eventos creada: {eventos.shape[0]} spikes × {eventos.shape[1]} muestras")

# Crear vector de tiempo para la ventana del spike (centrado en 0)
spike_time = np.linspace(-window_ms, window_ms, eventos.shape[1])

# Graficar superposición de todos los spikes
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

# Graficar todos los eventos (equivalente a plot(eventos') en MATLAB)
for i in range(eventos.shape[0]):
    ax.plot(spike_time * 1000, eventos[i, :], 'b-', alpha=0.3, linewidth=0.5)

# Calcular y graficar promedio
spike_promedio = np.mean(eventos, axis=0)
ax.plot(spike_time * 1000, spike_promedio, 'r-', linewidth=3, label=f'Promedio ({len(valid_peaks)} spikes)')

ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Momento del pico')
ax.set_title('Superposición de Todos los Spikes Extracelulares')
ax.set_xlabel('Tiempo (ms)')
ax.set_ylabel('Amplitud (μV)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show(block=True)

# 3. Construir matriz de características
print("\n=== EXTRACCIÓN DE CARACTERÍSTICAS ===")

# Matriz de características: [máximo, mínimo, pendiente]
# Filas = spikes, Columnas = características
caracteristicas = np.zeros((len(valid_peaks), 3))

for i in range(len(valid_peaks)):
    spike = eventos[i, :]

    # Máximo valor del spike
    maximo = np.max(spike)

    # Mínimo valor del spike (recordar que los spikes extracelulares son negativos)
    minimo = np.min(spike)

    # Pendiente entre máximo y mínimo
    idx_max = np.argmax(spike)
    idx_min = np.argmin(spike)

    # Calcular pendiente como diferencia de amplitud / diferencia de tiempo
    delta_amplitud = maximo - minimo
    delta_tiempo = abs(idx_max - idx_min) * Ts  # Convertir índices a tiempo

    if delta_tiempo > 0:
        pendiente = delta_amplitud / delta_tiempo
    else:
        pendiente = 0

    caracteristicas[i, :] = [maximo, minimo, pendiente]

print(f"✓ Matriz de características creada: {caracteristicas.shape[0]} spikes × 3 características")
print(f"  - Máximo: rango [{np.min(caracteristicas[:,0]):.2f}, {np.max(caracteristicas[:,0]):.2f}] μV")
print(f"  - Mínimo: rango [{np.min(caracteristicas[:,1]):.2f}, {np.max(caracteristicas[:,1]):.2f}] μV")
print(f"  - Pendiente: rango [{np.min(caracteristicas[:,2]):.2f}, {np.max(caracteristicas[:,2]):.2f}] μV/s")

# Graficar el espacio de características
fig3 = plt.figure(figsize=(15, 5))

# Subplot 1: Máximo vs Mínimo
ax1 = fig3.add_subplot(131)
ax1.scatter(caracteristicas[:, 0], caracteristicas[:, 1], alpha=0.7, s=20)
ax1.set_xlabel('Máximo (μV)')
ax1.set_ylabel('Mínimo (μV)')
ax1.set_title('Máximo vs Mínimo')
ax1.grid(True, alpha=0.3)

# Subplot 2: Máximo vs Pendiente
ax2 = fig3.add_subplot(132)
ax2.scatter(caracteristicas[:, 0], caracteristicas[:, 2], alpha=0.7, s=20)
ax2.set_xlabel('Máximo (μV)')
ax2.set_ylabel('Pendiente (μV/s)')
ax2.set_title('Máximo vs Pendiente')
ax2.grid(True, alpha=0.3)

# Subplot 3: Mínimo vs Pendiente
ax3 = fig3.add_subplot(133)
ax3.scatter(caracteristicas[:, 1], caracteristicas[:, 2], alpha=0.7, s=20)
ax3.set_xlabel('Mínimo (μV)')
ax3.set_ylabel('Pendiente (μV/s)')
ax3.set_title('Mínimo vs Pendiente')
ax3.grid(True, alpha=0.3)

plt.suptitle('Espacio de Características de los Spikes')
plt.tight_layout()
plt.show(block=True)

# Gráfico 3D del espacio de características completo
fig4 = plt.figure(figsize=(10, 8))
ax = fig4.add_subplot(111, projection='3d')

scatter = ax.scatter(caracteristicas[:, 0], caracteristicas[:, 1], caracteristicas[:, 2],
                    alpha=0.7, s=30, c=range(len(caracteristicas)), cmap='viridis')

ax.set_xlabel('Máximo (μV)')
ax.set_ylabel('Mínimo (μV)')
ax.set_zlabel('Pendiente (μV/s)')
ax.set_title('Espacio de Características 3D')

plt.colorbar(scatter, ax=ax, label='Índice del Spike')
plt.tight_layout()
plt.show(block=True)

# 4. Clustering con K-means
print("\n=== CLUSTERING CON K-MEANS ===")

# Normalizar las características para mejor clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
caracteristicas_norm = scaler.fit_transform(caracteristicas)

# Determinar número óptimo de clusters usando método del codo y silhouette score
from sklearn.metrics import silhouette_score

# Probar diferentes números de clusters
n_clusters_range = range(2, 8)
inertias = []
silhouette_scores = []

print("✓ Evaluando diferentes números de clusters...")
for n in n_clusters_range:
    kmeans_temp = KMeans(n_clusters=n, random_state=42, n_init=10)
    cluster_labels_temp = kmeans_temp.fit_predict(caracteristicas_norm)

    inertias.append(kmeans_temp.inertia_)
    silhouette_avg = silhouette_score(caracteristicas_norm, cluster_labels_temp)
    silhouette_scores.append(silhouette_avg)

    print(f"  - {n} clusters: inercia = {kmeans_temp.inertia_:.2f}, silhouette = {silhouette_avg:.3f}")

# Método del codo: buscar el punto donde la reducción de inercia se estabiliza
# Calcular diferencias de segunda derivada para encontrar el "codo"
if len(inertias) >= 3:
    # Calcular diferencias de segunda derivada
    diff1 = np.diff(inertias)
    diff2 = np.diff(diff1)
    # El codo está donde la segunda derivada es más positiva (menor aceleración)
    codo_idx = np.argmax(diff2) + 2  # +2 porque diff2 tiene 2 elementos menos
    n_clusters_codo = list(n_clusters_range)[codo_idx]
else:
    n_clusters_codo = 3

# Silhouette score: elegir el número con mayor silhouette score
n_clusters_silhouette = list(n_clusters_range)[np.argmax(silhouette_scores)]

# Graficar análisis de clusters
fig5, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Método del codo
ax1.plot(n_clusters_range, inertias, 'bo-')
ax1.axvline(x=n_clusters_codo, color='red', linestyle='--',
           label=f'Codo óptimo: {n_clusters_codo}')
ax1.set_xlabel('Número de Clusters')
ax1.set_ylabel('Inercia')
ax1.set_title('Método del Codo para K-means')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Silhouette score
ax2.plot(n_clusters_range, silhouette_scores, 'go-')
ax2.axvline(x=n_clusters_silhouette, color='red', linestyle='--',
           label=f'Máximo silhouette: {n_clusters_silhouette}')
ax2.set_xlabel('Número de Clusters')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Análisis Silhouette Score')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Decidir número final de clusters
# Priorizar silhouette score, pero considerar el codo si es razonable
if abs(n_clusters_silhouette - n_clusters_codo) <= 1:
    # Si están cerca, usar el del silhouette
    n_clusters = n_clusters_silhouette
    criterio = "silhouette score"
else:
    # Si están lejos, evaluar cuál tiene mejor balance
    # Usar silhouette score como criterio principal
    n_clusters = n_clusters_silhouette
    criterio = "silhouette score (codo difiere)"

print(f"\n✓ DECISIÓN AUTOMÁTICA:")
print(f"  - Método del codo sugiere: {n_clusters_codo} clusters")
print(f"  - Silhouette score sugiere: {n_clusters_silhouette} clusters")
print(f"  - Seleccionado: {n_clusters} clusters (basado en {criterio})")
print(f"  - Silhouette score final: {silhouette_scores[n_clusters-2]:.3f}")

# Aplicar K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(caracteristicas_norm)

print(f"✓ Clustering completado")
for i in range(n_clusters):
    count = np.sum(cluster_labels == i)
    print(f"  - Cluster {i+1}: {count} spikes ({count/len(cluster_labels)*100:.1f}%)")

# Visualizar clusters en el espacio de características
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(n_clusters):
    mask = cluster_labels == i
    ax3.scatter(caracteristicas[mask, 0], caracteristicas[mask, 1],
               c=colors[i], alpha=0.7, s=30, label=f'Cluster {i+1}')

ax3.set_xlabel('Máximo (μV)')
ax3.set_ylabel('Mínimo (μV)')
ax3.set_title('Clusters en Espacio de Características')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=True)

# Visualizar clusters en 3D
fig6 = plt.figure(figsize=(12, 5))

# Subplot 1: Clusters en 3D
ax1 = fig6.add_subplot(121, projection='3d')
for i in range(n_clusters):
    mask = cluster_labels == i
    ax1.scatter(caracteristicas[mask, 0], caracteristicas[mask, 1], caracteristicas[mask, 2],
               c=colors[i], alpha=0.7, s=30, label=f'Cluster {i+1}')

ax1.set_xlabel('Máximo (μV)')
ax1.set_ylabel('Mínimo (μV)')
ax1.set_zlabel('Pendiente (μV/s)')
ax1.set_title('Clusters en 3D')
ax1.legend()

# Subplot 2: Spikes promedio por cluster
ax2 = fig6.add_subplot(122)
for i in range(n_clusters):
    mask = cluster_labels == i
    if np.sum(mask) > 0:
        spike_promedio_cluster = np.mean(eventos[mask, :], axis=0)
        ax2.plot(spike_time * 1000, spike_promedio_cluster,
                color=colors[i], linewidth=2, label=f'Cluster {i+1} (n={np.sum(mask)})')

ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel('Tiempo (ms)')
ax2.set_ylabel('Amplitud (μV)')
ax2.set_title('Forma Promedio de Spikes por Cluster')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=True)

# 5. Crear vectores binarios para cada spike-train separado
print("\n=== SPIKE TRAINS SEPARADOS POR CLUSTER ===")

# Crear vectores binarios para cada cluster
spike_trains = []
for i in range(n_clusters):
    # Vector binario de la misma longitud que la señal original
    binary_train = np.zeros(len(extracelular_filtrado))

    # Marcar spikes que pertenecen a este cluster
    cluster_mask = cluster_labels == i
    cluster_peaks = valid_peaks[cluster_mask]

    binary_train[cluster_peaks] = 1
    spike_trains.append(binary_train)

    print(f"✓ Spike train Cluster {i+1}: {len(cluster_peaks)} spikes")

# Graficar los spike trains separados
fig7, axes = plt.subplots(n_clusters + 1, 1, figsize=(14, 8), sharex=True)

# Subplot superior: Señal original con todos los spikes
axes[0].plot(vtime, extracelular_filtrado, 'k-', linewidth=0.6, alpha=0.7)
axes[0].plot(vtime[valid_peaks], extracelular_filtrado[valid_peaks], 'ro', markersize=2)
axes[0].set_title('Señal Extracelular con Todos los Spikes Detectados')
axes[0].set_ylabel('Amplitud (μV)')
axes[0].grid(True, alpha=0.3)

# Subplots para cada cluster
for i in range(n_clusters):
    # Mostrar eventos como líneas verticales
    cluster_mask = cluster_labels == i
    cluster_peaks = valid_peaks[cluster_mask]
    cluster_times = vtime[cluster_peaks]

    # Crear líneas verticales para cada spike
    for spike_time in cluster_times:
        axes[i+1].axvline(x=spike_time, color=colors[i], alpha=0.8, linewidth=1)

    # Alternativamente, mostrar como vector binario
    axes[i+1].plot(vtime, spike_trains[i], color=colors[i], linewidth=1)
    axes[i+1].set_title(f'Spike Train - Cluster {i+1} ({len(cluster_peaks)} spikes)')
    axes[i+1].set_ylabel(f'Neurona {i+1}')
    axes[i+1].set_ylim([-0.1, 1.1])
    axes[i+1].grid(True, alpha=0.3)

axes[-1].set_xlabel('Tiempo (s)')
plt.suptitle('Spike Trains Separados por Clustering')
plt.tight_layout()
plt.show(block=True)

# Estadísticas finales
print("\n=== RESUMEN FINAL ===")
print(f"✓ Total de spikes analizados: {len(valid_peaks)}")
print(f"✓ Número de clusters encontrados: {n_clusters}")
print(f"✓ Duración total del registro: {vtime[-1]:.2f} s")

# Calcular frecuencias de disparo
print("\nFrecuencias de disparo por neurona:")
for i in range(n_clusters):
    cluster_mask = cluster_labels == i
    n_spikes = np.sum(cluster_mask)
    freq = n_spikes / vtime[-1]  # Hz
    print(f"  - Neurona {i+1}: {freq:.2f} Hz ({n_spikes} spikes)")

# Matriz de correlación temporal entre spike trains
print("\nAnálisis de correlación entre spike trains:")
correlaciones = np.corrcoef(spike_trains)
print("Matriz de correlación:")
for i in range(n_clusters):
    for j in range(n_clusters):
        print(f"  Neurona {i+1} vs Neurona {j+1}: r = {correlaciones[i,j]:.3f}")

# Graficar matriz de correlación
fig8, ax = plt.subplots(1, 1, figsize=(8, 6))
im = ax.imshow(correlaciones, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(n_clusters))
ax.set_yticks(range(n_clusters))
ax.set_xticklabels([f'Neurona {i+1}' for i in range(n_clusters)])
ax.set_yticklabels([f'Neurona {i+1}' for i in range(n_clusters)])
ax.set_title('Matriz de Correlación entre Spike Trains')

# Agregar valores en las celdas
for i in range(n_clusters):
    for j in range(n_clusters):
        ax.text(j, i, f'{correlaciones[i,j]:.2f}',
               ha='center', va='center', color='black', fontweight='bold')

plt.colorbar(im, ax=ax, label='Coeficiente de Correlación')
plt.tight_layout()
plt.show(block=True)

print("\n¡Taller 3 completado exitosamente!")
print("\nCaracterísticas implementadas:")
print("✓ Detección de spikes extracelulares basada en Taller 2")
print("✓ Extracción de eventos con ventanas de ±1ms")
print("✓ Superposición de spikes (plot(eventos'))")
print("✓ Matriz de características: máximo, mínimo, pendiente")
print("✓ Visualización del espacio de características (2D y 3D)")
print("✓ Clustering K-means con normalización")
print("✓ Spike trains separados por neurona")
print("✓ Análisis de frecuencias y correlaciones")

print("\nTaller 3 finalizado. Cerrando todas las ventanas...")
plt.close('all')
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import os

# Configurar matplotlib para gráficas interactivas
plt.ion()

# 1. Cargar el archivo IntraExtracelular.mat
current_dir = os.path.dirname(os.path.abspath(__file__))
mat_file_path = os.path.join(current_dir, 'IntraExtracelular_py.mat')
data = scipy.io.loadmat(mat_file_path)

print("Variables en el archivo .mat:")
for key in data.keys():
    if not key.startswith('__'):
        print(f"  {key}: {data[key].shape if hasattr(data[key], 'shape') else type(data[key])}")

# 2. Extraer las variables
variable1 = data['variable1']  # Matriz 2x221077 (intracelular y extracelular)

# Extraer las señales
intracelular = variable1[0, :]  # Primer canal (intracelular)
extracelular = variable1[1, :]  # Segundo canal (extracelular)

# Extraer frecuencia de muestreo de SamplingRateHz
Fs = float(data['SamplingRateHz'][0][0])
print(f"\nFrecuencia de muestreo: {Fs} Hz")

# 3. Crear vector de tiempo
N = len(intracelular)
Ts = 1/Fs
tiempo_total = N * Ts
vtime = np.arange(0, tiempo_total, Ts)

# Ajustar longitud si es necesario
if len(vtime) > N:
    vtime = vtime[:N]

print(f"Número de muestras: {N}")
print(f"Tiempo total: {tiempo_total:.2f} s")

# 4. Graficar las señales intracelular y extracelular
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Señal intracelular
ax1.plot(vtime, intracelular, 'b-', linewidth=0.8)
ax1.set_title('Señal Intracelular')
ax1.set_ylabel('Amplitud (mV)')
ax1.grid(True, alpha=0.3)

# Señal extracelular
ax2.plot(vtime, extracelular, 'g-', linewidth=0.8)
ax2.set_title('Señal Extracelular')
ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('Amplitud (mV)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=True)

# 5. Detección de picos en señal intracelular
print("\n=== PROCESAMIENTO DE SEÑAL INTRACELULAR ===")

# Calcular umbral estadístico para la señal intracelular
mean_intra = np.mean(intracelular)
std_intra = np.std(intracelular)
threshold_intra = mean_intra + 2 * std_intra  # Umbral: media + 2 desviaciones estándar

print(f"Media señal intracelular: {mean_intra:.4f} mV")
print(f"Desviación estándar: {std_intra:.4f} mV")
print(f"Umbral calculado: {threshold_intra:.4f} mV")

# Aplicar find_peaks con el umbral calculado
peaks_intra, properties = find_peaks(intracelular,
                                   height=threshold_intra,
                                   distance=int(0.001*Fs),
                                   prominence=std_intra*0.5)  # Prominencia para reducir falsos positivos

print(f"Número de picos detectados: {len(peaks_intra)}")
print(f"Distancia mínima entre picos: {int(0.001*Fs)} muestras ({0.001*1000} ms)")

# Crear vector binario con las posiciones de los picos
binary_intra = np.zeros_like(intracelular)
binary_intra[peaks_intra] = 1

# Graficar la señal intracelular con los picos detectados
fig2, ax = plt.subplots(1, 1, figsize=(14, 6))

# Señal original
ax.plot(vtime, intracelular, 'b-', linewidth=0.8, label='Señal Intracelular')

# Línea de umbral
ax.axhline(y=threshold_intra, color='orange', linestyle='--', linewidth=1, label=f'Umbral ({threshold_intra:.4f} mV)')

# Picos detectados
ax.plot(vtime[peaks_intra], intracelular[peaks_intra], 'ro', markersize=4, label=f'Picos detectados ({len(peaks_intra)})')

ax.set_title('Detección de Picos - Señal Intracelular')
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Amplitud (mV)')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show(block=True)

# 6. Procesamiento de señal extracelular
print("\n=== PROCESAMIENTO DE SEÑAL EXTRACELULAR ===")

# Diseño del filtro para la señal extracelular
# Para señales extracelulares neuronales, típicamente se usa un filtro pasa-banda entre 300-3000 Hz
lowcut = 300.0   # Hz
highcut = 3000.0  # Hz
nyquist = Fs / 2
low = lowcut / nyquist
high = highcut / nyquist

print(f"Diseñando filtro Butterworth pasa-banda:")
print(f"  Frecuencia de corte inferior: {lowcut} Hz")
print(f"  Frecuencia de corte superior: {highcut} Hz")
print(f"  Frecuencia de Nyquist: {nyquist} Hz")

# Crear filtro Butterworth de 4to orden
order = 4
b, a = butter(order, [low, high], btype='band')

# Aplicar filtro con filtfilt (corrección automática de fase)
extracelular_filtrado = filtfilt(b, a, extracelular)

print("Filtro aplicado exitosamente con filtfilt (sin corrimiento de fase)")

# Graficar señal original vs filtrada
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Señal original
ax1.plot(vtime, extracelular, 'g-', linewidth=0.8)
ax1.set_title('Señal Extracelular - Original')
ax1.set_ylabel('Amplitud (μV)')
ax1.grid(True, alpha=0.3)

# Señal filtrada
ax2.plot(vtime, extracelular_filtrado, 'r-', linewidth=0.8)
ax2.set_title(f'Señal Extracelular - Filtrada ({lowcut}-{highcut} Hz)')
ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('Amplitud (μV)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=True)

# 7. Detección de picos en señal extracelular filtrada
# ESTRATEGIA PARA PICOS NEGATIVOS: Invertir la señal o usar el valor absoluto
print("\n=== DETECCIÓN DE PICOS EN SEÑAL EXTRACELULAR ===")

# Los picos extracelulares son típicamente negativos, por lo que invertimos la señal
extracelular_invertido = -extracelular_filtrado

# Calcular umbral estadístico para la señal filtrada invertida
mean_extra = np.mean(extracelular_invertido)
std_extra = np.std(extracelular_invertido)
threshold_extra = mean_extra + 2.5 * std_extra  # Umbral más estricto para extracellular

print(f"Media señal extracelular invertida: {mean_extra:.4f} μV")
print(f"Desviación estándar: {std_extra:.4f} μV")
print(f"Umbral calculado: {threshold_extra:.4f} μV")

# Aplicar find_peaks con parámetros adecuados para señales extracelulares
peaks_extra, properties = find_peaks(extracelular_invertido,
                                   height=threshold_extra,
                                   distance=int(0.001*Fs),  # Mínimo 1ms entre picos
                                   prominence=std_extra)    # Prominencia mínima

print(f"Número de picos detectados: {len(peaks_extra)}")

# Crear vector binario con las posiciones de los picos
binary_extra = np.zeros_like(extracelular_filtrado)
binary_extra[peaks_extra] = -1  # Usar -1 para indicar picos negativos

# Graficar la señal extracelular filtrada con los picos detectados
fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Señal invertida con picos
ax1.plot(vtime, extracelular_invertido, 'b-', linewidth=0.8, label='Señal Extracelular Invertida')
ax1.axhline(y=threshold_extra, color='orange', linestyle='--', linewidth=1, label=f'Umbral ({threshold_extra:.4f} μV)')
ax1.plot(vtime[peaks_extra], extracelular_invertido[peaks_extra], 'ro', markersize=4, label=f'Picos detectados ({len(peaks_extra)})')
ax1.set_title('Detección de Picos - Señal Extracelular Invertida')
ax1.set_ylabel('Amplitud (μV)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Subplot 2: Señal original filtrada con picos marcados
ax2.plot(vtime, extracelular_filtrado, 'g-', linewidth=0.8, label='Señal Extracelular Filtrada')
ax2.plot(vtime[peaks_extra], extracelular_filtrado[peaks_extra], 'ro', markersize=4, label=f'Picos detectados ({len(peaks_extra)})')
ax2.set_title('Señal Extracelular Filtrada con Picos Detectados')
ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('Amplitud (μV)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show(block=True)

# 8. Resumen final
print("\n=== RESUMEN FINAL ===")
print(f"Señal Intracelular:")
print(f"  - Picos detectados: {len(peaks_intra)}")
print(f"  - Umbral usado: {threshold_intra:.4f} mV")
print(f"  - Duración total: {tiempo_total:.2f} s")

print(f"\nSeñal Extracelular:")
print(f"  - Picos detectados: {len(peaks_extra)}")
print(f"  - Umbral usado: {threshold_extra:.4f} μV")
print(f"  - Filtro aplicado: {lowcut}-{highcut} Hz")
print(f"  - Estrategia: Inversión de señal para picos negativos")

# Crear figura comparativa final
fig5, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

# Intracelular - señal y picos
ax1.plot(vtime, intracelular, 'b-', linewidth=0.6)
ax1.plot(vtime[peaks_intra], intracelular[peaks_intra], 'ro', markersize=3)
ax1.set_title(f'Intracelular - {len(peaks_intra)} picos')
ax1.set_ylabel('Amplitud (mV)')
ax1.grid(True, alpha=0.3)

# Intracelular - vector binario
ax2.plot(vtime, binary_intra, 'r-', linewidth=1)
ax2.set_title('Vector Binario - Intracelular')
ax2.set_ylabel('Amplitud')
ax2.set_ylim([-0.1, 1.1])
ax2.grid(True, alpha=0.3)

# Extracelular - señal filtrada y picos
ax3.plot(vtime, extracelular_filtrado, 'g-', linewidth=0.6)
ax3.plot(vtime[peaks_extra], extracelular_filtrado[peaks_extra], 'ro', markersize=3)
ax3.set_title(f'Extracelular Filtrada - {len(peaks_extra)} picos')
ax3.set_xlabel('Tiempo (s)')
ax3.set_ylabel('Amplitud (μV)')
ax3.grid(True, alpha=0.3)

# Extracelular - vector binario
ax4.plot(vtime, binary_extra, 'r-', linewidth=1)
ax4.set_title('Vector Binario - Extracelular')
ax4.set_xlabel('Tiempo (s)')
ax4.set_ylabel('Amplitud')
ax4.set_ylim([-1.1, 0.1])
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=True)

print("\n¡Taller 2 completado exitosamente!")
print("\nCaracterísticas implementadas:")
print("✓ Carga de datos desde archivo .mat")
print("✓ Extracción de variables y frecuencia de muestreo")
print("✓ Gráficas con subplots y títulos")
print("✓ Umbral estadístico para detección de picos")
print("✓ Detección de picos con find_peaks()")
print("✓ Vectores binarios para localización de picos")
print("✓ Filtrado pasa-banda para señal extracelular")
print("✓ Estrategia para manejo de picos negativos")
print("✓ Visualización completa con overlays en rojo")

print("\nTaller 2 finalizado. Cerrando todas las ventanas...")
plt.close('all')
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Configurar matplotlib para gráficas interactivas
plt.ion()  # Modo interactivo

# 1. Cargar el archivo Data_Clase3_Vis.mat
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
mat_file_path = os.path.join(current_dir, 'Data_Clase3_Vis.mat')
data = scipy.io.loadmat(mat_file_path)

# 2. Extraer las señales biomédicas
co2 = data['co2'].flatten()
pleth = data['pleth'].flatten()
ecg = data['ecg'].flatten()
Fs = data['fs'][0][0]  # Frecuencia de muestreo

# 3. Calcular el tiempo de muestreo
Ts = 1/Fs

# 4. Construir vector de tiempo
N = len(ecg)  # Número de muestras (asumiendo que todas las señales tienen la misma longitud)
tiempo_total = N * Ts
vtime = np.arange(0, tiempo_total, Ts)

# Ajustar longitud si es necesario
if len(vtime) > N:
    vtime = vtime[:N]

# 5. Crear figura con 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Graficar CO2 (Respiración)
ax1.plot(vtime, co2, 'b-', linewidth=0.8)
ax1.set_title('CO2 - Respiración')
ax1.set_ylabel('Amplitud')
ax1.grid(True, alpha=0.3)

# Graficar ECG (Electrocardiograma)
ax2.plot(vtime, ecg, 'r-', linewidth=0.8)
ax2.set_title('ECG - Electrocardiograma')
ax2.set_ylabel('Amplitud')
ax2.grid(True, alpha=0.3)

# Graficar PLETH (Photoplethysmography)
ax3.plot(vtime, pleth, 'g-', linewidth=0.8)
ax3.set_title('PLETH - Photoplethysmography (Pulsioximetría)')
ax3.set_xlabel('Tiempo (s)')
ax3.set_ylabel('Amplitud')
ax3.grid(True, alpha=0.3)

# 6. sharex=True ya está implementado - equivalente a linkaxes de MATLAB
# Esto permite que el zoom en un subplot afecte los otros dos

plt.tight_layout()
plt.show()

# Mantener la ventana abierta para interacción
input("Presiona Enter para cerrar las gráficas...")

print(f"Frecuencia de muestreo: {Fs} Hz")
print(f"Tiempo de muestreo: {Ts:.6f} s")
print(f"Número de muestras: {N}")
print(f"Tiempo total: {tiempo_total:.2f} s")
print("\nLas gráficas son interactivas:")
print("- Usa el zoom para hacer zoom en cualquier subplot")
print("- El eje x está compartido, por lo que el zoom se aplicará a los 3 subplots")
print("- Usa los controles de matplotlib para navegar por las señales")
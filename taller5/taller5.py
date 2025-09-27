import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from scipy.io import loadmat

# --- Código visible en la imagen ---

# Cargar archivo .mat (reemplaza 'archivo.mat' con el nombre real)
matlab = loadmat('f_ECG.mat')
print("variables = ", list(matlab.keys()))
variables = matlab['variable3']
Variable = variables  # Descomentado y simplificado
Signalx = Variable[0]
Signalz = Variable[1]

# Asumiendo que Fs está en el archivo .mat
Fs = matlab['Fs']  # Ajusta el nombre según tu archivo
fs = Fs[0,0]
Ts = 1/fs
timex = len(Signalx)
timez = len(Signalz)
tievecx = np.arange(0, timex*Ts, Ts)
tievecz = np.arange(0, timez*Ts, Ts)

plt.figure(figsize = (8, 4))
plt.plot(tievecx, Signalx, 'b-')
plt.title('')
plt.tight_layout()
plt.show()

plt.figure(figsize = (8, 4))
plt.plot(tievecz, Signalz, 'b-')
plt.title('')
plt.tight_layout()
plt.show()

# --- Sección de ICA ---
ica = FastICA(n_components=2, random_state=0, max_iter=1000)
# Crear matriz X combinando las señales
X = np.column_stack((Signalx.flatten(), Signalz.flatten()))
X_transformed = ica.fit_transform(X)
A_estimated = ica.mixing_

print("Mixing matrix A:")
print(A_estimated)

# --- Gráficas de las señales estimadas ---
# Las siguientes líneas parecen graficar las señales separadas por ICA,
# aunque los nombres de las variables no coinciden exactamente con la salida de ICA.
# He corregido los nombres para que el código sea funcional.
estimated_signal_1 = X_transformed[:, 0]
estimated_signal_2 = X_transformed[:, 1]

# Crear vector de tiempo apropiado para las señales estimadas
time_vec = np.arange(0, len(estimated_signal_1)*Ts, Ts)

plt.figure(figsize = (8, 4))
plt.plot(time_vec, estimated_signal_1, 'b-')
plt.title('Señal Estimada 1')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.tight_layout()
plt.show()

plt.figure(figsize = (8, 4))
plt.plot(time_vec, estimated_signal_2, 'b-')
plt.title('Señal Estimada 2')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.tight_layout()
plt.show()
# Taller 1 - Señales Biomédicas

Este proyecto implementa en Python el ejercicio del Taller 1 para visualización de señales biomédicas.

## Configuración del Entorno

### 1. Activar el entorno virtual
```bash
source ../venv/bin/activate
```

### 2. Instalar dependencias (si es necesario)
```bash
pip install -r ../requirements.txt
```

## Archivos del Proyecto

- `taller1_python.py` - Script principal que implementa el ejercicio (equivalente a MATLAB)
- `Data_Clase3_Vis.mat` - Archivo de datos con las señales biomédicas

## Ejecución

```bash
python taller1_python.py
```

## Señales Incluidas

1. **CO2** - Señal de respiración
2. **ECG** - Electrocardiograma
3. **PLETH** - Photoplethysmography (Pulsioximetría)

## Características Implementadas

✅ Carga del archivo .mat
✅ Extracción de señales biomédicas
✅ Cálculo del vector de tiempo
✅ Visualización en 3 subplots
✅ Eje X compartido para zoom sincronizado
✅ Gráficas interactivas
✅ Títulos y etiquetas de ejes

## Información de las Señales

- **Frecuencia de muestreo:** 300 Hz
- **Número de muestras:** 144,001
- **Duración total:** 8 minutos (480 segundos)
- **Tiempo de muestreo:** 0.003333 s

## Funcionalidades Interactivas

- Zoom con selección de área o rueda del ratón
- Navegación con pan (clic y arrastrar)
- Eje X compartido entre los 3 subplots (equivalente a linkaxes de MATLAB)
- Controles de la barra de herramientas de matplotlib
- Resetear zoom con el botón "Home" o doble clic
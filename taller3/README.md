# Taller 3 - Spike Sorting y Clustering de Señales Neuronales

## Descripción
Script en Python que implementa spike sorting y clustering de señales neuronales extracelulares basado en el Taller 2, incluyendo extracción de características y separación de múltiples neuronas.

## Archivos
- `taller3_python.py` - Script principal con toda la implementación
- `taller3.txt` - Instrucciones originales del taller
- Depende de: `../taller2/IntraExtracelular_py.mat` - Datos del Taller 2

## Funcionalidades Implementadas

### ✅ 1. Detección de Spikes
- Reutilización del procesamiento del Taller 2
- Filtrado pasa-banda (300-3000 Hz)
- Detección de picos con umbral estadístico
- Manejo de picos negativos extracelulares

### ✅ 2. Matriz de Eventos
- Extracción de ventanas de ±1ms alrededor de cada spike
- Matriz: filas = spikes, columnas = tiempo (2ms total)
- Filtrado de spikes cerca de los bordes
- Superposición de todos los spikes (`plot(eventos')`)

### ✅ 3. Extracción de Características
- **Máximo**: Valor máximo del spike
- **Mínimo**: Valor mínimo del spike (pico negativo)
- **Pendiente**: Entre máximo y mínimo normalizada por tiempo
- Visualización 2D y 3D del espacio de características

### ✅ 4. Clustering K-means
- Normalización de características con StandardScaler
- Método del codo para determinar número óptimo de clusters
- K-means con 3 clusters (configurable)
- Visualización de clusters en espacio de características

### ✅ 5. Spike Trains Separados
- Vectores binarios para cada neurona identificada
- Visualización de spike trains individuales
- Análisis de frecuencias de disparo por neurona
- Matriz de correlación entre spike trains

## Dependencias
```bash
pip install scipy numpy matplotlib scikit-learn
```

## Ejecución
```bash
python taller3_python.py
```

## Resultados del Análisis

### Visualizaciones Generadas
1. **Detección de Spikes**: Señal extracelular con spikes marcados
2. **Superposición**: Todos los spikes alineados con promedio
3. **Espacio de Características**: Gráficas 2D de máximo vs mínimo, etc.
4. **Espacio 3D**: Visualización completa de las 3 características
5. **Método del Codo**: Para selección óptima de clusters
6. **Clusters**: Spikes coloreados según su cluster asignado
7. **Formas Promedio**: Morfología típica de cada neurona
8. **Spike Trains**: Vectores binarios separados por neurona
9. **Matriz de Correlación**: Análisis de sincronización entre neuronas

### Métricas Calculadas
- Número total de spikes detectados
- Distribución de spikes por cluster/neurona
- Frecuencias de disparo individuales (Hz)
- Coeficientes de correlación entre spike trains
- Estadísticas de características por cluster

## Características Técnicas
- **Ventana de análisis**: ±1ms (133 muestras a 66.67 kHz)
- **Características**: 3D (máximo, mínimo, pendiente)
- **Clustering**: K-means con normalización Z-score
- **Clusters**: 3 neuronas identificadas (configurable)
- **Visualización**: 9 gráficas interactivas secuenciales

## Notas Metodológicas
- Basado en técnicas estándar de spike sorting neuronal
- Manejo correcto de polaridad de spikes extracelulares
- Normalización de características para clustering robusto
- Análisis temporal y correlacional de spike trains
- Implementación completa siguiendo el flujo MATLAB→Python

## Flujo de Ejecución
1. Carga datos del Taller 2 → Detección de spikes
2. Extracción de eventos → Superposición visual
3. Cálculo de características → Espacio 3D
4. K-means clustering → Separación de neuronas
5. Análisis de spike trains → Correlaciones

¡Script listo para análisis completo de spike sorting!
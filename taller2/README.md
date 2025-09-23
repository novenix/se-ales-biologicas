# Taller 2 - Procesamiento de Señales Intracelulares y Extracelulares

## Descripción
Script en Python que implementa el procesamiento y análisis de señales neuronales intracelulares y extracelulares, incluyendo detección de picos y filtrado digital.

## Archivos
- `taller2_python.py` - Script principal con toda la implementación
- `IntraExtracelular_py.mat` - Datos de señales neuronales
- `taller2.txt` - Instrucciones originales del taller

## Funcionalidades Implementadas

### ✅ Carga y Visualización
- Carga de archivo MATLAB (.mat)
- Extracción de señales intracelular y extracelular
- Gráficas con subplots y títulos apropiados
- Vector de tiempo con frecuencia de muestreo

### ✅ Señal Intracelular
- Cálculo de umbral estadístico (media + 2σ)
- Detección de picos con `find_peaks()`
- Vector binario con localización de picos
- Visualización con picos superpuestos en rojo

### ✅ Señal Extracelular
- Filtro pasa-banda Butterworth (300-3000 Hz)
- Uso de `filtfilt()` para corrección de fase
- Estrategia para picos negativos (inversión de señal)
- Detección de picos con parámetros optimizados
- Visualización completa del procesamiento

## Ejecución
```bash
python taller2_python.py
```

## Características Técnicas
- **Frecuencia de muestreo**: 66.67 kHz
- **Duración**: ~3.3 segundos
- **Filtro extracelular**: Butterworth orden 4, 300-3000 Hz
- **Umbral intracelular**: Media + 2 desviaciones estándar
- **Umbral extracelular**: Media + 2.5 desviaciones estándar
- **Separación mínima**: 1 ms entre picos

## Notas
- Script interactivo con pausas para revisión de resultados
- Gráficas optimizadas para análisis visual
- Manejo adecuado de picos negativos en señales extracelulares
- Implementación siguiendo las mejores prácticas de procesamiento neuronal
#!/usr/bin/env python3
"""
Taller 2 - Procesamiento de Señales Intracelulares y Extracelulares
Detección de picos usando findpeaks con prominence y filtrado de señales
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import find_peaks, butter, filtfilt, freqz
import os

# Configurar matplotlib para gráficas secuenciales
plt.ioff()  # Modo no interactivo (bloqueante)

def main():
    """Script principal del Taller 2"""

    print("🧠 TALLER 2 - PROCESAMIENTO DE SEÑALES ELECTROFISIOLÓGICAS")
    print("🔬 Detección de picos con findpeaks y filtrado de señales")
    print("=" * 70)

    try:
        # ============================================================
        # 1. CARGAR DATOS DEL ARCHIVO .MAT
        # ============================================================
        print("=" * 60)
        print("CARGANDO DATOS DEL ARCHIVO .MAT")
        print("=" * 60)

        # Obtener la ruta del archivo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mat_file_path = os.path.join(current_dir, 'IntraExtracelular_py.mat')

        # Cargar el archivo .mat
        data = loadmat(mat_file_path)

        # Explorar las variables
        print("\nClaves en el archivo .mat:")
        for key in data.keys():
            if not key.startswith('__'):
                print(f"  {key}: {type(data[key])}, shape: {data[key].shape if hasattr(data[key], 'shape') else 'N/A'}")

        # Extraer las variables principales
        variable1 = data['variable1']  # Matriz de 2 canales
        fs = float(data['SamplingRateHz'][0][0])  # Frecuencia de muestreo
        samples = data['Samples'][0][0] if 'Samples' in data else None

        print(f"\nvariable1 shape: {variable1.shape}")
        print(f"Frecuencia de muestreo: {fs} Hz")
        if samples is not None:
            print(f"Número de muestras reportado: {samples}")

        # Separar las señales (transponer si es necesario)
        if variable1.shape[0] == 2:  # Si las señales están en filas
            señal_intracelular = variable1[0, :]
            señal_extracelular = variable1[1, :]
        else:  # Si las señales están en columnas
            señal_intracelular = variable1[:, 0]
            señal_extracelular = variable1[:, 1]

        # Crear vector de tiempo
        n_muestras = len(señal_intracelular)
        tiempo = np.arange(n_muestras) / fs

        print(f"\nNúmero de muestras: {n_muestras:,}")
        print(f"Duración de la señal: {tiempo[-1]:.2f} segundos")
        print(f"Rango señal intracelular: [{señal_intracelular.min():.6f}, {señal_intracelular.max():.6f}] V")
        print(f"Rango señal extracelular: [{señal_extracelular.min():.6f}, {señal_extracelular.max():.6f}] V")

        # ============================================================
        # 2. GRAFICAR SEÑALES ORIGINALES
        # ============================================================
        print("\n📊 Graficando señales originales...")

        # Crear subplots con sharex=True (equivalente a linkaxes de MATLAB)
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Graficar señal intracelular
        ax1.plot(tiempo, señal_intracelular, 'b-', linewidth=0.8)
        ax1.set_title('Señal Intracelular')
        ax1.set_ylabel('Amplitud (V)')
        ax1.grid(True, alpha=0.3)

        # Graficar señal extracelular
        ax2.plot(tiempo, señal_extracelular, 'g-', linewidth=0.8)
        ax2.set_title('Señal Extracelular')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Amplitud (V)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # ============================================================
        # 3. DETECTAR PICOS EN SEÑAL INTRACELULAR
        # ============================================================
        print("\n🔵 PROCESANDO SEÑAL INTRACELULAR")
        print("-" * 40)

        # Calcular umbral estadístico
        media_intra = np.mean(señal_intracelular)
        std_intra = np.std(señal_intracelular)
        umbral_intra = media_intra + 3 * std_intra  # Umbral a 3 desviaciones estándar

        print(f"Estadísticas señal intracelular:")
        print(f"  Media: {media_intra:.6f} V")
        print(f"  Desviación estándar: {std_intra:.6f} V")
        print(f"  Umbral (media + 3*std): {umbral_intra:.6f} V")

        # Detectar picos usando find_peaks con prominence
        picos_intra, propiedades_intra = find_peaks(
            señal_intracelular,
            height=umbral_intra,
            prominence=std_intra,  # Usar prominence para mejor detección
            distance=int(0.001 * fs)  # Distancia mínima entre picos (1ms)
        )

        print(f"\nPicos detectados: {len(picos_intra)}")
        if len(picos_intra) > 0:
            print(f"Alturas: rango [{propiedades_intra['peak_heights'].min():.6f}, {propiedades_intra['peak_heights'].max():.6f}] V")
            print(f"Prominencias: rango [{propiedades_intra['prominences'].min():.6f}, {propiedades_intra['prominences'].max():.6f}] V")
            print(f"Frecuencia de picos: {len(picos_intra)/(tiempo[-1]/60):.1f} picos/minuto")

        # Crear vector binario de picos
        vector_picos_intra = np.zeros_like(señal_intracelular)
        vector_picos_intra[picos_intra] = 1

        print(f"Vector de picos creado: {np.sum(vector_picos_intra)} picos marcados")

        # Graficar detección de picos intracelulares
        print("\n📈 Graficando detección de picos intracelulares...")

        fig2, ax = plt.subplots(figsize=(12, 6))

        # Señal original
        ax.plot(tiempo, señal_intracelular, 'b-', linewidth=0.8, label='Señal Intracelular')

        # Línea de umbral
        ax.axhline(y=umbral_intra, color='orange', linestyle='--',
                   label=f'Umbral: {umbral_intra:.6f} V')

        # Picos detectados
        if len(picos_intra) > 0:
            ax.plot(tiempo[picos_intra], señal_intracelular[picos_intra],
                    'ro', markersize=6, label=f'Picos ({len(picos_intra)})')

        ax.set_title('Detección de Picos - Señal Intracelular')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Amplitud (V)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

        # ============================================================
        # 4. DISEÑAR FILTRO PARA SEÑAL EXTRACELULAR
        # ============================================================
        print("\n🔧 DISEÑANDO FILTRO PARA SEÑAL EXTRACELULAR")
        print("-" * 50)

        # Parámetros del filtro pasa-banda
        fc_low = 100   # Frecuencia de corte baja (Hz)
        fc_high = 300  # Frecuencia de corte alta (Hz)
        nyquist = fs / 2
        orden = 4

        # Normalizar frecuencias
        low = fc_low / nyquist
        high = fc_high / nyquist

        # Crear filtro Butterworth
        b, a = butter(orden, [low, high], btype='band')

        print(f"Filtro pasa-banda diseñado:")
        print(f"  Frecuencia de corte baja: {fc_low} Hz")
        print(f"  Frecuencia de corte alta: {fc_high} Hz")
        print(f"  Orden del filtro: {orden}")
        print(f"  Frecuencia de Nyquist: {nyquist} Hz")

        # Ver respuesta en frecuencia del filtro
        w, h = freqz(b, a, worN=8000)
        freq_hz = w * fs / (2 * np.pi)

        # Graficar respuesta en frecuencia
        fig3, ax = plt.subplots(figsize=(10, 6))
        ax.plot(freq_hz, 20 * np.log10(abs(h)), 'purple', linewidth=2)
        ax.axvline(x=fc_low, color='red', linestyle='--', label=f'fc_low = {fc_low} Hz')
        ax.axvline(x=fc_high, color='red', linestyle='--', label=f'fc_high = {fc_high} Hz')

        ax.set_title('Respuesta en Frecuencia del Filtro Pasa-Banda')
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('Magnitud (dB)')
        ax.set_xlim([0, 5000])
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

        # ============================================================
        # 5. APLICAR FILTRO A SEÑAL EXTRACELULAR
        # ============================================================
        print("\n⚡ Aplicando filtro con filtfilt...")

        # Aplicar filtro usando filtfilt (corrección automática de fase)
        señal_extra_filtrada = filtfilt(b, a, señal_extracelular)

        print(f"Señal extracelular filtrada:")
        print(f"  Rango original: [{señal_extracelular.min():.6f}, {señal_extracelular.max():.6f}] V")
        print(f"  Rango filtrada: [{señal_extra_filtrada.min():.6f}, {señal_extra_filtrada.max():.6f}] V")

        # Comparar señal original vs filtrada
        fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Usar un subconjunto de datos para mejor visualización
        t_inicio, t_fin = 10, 15  # Mostrar 5 segundos
        fs_calc = len(tiempo) / tiempo[-1]
        idx_inicio = int(t_inicio * fs_calc)
        idx_fin = int(t_fin * fs_calc)

        # Señal original
        ax1.plot(tiempo[idx_inicio:idx_fin], señal_extracelular[idx_inicio:idx_fin],
                 'g-', linewidth=1, label='Original')
        ax1.set_title('Señal Extracelular Original')
        ax1.set_ylabel('Amplitud (V)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Señal filtrada
        ax2.plot(tiempo[idx_inicio:idx_fin], señal_extra_filtrada[idx_inicio:idx_fin],
                 'r-', linewidth=1, label='Filtrada')
        ax2.set_title('Señal Extracelular Filtrada')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Amplitud (V)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # ============================================================
        # 6. DETECTAR PICOS EN SEÑAL EXTRACELULAR
        # ============================================================
        print("\n🟢 PROCESANDO SEÑAL EXTRACELULAR")
        print("-" * 40)

        # Para detectar picos negativos, invertimos la señal
        señal_extra_invertida = -señal_extra_filtrada

        # Calcular umbral estadístico
        media_extra = np.mean(señal_extra_filtrada)
        std_extra = np.std(señal_extra_filtrada)
        umbral_extra = media_extra - 4 * std_extra  # Umbral más estricto para extracelular

        print(f"Estadísticas señal extracelular filtrada:")
        print(f"  Media: {media_extra:.6f} V")
        print(f"  Desviación estándar: {std_extra:.6f} V")
        print(f"  Umbral para picos negativos (media - 4*std): {umbral_extra:.6f} V")

        # Detectar picos en la señal invertida (para encontrar picos negativos)
        picos_extra, propiedades_extra = find_peaks(
            señal_extra_invertida,
            height=-umbral_extra,  # Invertir el umbral
            prominence=2*std_extra,  # Prominence más alta para extracelular
            distance=int(0.001 * fs)  # Distancia mínima de 1ms
        )

        print(f"\nPicos detectados en señal extracelular: {len(picos_extra)}")
        if len(picos_extra) > 0:
            alturas_reales = señal_extra_filtrada[picos_extra]  # Alturas en la señal original (negativas)
            print(f"Alturas de los picos (valores negativos): rango [{alturas_reales.min():.6f}, {alturas_reales.max():.6f}] V")
            print(f"Prominencias: rango [{propiedades_extra['prominences'].min():.6f}, {propiedades_extra['prominences'].max():.6f}] V")
            print(f"Frecuencia de picos: {len(picos_extra)/(tiempo[-1]/60):.1f} picos/minuto")

        # Crear vector binario de picos extracelulares
        vector_picos_extra = np.zeros_like(señal_extra_filtrada)
        vector_picos_extra[picos_extra] = -1  # Usar -1 para indicar picos negativos

        print(f"Vector de picos extracelulares creado: {np.sum(np.abs(vector_picos_extra))} picos marcados")

        # Graficar detección de picos extracelulares
        print("\n📈 Graficando detección de picos extracelulares...")

        fig5, ax = plt.subplots(figsize=(12, 6))

        # Mostrar un segmento más pequeño para mejor visualización
        t_start, t_end = 20, 25  # 5 segundos
        idx_start = int(t_start * fs)
        idx_end = int(t_end * fs)

        # Señal filtrada
        ax.plot(tiempo[idx_start:idx_end], señal_extra_filtrada[idx_start:idx_end],
                'g-', linewidth=1, label='Señal Extracelular Filtrada')

        # Línea de umbral
        ax.axhline(y=umbral_extra, color='orange', linestyle='--',
                   label=f'Umbral: {umbral_extra:.6f} V')

        # Filtrar picos que están en el rango de visualización
        picos_en_rango = picos_extra[(picos_extra >= idx_start) & (picos_extra < idx_end)]

        if len(picos_en_rango) > 0:
            # Picos detectados
            ax.plot(tiempo[picos_en_rango], señal_extra_filtrada[picos_en_rango],
                    'ro', markersize=8, label=f'Picos Negativos ({len(picos_en_rango)} en rango)')

        ax.set_title(f'Detección de Picos Negativos - Señal Extracelular (t={t_start}-{t_end}s)')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Amplitud (V)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

        # ============================================================
        # 7. VISUALIZACIÓN FINAL COMPLETA
        # ============================================================
        print("\n🎯 CREANDO VISUALIZACIÓN FINAL COMPLETA")
        print("-" * 45)

        # Crear subplots con sharex=True
        fig6, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

        # Subconjunto de datos para visualización (primeros 30 segundos)
        t_max = 30
        idx_max = int(t_max * fs)
        t_vis = tiempo[:idx_max]
        intra_vis = señal_intracelular[:idx_max]
        extra_vis = señal_extracelular[:idx_max]
        extra_filt_vis = señal_extra_filtrada[:idx_max]
        picos_intra_vis = vector_picos_intra[:idx_max]
        picos_extra_vis = vector_picos_extra[:idx_max]

        # Filtrar picos en el rango de visualización
        picos_intra_rango = picos_intra[picos_intra < idx_max]
        picos_extra_rango = picos_extra[picos_extra < idx_max]

        # 1. Señal intracelular con picos
        ax1.plot(t_vis, intra_vis, 'b-', linewidth=0.8, label='Intracelular')
        if len(picos_intra_rango) > 0:
            ax1.plot(tiempo[picos_intra_rango], señal_intracelular[picos_intra_rango],
                    'ro', markersize=4, label=f'Picos Intra ({len(picos_intra_rango)})')
        ax1.set_title('Señal Intracelular con Picos Detectados')
        ax1.set_ylabel('Amplitud (V)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Señal extracelular original
        ax2.plot(t_vis, extra_vis, 'g-', linewidth=0.8, label='Extra Original')
        ax2.set_title('Señal Extracelular Original')
        ax2.set_ylabel('Amplitud (V)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Señal extracelular filtrada con picos
        ax3.plot(t_vis, extra_filt_vis, 'darkgreen', linewidth=0.8, label='Extra Filtrada')
        if len(picos_extra_rango) > 0:
            ax3.plot(tiempo[picos_extra_rango], señal_extra_filtrada[picos_extra_rango],
                    'ro', markersize=4, label=f'Picos Extra ({len(picos_extra_rango)})')
        ax3.set_title('Señal Extracelular Filtrada con Picos Detectados')
        ax3.set_ylabel('Amplitud (V)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Señales binarias
        ax4.plot(t_vis, picos_intra_vis, 'r-', linewidth=1, label='Picos Intra Binarios')
        ax4.plot(t_vis, picos_extra_vis, 'purple', linewidth=1, label='Picos Extra Binarios')
        ax4.set_title('Señales Binarias de Picos')
        ax4.set_xlabel('Tiempo (s)')
        ax4.set_ylabel('Binario')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        plt.show()

        # ============================================================
        # 8. RESUMEN FINAL
        # ============================================================
        print("\n" + "=" * 60)
        print("RESUMEN DEL ANÁLISIS DE SEÑALES ELECTROFISIOLÓGICAS")
        print("=" * 60)

        print(f"\n📊 DATOS GENERALES:")
        print(f"   • Frecuencia de muestreo: {fs:.0f} Hz")
        print(f"   • Número de muestras: {len(señal_intracelular):,}")
        print(f"   • Duración total: {tiempo[-1]:.2f} segundos")

        print(f"\n🔵 SEÑAL INTRACELULAR:")
        print(f"   • Rango de amplitud: [{señal_intracelular.min():.6f}, {señal_intracelular.max():.6f}] V")
        print(f"   • Media: {media_intra:.6f} V")
        print(f"   • Desviación estándar: {std_intra:.6f} V")
        print(f"   • Umbral utilizado: {umbral_intra:.6f} V")
        print(f"   • Picos detectados: {len(picos_intra)}")
        print(f"   • Frecuencia de picos: {len(picos_intra)/(tiempo[-1]/60):.1f} picos/minuto")

        print(f"\n🟢 SEÑAL EXTRACELULAR:")
        print(f"   • Rango original: [{señal_extracelular.min():.6f}, {señal_extracelular.max():.6f}] V")
        print(f"   • Rango filtrada: [{señal_extra_filtrada.min():.6f}, {señal_extra_filtrada.max():.6f}] V")
        print(f"   • Filtro aplicado: Pasa-banda {fc_low}-{fc_high} Hz, orden 4")
        print(f"   • Media (filtrada): {media_extra:.6f} V")
        print(f"   • Desviación estándar (filtrada): {std_extra:.6f} V")
        print(f"   • Umbral utilizado: {umbral_extra:.6f} V")
        print(f"   • Picos negativos detectados: {len(picos_extra)}")
        print(f"   • Frecuencia de picos: {len(picos_extra)/(tiempo[-1]/60):.1f} picos/minuto")

        print(f"\n🔧 PARÁMETROS DE DETECCIÓN:")
        print(f"   • Prominence intracelular: {std_intra:.6f} V")
        print(f"   • Prominence extracelular: {2*std_extra:.6f} V")
        print(f"   • Distancia mínima entre picos: {int(0.001 * fs)} muestras (1 ms)")

        print(f"\n✅ ANÁLISIS COMPLETADO")
        print(f"   • find_peaks utilizado con parámetros height y prominence")
        print(f"   • Filtrado con filtfilt para corrección automática de fase")
        print(f"   • Estrategia para picos negativos: inversión de señal")
        print(f"   • Visualizaciones interactivas generadas con matplotlib")

        if len(picos_intra) > 0 and len(picos_extra) > 0:
            ratio = len(picos_extra) / len(picos_intra)
            print(f"\n📈 COMPARACIÓN:")
            print(f"   • Ratio picos extra/intra: {ratio:.2f}")
            if ratio > 2:
                print(f"   • La señal extracelular muestra mayor actividad de picos")
            elif ratio < 0.5:
                print(f"   • La señal intracelular muestra mayor actividad de picos")
            else:
                print(f"   • Actividad de picos similar entre ambas señales")

        print(f"\n🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
        print(f"   📊 Total de gráficos generados: 6")
        print(f"   🔵 Picos intracelulares detectados: {len(picos_intra)}")
        print(f"   🟢 Picos extracelulares detectados: {len(picos_extra)}")

        print("\nLas gráficas son interactivas:")
        print("- Usa el zoom para hacer zoom en cualquier subplot")
        print("- Los ejes x están compartidos (sharex=True)")
        print("- Cierra cada gráfico para ver el siguiente")

    except FileNotFoundError:
        print("❌ ERROR: No se encontró el archivo 'IntraExtracelular_py.mat'")
        print("   Asegúrate de que el archivo esté en el directorio actual.")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("   Revisa que tengas instaladas todas las librerías necesarias:")
        print("   pip install numpy matplotlib scipy")

if __name__ == "__main__":
    main()
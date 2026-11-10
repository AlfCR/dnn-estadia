# Desarrollo y Validación de Redes Neuronales Profundas (DNN) para el Pronóstico de Carga del Motor en Mantenimiento Predictivo

El presente proyecto implementa y evalúa arquitecturas de redes neuronales recurrentes (**LSTM** y **GRU**) desarrolladas en TensorFlow, con el objetivo de realizar el **pronóstico a 3 segundos de la carga del motor** a partir de series temporales multivariadas obtenidas de sensores OBD-II. 

> **Importante:** Dado que el conjunto de datos carece de etiquetas de falla explícitas, el entrenamiento se aborda mediante **el aprendizaje supervisado de regresión**, modelando el comportamiento dinámico y la proyección temporal de la carga del motor para la detección temprana de anomalías.

---

## Estado y Evolución del Proyecto

| Fase | Descripción Técnica | Entregables / Resultados | Rutas de Acceso |
| :--- | :--- | :--- | :--- |
| **Procesamiento de Datos IoT Vehiculares** | Limpieza de series temporales, remuestreo uniforme a 1 Hz, segmentación por viajes completos, normalización y generación de tensores de ventana temporal (3D). | Script de preprocesamiento, pipeline interactivo y especificación del dataset. | [`src/preprocesamiento.py`](src/preprocesamiento.py)<br>[`notebooks/dnn_estadia_ACP.ipynb`](notebooks/dnn_estadia_ACP.ipynb)<br>[`README_dataset.txt`](README_dataset.txt) |
| **Diseño y Entrenamiento de Modelos DNN** | Implementación modular de arquitecturas LSTM y GRU en TensorFlow, con esquemas de entrenamiento robustos evaluados a través de 5 semillas aleatorias. | Módulos de arquitectura y notebooks independientes de entrenamiento por modelo. | [`src/modelos.py`](src/modelos.py)<br>[`notebooks/LSTM_ENTRENAMIENTO_ACP.ipynb`](notebooks/LSTM_ENTRENAMIENTO_ACP.ipynb)<br>[`notebooks/GRU_ENTRENAMIENTO_ACP.ipynb`](notebooks/GRU_ENTRENAMIENTO_ACP.ipynb) |
| **Validación, Ajuste y Pruebas** | Evaluación de rendimiento predictivo, análisis de métricas de error (RMSE/MAE) y validación cruzada de hiperparámetros. | *En desarrollo* | *Por definir* |

# Pasos para la ejecución del repositorio:

## Clonar el repositorio:
- git clone https://github.com/AlfCR/dnn-estadia.git
- cd dnn-estadia


## Crear entorno virtual
1. Windows: 
python -m venv venv 

## Estructura del Repositorio

```text

├── notebooks/                  # Cuadernos de Google Colab
│   ├── dnn_estadia_ACP.ipynb           # Pipeline de procesamiento.
│   ├── LSTM_ENTRENAMIENTO_ACP.ipynb    # Entrenamiento y evaluación LSTM 
│   └── GRU_ENTRENAMIENTO_ACP.ipynb     # Entrenamiento y evaluación GRU 
│
├── src/                      
│   ├── modelos.py              # Arquitecturas LSTM y GRU documentadas
│   └── preprocesamiento.py     # Script automatizado de limpieza
│
├── README.md                   # Este archivo
└── requirements.txt            # Dependencias del proyecto (TensorFlow, Pandas, etc.)

"""Arquitecturas LSTM y GRU para pronosticar la carga del motor.
"""

# 1. IMPORTS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential

# Capas necesarias para construir las arquitecturas LSTM y GRU (Autoencoder / Pronóstico) 
from tensorflow.keras.layers import ( Input, LSTM, GRU, Dense, Dropout)

# Optimizador para ajustar los pesos del modelo durante el entrenamiento 
from tensorflow.keras.optimizers import Adam

# 2. CONSTANTES
VENTANA = 30 # Tamaño de la ventana temporal
CARACTERISTICAS = 23 #Caracteristicas
NEURONAS = 64
DROPOUT = 0.2 # Tasa de avandono, ayuda a evitar el sobreajuste, apagando una cantidad de neuronas en cada entrenamiento.


# 3. CLASE DEL MODELO
class ModeloPronostico:
    def __init__(self, tipo = 'LSTM', ventana = VENTANA, caracteristicas = CARACTERISTICAS, neuronas = NEURONAS, dropout= DROPOUT):
        # Guardar parametros
        self.tipo = tipo.upper()
        self.ventana = ventana
        self.caracteristicas = caracteristicas
        self.neuronas = neuronas
        self.dropout = dropout

        # Validar el tipo del modelo, solo se usan dos para el proyecto.
        if self.tipo not in ['LSTM', 'GRU']:
            raise ValueError(f"El tipo '{self.tipo}' no es válido. Debe ser 'LSTM' o 'GRU'")
        
        #Construcción del modelo y guardad
        self.modelo = self._construir_modelo()

    def _construir_modelo(self):

        """Método interno que crea las capas de la red neuronal"""
        modelo = Sequential()

        # 1. Capa de entrada
        modelo.add(Input(shape=(self.ventana, self.caracteristicas)))

        # 2. Capa recurrente  (LSTM o GRU)
        
        if self.tipo == 'LSTM':
            modelo.add(LSTM(self.neuronas, return_sequences=False))
        else: #Para el caso de trabajar con GRU
            modelo.add(GRU(self.neuronas, return_sequences=False))    

        #NOTA: El return_sequences=False, solo necesitamos la predicción final de la secuencia,
        # no la secuencia completa porque estamos haciendo un pronóstico de un solo paso (1 valor de salida).


        # 3. Dropout para evitar el sobreajuste (Memorizar)
        modelo.add(Dropout(self.dropout))

        # 4. Capas densas 
        modelo.add(Dense(16, activation='relu')) # 16 -> cantidad de neuronas por capas | relu -> no linealidad en la red (Para que la red aprenda)

        # 5. Capas de salida (1 neurona sin activación porque es regresión para predecir un valor continuo.)
        modelo.add(Dense(1)) 

        # Compilación
        modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return modelo
    def resumen(self):
        """Ver el resumen del modelo en pantalla"""
        return self.modelo.summary()

# 4. FUNCIÓN QUE CREA EL MODELO
def crear_modelo(tipo = 'LSTM', ventana = VENTANA, caracteristicas = CARACTERISTICAS, neuronas = NEURONAS, dropout = DROPOUT):
    """
    Se instancia la clase ModeloPronostico 
    y devuelve el objeto Sequential de Keras listo para entrenar.
    """
    constructor = ModeloPronostico (tipo=tipo, ventana=ventana, caracteristicas=caracteristicas, neuronas=neuronas, dropout=dropout)
    return constructor.modelo

# 5. UTILIDADES
def obtener_callbacks(paciencia=10):
    """
    Devuelve una lista con el callback de parada temprana.
    Evita el sobreajuste deteniendo el entrenamiento si la pérdida de validación no mejora.
    """
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=paciencia, restore_best_weights=True, verbose=1)
    
    return [early_stop]
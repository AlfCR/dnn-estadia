"""Arquitecturas LSTM y GRU para pronosticar la carga del motor.
"""

# 1. IMPORTS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam


# 2. CONSTANTES
VENTANA = 30          # Tamaño de la ventana temporal
CARACTERISTICAS = 23  # Caracteristicas
NEURONAS = 64
DROPOUT = 0.2 # Tasa de abandono, ayuda a evitar el sobreajuste, apagando una cantidad de neuronas en cada entrenamiento.
LEARNING_RATE = 0.001 # Tasa de aprendizaje, controla la velocidad a la que el modelo aprende.
CAPAS = 1 # Número de capas recurrentes (LSTM o GRU) en la red neuronal.
OPTIMIZADOR = 'adam' #  'adam', 'rmsprop' o 'nadam'
OPTIMIZADORES = {'adam': Adam, 'rmsprop': RMSprop, 'nadam': Nadam}

# 3. CLASE DEL MODELO
class ModeloPronostico:
    def __init__(self, tipo = 'LSTM', ventana = VENTANA, caracteristicas = CARACTERISTICAS, neuronas = NEURONAS, dropout= DROPOUT, learning_rate = LEARNING_RATE, capas = CAPAS, optimizador = OPTIMIZADOR):
        # Guardar parametros
        self.tipo = tipo.upper()
        self.ventana = ventana
        self.caracteristicas = caracteristicas
        self.neuronas = neuronas
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.capas = capas
        self.optimizador = optimizador.lower()

        # Validar el tipo del modelo, solo se usan dos para el proyecto.
        if self.tipo not in ['LSTM', 'GRU']:
            raise ValueError(f"El tipo '{self.tipo}' no es válido. Debe ser 'LSTM' o 'GRU'.")

        if self.optimizador not in OPTIMIZADORES:
            raise ValueError(f"Optimizador '{self.optimizador}' no es válido. Debe ser uno de {list(OPTIMIZADORES.keys())}")
        
        if self.capas < 1:
            raise ValueError(f"El número de capas debe ser al menos 1 o más.")

        #Construcción del modelo y guardado. 
        self.modelo = self._construir_modelo()

    def _construir_modelo(self):

        """Método interno que crea las capas de la red neuronal"""
        Recurrente = LSTM if self.tipo == 'LSTM' else GRU
        modelo = Sequential()

        # 1. Capa de entrada
        modelo.add(Input(shape=(self.ventana, self.caracteristicas)))

        # 2. Capa recurrente  (LSTM o GRU)
        for i in range(self.capas):
            es_ultima_capa = (i == self.capas - 1)
            modelo.add(Recurrente(self.neuronas, return_sequences=not es_ultima_capa))

        # 3. Dropout para evitar el sobreajuste (Memorizar)
        modelo.add(Dropout(self.dropout))

        # 4. Capas densas 
        modelo.add(Dense(16, activation='relu')) # 16 -> cantidad de neuronas por capas | relu -> no linealidad en la red (Para que la red aprenda)

        # 5. Capas de salida (1 neurona sin activación porque es regresión para predecir un valor continuo.)
        modelo.add(Dense(1)) 

        # Compilación
        opt = OPTIMIZADORES[self.optimizador](learning_rate=self.learning_rate)
        modelo.compile(optimizer=opt, loss='mse', metrics=['mae'])
        return modelo
    
    def resumen(self):
        return self.modelo.summary()

# 4. FUNCIÓN QUE CREA EL MODELO
def crear_modelo(tipo = 'LSTM', ventana = VENTANA, caracteristicas = CARACTERISTICAS, neuronas = NEURONAS, dropout = DROPOUT, learning_rate = LEARNING_RATE, capas = CAPAS, optimizador = OPTIMIZADOR):
    """
    Se instancia la clase ModeloPronostico 
    y devuelve el objeto Sequential de Keras listo para entrenar.
    """
    constructor = ModeloPronostico (tipo=tipo, ventana=ventana, caracteristicas=caracteristicas, neuronas=neuronas, dropout=dropout, learning_rate=learning_rate, capas=capas, optimizador=optimizador)
    return constructor.modelo

# 5. UTILIDADES
def obtener_callbacks(paciencia=10):
    """
    Devuelve una lista con el callback de parada temprana.
    Evita el sobreajuste deteniendo el entrenamiento si la pérdida de validación no mejora.
    """
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=paciencia, restore_best_weights=True, verbose=1)
    return [early_stop]
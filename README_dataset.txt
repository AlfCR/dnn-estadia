Dataset para ser consumido por la neurona: dataset_preparado_3d.npz 

Descripción:
Este archivo es un .npz de NumPy que contiene 8 arrays
para entrenar la red neuronal de la Fase 3:

  X_train        → (6959, 30, 23)    ventanas de entrenamiento
  y_train        → (6959,)            objetivo de entrenamiento
  X_val          → (1343, 30, 23)    ventanas de validación
  y_val          → (1343,)            objetivo de validación
  X_test         → (2342, 30, 23)    ventanas de prueba
  y_test         → (2342,)            objetivo de prueba
  columnas       → (23,)               nombres de características
  parametros     → [30, 3, 10]        ventana, horizonte, móvil


NOTA IMPORTANTE:
Si usan Google Drive para visualizar el contenido del archivo,
puede mostrar solo 2 de los 8 archivos internos (X_train.npy y X_test.npy). 
Esto es normal: es una limitación del visor de Drive, no del archivo.

Para verificar el contenido completo:

    import numpy as np
    datos = np.load('dataset_preparado_3d.npz')
    print(datos.files)

Salida esperada:
['X_train', 'y_train', 'X_val', 'y_val', 'X_test',
 'y_test', 'columnas', 'parametros']

Verificación de formas:

    print(datos['X_train'].shape)   # (6959, 30, 23)
    print(datos['X_val'].shape)     # (1343, 30, 23)
    print(datos['X_test'].shape)    # (2342, 30, 23)

Generado por: Alfonso Calderón Pérez
Proyecto: DNN para Mantenimiento Predictivo
Institución: INAOE — Ciencias Computacionales
Fecha: Octubre 2026
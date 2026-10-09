import csv
import io
import hashlib
import json 
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ===== CONSTANTES =====

MESES = {
    'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5,
    'jun': 6, 'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10,
    'nov': 11, 'dic': 12
}

# Configuración por defecto de ventanas ()
VENTANA_SEG_DEFAULT = 30
HORIZONTE_SEG_DEFAULT = 3
VENTANA_MOVIL_SEG_DEFAULT = 10

# Variables base del modelo
VARIABLES_BASE = [
    'rpm',
    'carga_motor_pct',
    'temp_refrigerante_C',
    'temp_aire_admision_C',
    'presion_multiple_kPa',
    'acelerador_b_pct',
    'afr',
    'velocidad_obd_kmh',
    'voltaje_adaptador_V',
    'velocidad_gps_kmh',
    'acel_total_ms2',   
]

# 3. FUNCIONES
'''
1. Devuelve un valor como texto limpio.

Si el valor es nulo devuelve ''; si no, lo convierte a texto, 
cambia el espacio no separable por espacio normal y elimina 
espacios al inicio y al final.
'''
def normalizar_texto(valor):
    if pd.isna(valor):
        return ''
    return str(valor).replace('\u00a0', ' ').strip()


''' 
2.  Lee un archivo CSV y devuelve (filas, detalle).
'''
def leer_filas(ruta):
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f'No existe el archivo: {ruta}')


    if ruta.suffix.lower() != '.csv':
        raise ValueError(f'Solo se aceptan archivos .CSV: {ruta.name}')

    for cod in ('utf-8-sig', 'cp1252'):
        try:
            with open(ruta, 'r', encoding=cod, newline='') as f:
                texto = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f'No se pudo leer el archivo {ruta.name}')

    # Detectar el delimitador ejemplos de delimitadores: ',', ';', '\t', '|'
    try:
        sep = csv.Sniffer().sniff(texto[:8192], delimiters=',;\t|').delimiter
    except csv.Error:
        sep = ','  # Valor por defecto si no se puede detectar

    # Convertir el texto en filas (listas de valores) y devolverlas junto con el detalle del archivo
    filas = list(csv.reader(io.StringIO(texto, newline=''), delimiter=sep))
    return filas, f'csv: separador={sep!r}, encoding={cod}'


'''
3. Dividir las filas en bloques
'''
def partir_en_bloques(filas):
    
    #Filtrar filas completamente vacías
    pares = [
        (n, f) for n, f in enumerate(filas, start=1) if any(normalizar_texto(x) for x in f)
    ]

    if not pares:
        return []

    # Detectar el inicio de cada bloque
    primer_campo = normalizar_texto(pares[0][1][0])
    bloques = []

    for n, fila in pares:
        if normalizar_texto(fila[0]) == primer_campo:
            bloques.append({
                'linea': n,
                'encabezado': [normalizar_texto(x) for x in fila],
                'filas': []
            })
        else:
            bloques[-1]['filas'].append([normalizar_texto(x) for x in fila])

    return bloques


def huella_bloque(encabezado, filas):

    #Ancho del encabezado
    ancho = len(encabezado)

    #Listas de filas normalizadas
    filas_norm = []
    for fila in filas:
        if len(fila) > ancho:
            raise ValueError(f'Fila con {len(fila)} campos y encabezado con {ancho} columnas')
        filas_norm.append(fila + [''] * (ancho - len(fila)))
        
    contenido = json.dumps(
        [encabezado, filas_norm],
        ensure_ascii=False,
        separators=(',', ':'),
        )
    
    huella = hashlib.sha256(contenido.encode('utf-8')).hexdigest()
    
    return huella, filas_norm

# ==== FUNCIONES DE CARGA ======

def catalogar_viajes(ruta):
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f'No existe la carpeta: {ruta}')

    #Una huella es un bloque de datos conformado 
    #por el encabezado y filas de un mismo formato. 
    #Se puede repetir en varios archivos.
    
    bloques_por_huella = {} 
    for archivo in sorted(ruta.iterdir()):
        if archivo.suffix.lower() != '.csv':
            continue

        filas, _ = leer_filas(archivo)
        bloques = partir_en_bloques(filas)

        for numero_bloque, bloque in enumerate(bloques, start=1):
            if not bloque['filas']:
                continue  # Ignorar bloques sin filas

            encabezado = bloque['encabezado']
            filas_bloque = bloque['filas']
            huella, filas_norm = huella_bloque(encabezado, filas_bloque)

            # Si la huella es nueva, agregarla al catálogo
            if huella not in bloques_por_huella:
                bloques_por_huella[huella] = {
                    'encabezado': encabezado,
                    'filas': filas_norm,
                    'fuentes': [],
                }

            # Registrar la fuente (incluso si ya existía)
            bloques_por_huella[huella]['fuentes'].append(
                f'{archivo.name} (bloque {numero_bloque})'
            )

    return bloques_por_huella

# === FUNCIONES DE CARGA DE VIAJES ======
def cargar_viajes_unicos(catalogo, fuente, viaje_num_inicial=1):
    partes = [] #Inicializar lista de partes de DataFrame
    viaje_num = viaje_num_inicial

    for huella, bloque in catalogo.items():
        encabezado = bloque['encabezado'] #Extraer el encabezado de las columnas.
        filas = bloque['filas'] #Extraer las filas de datos.

        # Saltar bloques vacíos
        if not filas:
            continue

        # ID corto y estable del viaje
        id_viaje = f'V{huella[:16]}'

        # Crear DataFrame del bloque
        df_bloque = pd.DataFrame(filas, columns=encabezado)

        #Agregar columnas al inicio
        df_bloque.insert(0, 'viaje_num', viaje_num)
        df_bloque.insert(1, 'fuente', fuente)
        df_bloque.insert(2, 'id_viaje', id_viaje)

        partes.append(df_bloque)
        viaje_num += 1

    # Concatenar todas las partes en un solo DataFrame
    if not partes:
        return pd.DataFrame()  # Retornar DataFrame vacío si no hay partes

    # Unión de todas las partes en un solo DataFrame.
    df_final = pd.concat(partes, ignore_index=True, sort=False)
    return df_final


# === FUNCIONES DE CONVERSIÓN DE UNIDADES ======

def fahrenheit_a_celsius(temp_f):
    # Convierte un valor de temperatura de Fahrenheit a Celsius.
    return (temp_f - 32) * 5.0 / 9.0

def mph_a_kmh(vel_mph):
    # Convierte un valor de velocidad de millas por hora (mph) a kilómetros por hora (km/h).
    return vel_mph * 1.609344
    
def psi_a_kpa(presion_psi):
    # Convierte un valor de presión de libras por pulgada cuadrada (psi) a kilopascales (kPa).
    return presion_psi * 6.894757

def ms_a_kmh(vel_ms):
    # Convierte un valor de velocidad de metros por segundo (m/s) a kilómetros por hora (km/h).
    return vel_ms * 3.6

def convertir_fecha(texto):
    try:
        partes = str(texto).strip().lower().split()
        if len(partes) < 2:
            return pd.NaT
        fecha, hora = partes[0].replace('.', ''), partes[1]
        dia, mes_txt, anio = fecha.split('-')
        mes = MESES[mes_txt]
        return pd.to_datetime(f'{anio}-{mes:02d}-{int(dia):02d} {hora}')
    except (ValueError, IndexError, KeyError):
        return pd.NaT
        

def sin_cambio(valor):
    # Función de conversión que devuelve el valor sin cambios. (Para variables que no requieren conversión de unidades)
    return valor

def limpiar_dataset(df_raw):
        # Diccionario: columna_original → (nuevo_nombre, función_conversión)
    conversiones = {
        'Engine RPM(rpm)':                     ('rpm', sin_cambio),
        'Engine Load(%)':                      ('carga_motor_pct', sin_cambio),
        'Engine Coolant Temperature(°F)':      ('temp_refrigerante_C', fahrenheit_a_celsius),
        'Intake Air Temperature(°F)':          ('temp_aire_admision_C', fahrenheit_a_celsius),
        'Intake Manifold Pressure(psi)':       ('presion_multiple_kPa', psi_a_kpa),
        'Throttle Position(Manifold)(%)':      ('acelerador_manifold_pct', sin_cambio),
        'Absolute Throttle Position B(%)':     ('acelerador_b_pct', sin_cambio),
        'Air Fuel Ratio(Measured)(:1)':        ('afr', sin_cambio),
        'Speed (OBD)(mph)':                    ('velocidad_obd_kmh', mph_a_kmh),
        'Voltage (OBD Adapter)(V)':            ('voltaje_adaptador_V', sin_cambio),
        'GPS Speed (Meters/second)':           ('velocidad_gps_kmh', ms_a_kmh),
        'G(x)':                                ('g_x_ms2', sin_cambio),
        'G(y)':                                ('g_y_ms2', sin_cambio),
        'G(z)':                                ('g_z_ms2', sin_cambio),
    }

    # Extraer las columnas que están en el DataFrame y que tienen una conversión definida
    columnas_necesarias = ['viaje_num', 'id_viaje', 'fuente', 'Device Time'] + list(conversiones.keys())
    df_filtrado = df_raw[columnas_necesarias].copy()

    # Construir el Dataframe limpio
    df_limpio = pd.DataFrame()
    df_limpio['viaje_num'] = df_filtrado['viaje_num']
    df_limpio['id_viaje'] = df_filtrado['id_viaje']
    df_limpio['fuente'] = df_filtrado['fuente']


    # Aplicar conversiones 
    for col_original, (col_nueva, funcion) in conversiones.items():
        # Convertir a numérico y aplicar la función
        valores = pd.to_numeric(df_filtrado[col_original], errors='coerce')
        df_limpio[col_nueva] = funcion(valores)

    # Calcular aceleración total
    df_limpio['acel_total_ms2'] = (
        np.sqrt(
            df_limpio['g_x_ms2']**2
            + df_limpio['g_y_ms2']**2
            + df_limpio['g_z_ms2']**2
        ) - 9.80665
    )

    # Parsear timestamp
    df_limpio['timestamp'] = df_filtrado['Device Time'].apply(convertir_fecha)

    # Verificar que todas las fechas tuvieron conversion
    assert df_limpio['timestamp'].notna().all(), \
        "Hay fechas que no se pudieron convertir"

    # Redondear a 2 decimales
    columnas_numericas = [c for c in df_limpio.columns 
                          if c not in ['viaje_num', 'id_viaje', 'fuente', 'timestamp']]
    df_limpio[columnas_numericas] = df_limpio[columnas_numericas].round(2)

    # Este elemento nos ayuda a identificar si el viaje tiene datos del motor.
    df_limpio['tiene_motor'] = (
            df_limpio.groupby('viaje_num')['rpm']
            .transform(lambda s: s.notna().any())
        )

    # Eliminar duplicados
    antes = len(df_limpio)
    df_limpio = df_limpio.drop_duplicates(subset=['id_viaje', 'timestamp'])
    print(f'Duplicados eliminados: {antes - len(df_limpio)}')

    # Ordenar
    df_limpio = df_limpio.sort_values(['viaje_num', 'timestamp']).reset_index(drop=True)

    

    return df_limpio

# segmentar_y_remuestrear
def segmentar_y_remuestrear(
        df, 
        tiempo_recorte_seg=10,
        hueco_max_seg=5,
        rango_interpolacion=3,
        duracion_minima_seg=30
        ):
    """
    Segmenta cada viaje por huecos temporales y remuestrea a 1 Hz.

    - Recorta los primeros segundos de cada viaje.
    - Divide en segmentos cuando hay huecos > `hueco_max_seg`.
    - Remuestrea a 1 Hz promediando por segundo.
    - Interpola vacíos cortos dentro de cada segmento.
    - Descarta segmentos más cortos que `duracion_minima_seg`.

    Args:
        df (pd.DataFrame): DataFrame limpio.
        tiempo_recorte_seg (int): Segundos a recortar al inicio.
        hueco_max_seg (int): Umbral de hueco para abrir segmento.
        rango_interpolacion (int): Máximo de vacíos a interpolar.
        duracion_minima_seg (int): Duración mínima del segmento.

    Returns:
        pd.DataFrame: DataFrame segmentado y a 1 Hz.
    """
    
    inicio = df.groupby('id_viaje')['timestamp'].transform('min')
    df = df[df['timestamp'] >= inicio + pd.Timedelta(seconds=tiempo_recorte_seg)].copy()

    # Ordenar y detectar saltos temporales
    df = df.sort_values(['viaje_num', 'timestamp']).reset_index(drop=True)

    dt = df.groupby('id_viaje')['timestamp'].diff().dt.total_seconds()

    salto = (dt > hueco_max_seg) | dt.isna()

    df['segmento_id'] = salto.groupby(df['id_viaje']).cumsum() 

    # Remuestrear a 1 Hz promediando por segundo
    df['timestamp_1s'] = df['timestamp'].dt.floor('1s')

    # Columnas de metadata (se conservan)
    columnas_meta = ['viaje_num', 'id_viaje', 'fuente', 'segmento_id', 'tiene_motor']
    columnas_num = [c for c in df.columns 
                    if c not in columnas_meta + ['timestamp', 'timestamp_1s']]

    df_1hz = (df.groupby(columnas_meta + ['timestamp_1s'])[columnas_num]
                .mean()
                .reset_index()
                .rename(columns={'timestamp_1s': 'timestamp'}))

    # Interpolar vacíos cortos por segmento
    partes = []

    for (viaje, segmento), g in df_1hz.groupby(['id_viaje', 'segmento_id']):
        meta = g.iloc[0][columnas_meta]
        g = g.set_index('timestamp').resample('1s').asfreq()
        for c in columnas_meta:
            g[c] = meta[c]
        # Interpolar columnas numéricas
        g[columnas_num] = g[columnas_num].interpolate(
            method='linear', limit=rango_interpolacion
        )
        partes.append(g.reset_index())

    df_interp = pd.concat(partes, ignore_index=True)

    # Descartar segmentos de menos de N segundos
    df_interp['timestamp'] = pd.to_datetime(df_interp['timestamp'])
    duracion_seg = df_interp.groupby(['id_viaje', 'segmento_id'])['timestamp'].transform('size')
    df_final = df_interp[duracion_seg >= duracion_minima_seg].copy()

    # Redondear a 2 decimales
    df_final[columnas_num] = df_final[columnas_num].round(2)

    # Ordenar
    df_final = df_final.sort_values(['viaje_num', 'segmento_id', 'timestamp'])
    df_final = df_final.reset_index(drop=True)

    return df_final
    
# particionar
def particionar(df):
    """
    Asigna cada viaje a una partición.

    Args:
        df (pd.DataFrame): DataFrame segmentado.

    Returns:
        pd.DataFrame: Mismo DataFrame + columna 'particion'.
    """
    particiones_config = {
        'entrenamiento':  [1, 2, 3, 4, 8, 9],
        'validacion':     [7],
        'prueba':         [5, 6],
        'prueba_externa': [16],
        'sin_motor':      [10, 11, 12, 13, 15, 18],
    }

    todos = [v for viajes in particiones_config.values() for v in viajes]
    assert len(todos) == len(set(todos)), \
        "Hay viajes asignados a más de una partición."

    mapa_viajes = {
        viaje: particion
        for particion, viajes in particiones_config.items()
        for viaje in viajes
    }

    df = df.copy()
    df['particion'] = df['viaje_num'].map(mapa_viajes)

    assert df['particion'].notna().all(), \
        f"Viajes sin asignar: {sorted(df.loc[df['particion'].isna(), 'viaje_num'].unique())}"

    return df


# normalizar
def normalizar(df, variables=VARIABLES_BASE):

    modelo = df['particion'].isin(['entrenamiento', 'validacion', 'prueba'])
    assert df.loc[modelo, variables].notna().all().all(), \
        "Hay vacíos en las filas del modelo."

    # Filtrar solo entrenamiento
    train = df[df['particion'] == 'entrenamiento']

    # Calcular media y desviación
    media = train[variables].mean()
    desviacion = train[variables].std()

    # Verificar que no haya desviaciones cero
    assert (desviacion > 0).all(), \
        f"Variables con desviación cero: {desviacion[desviacion == 0].index.tolist()}"

    # Copiar y aplicar z-score
    df_norm = df.copy()
    df_norm[variables] = (df_norm[variables] - media) / desviacion

    # Verificar
    z_train = df_norm[df_norm['particion'] == 'entrenamiento'][variables]
    assert np.allclose(z_train.mean(), 0, atol=1e-6), "La media de train no es 0."
    assert np.allclose(z_train.std(), 1, atol=1e-6), "La desviación de train no es 1."

    # Diccionario de parámetros
    parametros = {
        'media': media.to_dict(),
        'desviacion': desviacion.to_dict(),
    }

    return df_norm, parametros

# Generar ventanas
def generar_ventanas(df_norm,
                     ventana_seg=VENTANA_SEG_DEFAULT,
                     horizonte_seg=HORIZONTE_SEG_DEFAULT,
                     ventana_movil_seg=VENTANA_MOVIL_SEG_DEFAULT,
                     variables_modelo=VARIABLES_BASE,
                     variables_moviles=None):
    
    if variables_moviles is None:
        variables_moviles = ['rpm', 'carga_motor_pct',
                             'acelerador_b_pct', 'velocidad_gps_kmh']

    particiones_modelo = ['entrenamiento', 'validacion', 'prueba']
    df_modelo = df_norm[df_norm['particion'].isin(particiones_modelo)].copy()
    df_modelo = df_modelo.sort_values(
        ['viaje_num', 'segmento_id', 'timestamp']
    ).reset_index(drop=True)

    # Características móviles
    nuevas_columnas = []
    for var in variables_moviles:
        col_mean = f'{var}_mean{ventana_movil_seg}'
        df_modelo[col_mean] = (
            df_modelo.groupby(['viaje_num', 'segmento_id'])[var]
            .transform(lambda x: x.rolling(ventana_movil_seg).mean())
        )
        col_std = f'{var}_std{ventana_movil_seg}'
        df_modelo[col_std] = (
            df_modelo.groupby(['viaje_num', 'segmento_id'])[var]
            .transform(lambda x: x.rolling(ventana_movil_seg).std())
        )
        col_diff = f'{var}_diff1'
        df_modelo[col_diff] = (
            df_modelo.groupby(['viaje_num', 'segmento_id'])[var]
            .transform(lambda x: x.diff(1))
        )
        nuevas_columnas.extend([col_mean, col_std, col_diff])

    caracteristicas = variables_modelo + nuevas_columnas

    # Eliminar NaN
    df_modelo = df_modelo.dropna(subset=caracteristicas).reset_index(drop=True)

    # Generar ventanas
    x_list = {'entrenamiento': [], 'validacion': [], 'prueba': []}
    y_list = {'entrenamiento': [], 'validacion': [], 'prueba': []}

    for (viaje, segmento), df_seg in df_modelo.groupby(['viaje_num', 'segmento_id']):
        particion = df_seg['particion'].iloc[0]
        data_x = df_seg[caracteristicas].values
        data_y = df_seg['carga_motor_pct'].values
        n_filas = len(df_seg)

        for i in range(n_filas - ventana_seg - horizonte_seg + 1):
            x_win = data_x[i : i + ventana_seg]
            objetivo = data_y[i + ventana_seg + horizonte_seg - 1]
            x_list[particion].append(x_win)
            y_list[particion].append(objetivo)

    X_train = np.array(x_list['entrenamiento'])
    y_train = np.array(y_list['entrenamiento'])
    X_val = np.array(x_list['validacion'])
    y_val = np.array(y_list['validacion'])
    X_test = np.array(x_list['prueba'])
    y_test = np.array(y_list['prueba'])

    # Verificar
    for nombre, arreglo in [('X_train', X_train), ('y_train', y_train),
                            ('X_val', X_val), ('y_val', y_val),
                            ('X_test', X_test), ('y_test', y_test)]:
        assert not np.isnan(arreglo).any(), f'Hay NaN en {nombre}'

    for nombre, X in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test)]:
        assert X.shape[1:] == (ventana_seg, len(caracteristicas)), \
            f'Forma incorrecta en {nombre}: {X.shape}'

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'columnas': caracteristicas,
        'parametros': [ventana_seg, horizonte_seg, ventana_movil_seg],
    }

def parsear_argumentos():
    parser = argparse.ArgumentParser(
        description='Preprocesamiento de datos de telemetría vehicular.'
    )
    parser.add_argument(
        '--measurement',
        required=True,
        help='Carpeta "Measurement complementary".'
    )
    parser.add_argument(
        '--chiapas',
        required=True,
        help='Carpeta "Mediciones Agosto 2024 Chiapas".'
    )
    parser.add_argument(
        '--salida',
        required=True,
        help='Carpeta donde se guardan los resultados.'
    )
    return parser.parse_args()


def main():
    args = parsear_argumentos()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    # 1. Cargar los viajes únicos de ambas fuentes
    cat_m = catalogar_viajes(args.measurement)
    cat_c = catalogar_viajes(args.chiapas)
    df_m = cargar_viajes_unicos(cat_m, 'Complementary', viaje_num_inicial=1)
    df_c = cargar_viajes_unicos(cat_c, 'Chiapas', viaje_num_inicial=len(cat_m) + 1)
    df_raw = pd.concat([df_m, df_c], ignore_index=True, sort=False)

    assert df_raw.groupby('id_viaje')['viaje_num'].nunique().max() == 1, \
        "Un viaje tiene dos números."
    assert df_raw['viaje_num'].nunique() == df_raw['id_viaje'].nunique(), \
        "viaje_num repetido en viajes distintos."
    print(f"1. Crudo:   {len(df_raw)} filas, {df_raw['viaje_num'].nunique()} viajes")

    # 2. Limpiar
    df_limpio = limpiar_dataset(df_raw)
    print(f"2. Limpio:  {len(df_limpio)} filas")

    # 3. Segmentar y remuestrear
    df_1hz = segmentar_y_remuestrear(df_limpio)
    print(f"3. A 1 Hz:  {len(df_1hz)} filas, {df_1hz['viaje_num'].nunique()} viajes")

    # 4. Particionar
    df_part = particionar(df_1hz)

    # 5. Normalizar
    df_norm, parametros = normalizar(df_part)

    # 6. Ventanas
    res = generar_ventanas(df_norm)
    print(f"6. Ventanas: {len(res['X_train'])} / {len(res['X_val'])} / {len(res['X_test'])}")

    # 7. Guardar
    pd.DataFrame({
        'variable': list(parametros['media']),
        'media': list(parametros['media'].values()),
        'desviacion': list(parametros['desviacion'].values()),
    }).to_csv(salida / 'parametros_normalizacion.csv', index=False)

    np.savez_compressed(
        salida / 'dataset_preparado_3d.npz',
        X_train=res['X_train'], y_train=res['y_train'],
        X_val=res['X_val'], y_val=res['y_val'],
        X_test=res['X_test'], y_test=res['y_test'],
        columnas=np.array(res['columnas']),
        parametros=np.array(res['parametros']),
    )
    print(f"Resultados guardados en: {salida}")


if __name__ == '__main__':
    main()
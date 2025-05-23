import pandas as pd
import pickle
from datetime import datetime

# ----------- 1. Cargar el modelo -----------
try:
    with open("model/mejor_modelo.pkl", "rb") as archivo_modelo:
        modelo = pickle.load(archivo_modelo)
    print("✅ Modelo cargado exitosamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit()

# ----------- 2. Crear datos para predecir -----------
# Ejemplo: fecha futura
nueva_fecha = "2025-06-15"  # yyyy-mm-dd
linea_servicio = "L1"       # Cambia según las líneas que entrenaste

# Procesar la fecha
fecha_dt = pd.to_datetime(nueva_fecha)
entrada = pd.DataFrame([{
    "día_mes": fecha_dt.day,
    "mes": fecha_dt.month,
    "día_semana": fecha_dt.weekday(),
    "Línea de servicio": linea_servicio
}])

# Convertir columna categórica a variables dummy (debe coincidir con el entrenamiento)
entrada = pd.get_dummies(entrada)

# Cargar los nombres de las columnas originales del modelo (opcional pero recomendable)
try:
    with open("model/columnas_entrenamiento.pkl", "rb") as f:
        columnas_modelo = pickle.load(f)
    # Asegurar que las columnas estén en el mismo orden y estructura
    for col in columnas_modelo:
        if col not in entrada.columns:
            entrada[col] = 0  # agregar columnas faltantes con 0
    entrada = entrada[columnas_modelo]
except:
    print("⚠️ No se cargó el archivo de columnas. Asegúrate que las columnas de entrada coincidan.")

# ----------- 3. Hacer predicción -----------
try:
    prediccion = modelo.predict(entrada)
    horas = [f"{h}:00" for h in range(4, 23)]
    resultado = pd.DataFrame(prediccion, columns=horas)
    print("✅ Predicción realizada:")
    print(resultado.T)  # mostrar horas en filas
except Exception as e:
    print(f"❌ Error al hacer la predicción: {e}")

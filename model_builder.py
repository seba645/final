import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def cargar_datos(ruta):
    try:
        df = pd.read_csv(ruta, encoding='latin1', sep=';')
        print(f"✅ Datos cargados. Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        print("Columnas disponibles:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")
        return None

def preparar_datos(df, columna_objetivo):
    try:
        # Rellenar NaN con media por línea
        horas = ['4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00', '12:00', '13:00',
                 '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00']

        df[horas] = df.groupby('Línea de servicio')[horas].transform(lambda x: x.fillna(x.mean()))

        # Convertir variables categóricas a dummies (excepto la variable objetivo)
        df = pd.get_dummies(df, columns=['Día', 'dia/semana'], drop_first=True)

        X = df.drop(columns=[columna_objetivo])
        y = df[columna_objetivo]
        return X, y
    except Exception as e:
        print(f"❌ Error al preparar los datos: {e}")
        return None, None

def dividir_datos(X, y, test_size=0.3, random_state=42):
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    except Exception as e:
        print(f"❌ Error al dividir los datos: {e}")
        return None, None, None, None

def buscar_mejor_modelo(X, y):
    modelos = {
        'LogisticRegression': {
            'modelo': LogisticRegression(max_iter=1000),
            'parametros': {'modelo__C': [0.01, 0.1, 1, 10]}
        },
        'RandomForest': {
            'modelo': RandomForestClassifier(),
            'parametros': {'modelo__n_estimators': [50, 100], 'modelo__max_depth': [None, 10, 20]}
        },
        'SVC': {
            'modelo': SVC(),
            'parametros': {'modelo__C': [0.1, 1, 10], 'modelo__kernel': ['linear', 'rbf']}
        },
        'KNN': {
            'modelo': KNeighborsClassifier(),
            'parametros': {'modelo__n_neighbors': [3, 5, 7]}
        },
        'DecisionTree': {
            'modelo': DecisionTreeClassifier(),
            'parametros': {'modelo__max_depth': [None, 5, 10]}
        }
    }

    mejor_modelo = None
    mejor_score = 0
    mejor_nombre = ""

    for nombre, config in modelos.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('modelo', config['modelo'])
        ])
        grid = GridSearchCV(pipeline, config['parametros'], cv=5, scoring='accuracy')

        try:
            grid.fit(X, y)
            score = grid.best_score_
            print(f"🔍 {nombre} - Score CV: {score:.4f}")
            if score > mejor_score:
                mejor_modelo = grid.best_estimator_
                mejor_score = score
                mejor_nombre = nombre
        except Exception as e:
            print(f"❌ Error entrenando {nombre}: {e}")

    if mejor_modelo:
        print(f"✅ Mejor modelo: {mejor_nombre} con score {mejor_score:.4f}")
    else:
        print("❌ No se encontró modelo válido.")

    return mejor_modelo

def evaluar_modelo(pipeline, X_test, y_test):
    try:
        y_pred = pipeline.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        print("\n✅ Resultados del modelo:")
        print("Accuracy:", metrics['accuracy'])
        print("Precision:", metrics['precision'])
        print("Recall:", metrics['recall'])
        print("F1 Score:", metrics['f1_score'])
        print("\nMatriz de confusión:\n", np.array(metrics['confusion_matrix']))
        print("\nReporte de clasificación:\n", metrics['classification_report'])

        return metrics
    except Exception as e:
        print(f"❌ Error al evaluar el modelo: {e}")
        return {}

def guardar_modelo_pickle(modelo, nombre_archivo="data/mejor_modelo.pkl"):
    try:
        with open(nombre_archivo, "wb") as f:
            pickle.dump(modelo, f)
        print(f"✅ Modelo guardado exitosamente como '{nombre_archivo}'")
    except Exception as e:
        print(f"❌ Error al guardar el modelo: {e}")

def main():
    ruta_archivo = "data/Libro1.csv"  # ← Cambia si tu archivo tiene otro nombre o extensión
    columna_objetivo = "Línea de servicio"  # ← Variable a predecir

    df = cargar_datos(ruta_archivo)
    if df is None:
        return

    X, y = preparar_datos(df, columna_objetivo)
    if X is None or y is None:
        return

    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    if X_train is None:
        return

    best_pipeline = buscar_mejor_modelo(X_train, y_train)
    if best_pipeline is None:
        return

    guardar_modelo_pickle(best_pipeline, "model/mejor_modelo.pkl")
    evaluar_modelo(best_pipeline, X_test, y_test)

if __name__ == "__main__":
    main()

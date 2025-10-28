# Proyecto Modelo Regresión en Python

# Cargamos las librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Leemos el archivo Excel
archivo = "TU RUTA CON LA CARPETA Y EL ARCHIVO EXCEL"
datos = pd.read_excel(archivo, sheet_name="Datos")

# Convertimos variable categórica a tipo categoría
datos["Tamaño_Empresa"] = datos["Tamaño_Empresa"].astype("category")

# Variables predictoras y objetivo
X = datos.drop(columns="Salario")
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding para variables categóricas
y = datos["Salario"]

# Separamos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Inicializamos los modelos
modelos = {
    "Regresión Lineal": LinearRegression(),
    "Árbol de Decisión": DecisionTreeRegressor(random_state=123),
    "Random Forest": RandomForestRegressor(random_state=123),
    "KNN": KNeighborsRegressor()
}

# Evaluamos los modelos
resultados = []

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))


    resultados.append({
        "Modelo": nombre,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    })


# Mostramos los resultados
df_resultados = pd.DataFrame(resultados)
print(df_resultados)



# Ejemplo: mostrar las predicciones del modelo que elijas

modelo_elegido = modelos["Regresión Lineal"]  # podés cambiarlo por otro
predicciones = modelo_elegido.predict(X_test)

# Crear un DataFrame con los valores reales y predichos
df_pred = pd.DataFrame({
    "Salario_Real": y_test,
    "Salario_Predicho": predicciones
})

print(df_pred.head(10))  # muestra las primeras 10 filas
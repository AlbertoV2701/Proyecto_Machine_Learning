# Proyecto_Machine_Learning
Proyecto realizado en curso de IPS Datax sobre Ciencia de Datos con Python y R.

🧠 Descripción General del Proyecto

Este proyecto tiene como objetivo desarrollar y comparar modelos predictivos para estimar el salario de una persona en función de diferentes variables, como su experiencia laboral, años de educación y el tamaño de la empresa.
Se implementaron dos versiones del mismo análisis: una en R y otra en Python, con el propósito de aplicar las mismas técnicas en ambos lenguajes y observar similitudes y diferencias en el flujo de trabajo.

## Carga y exploración de datos

Se importó un archivo Excel (`Datos_Modelo.xlsx`) con la información base.

Se revisaron los tipos de variables y se realizaron ajustes necesarios (por ejemplo, convertir variables categóricas a **factores** en R o a **categorías** en Python).

---

## Modelos de regresión iniciales

Se probaron modelos de **regresión lineal** con distintas combinaciones de variables para comprender su relación con el salario.

---

## Implementación de Machine Learning

Se aplicaron modelos más avanzados usando librerías especializadas:

* **En R:** `caret`
* **En Python:** `scikit-learn`

Los modelos utilizados fueron:

* Regresión Lineal
* Árbol de Decisión
* Random Forest
* K-Nearest Neighbors (KNN)

---

## División de los datos y evaluación

Los datos se separaron en conjuntos de **entrenamiento (80%)** y **prueba (20%)**.

Cada modelo fue evaluado mediante métricas como:

* **R²** (Coeficiente de Determinación)
* **MAE** (Error Absoluto Medio)
* **RMSE** (Raíz del Error Cuadrático Medio)

---

## Comparación de resultados

Finalmente, se compararon las métricas de todos los modelos para determinar cuál ofrece el mejor rendimiento predictivo.


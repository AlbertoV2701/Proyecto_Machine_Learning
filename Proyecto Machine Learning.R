# Proyecto Modelo Regresión

# Cargamos los datos
library(readxl)
Datos_Modelo <- read_excel("Tu RUTA DE LA CARPETA CON EL EXCEL", 
                           sheet = "Datos")

# Modelo real de machine learning

# Necesitamos:
# - Separar los datos en entrenamiento y testeo
# - Entrenar varios modelos a la vez
# - Evaluar estos modelos con los datos de testeo
# - Concluir cuál es mejor

# Instalamos los paquetes (si no están instalados)
install.packages("tidyverse")
install.packages("caret")

# Cargamos las librerías
library(tidyverse)
library(caret)


datos <- Datos_Modelo

# Convertimos variable categórica a factor
summary(datos)
datos$Tamaño_Empresa <- as.factor(datos$Tamaño_Empresa)

# Separamos los datos
set.seed(123)
trainIndex <- createDataPartition(datos$Salario, p = 0.8, list = FALSE)
trainData <- datos[trainIndex, ]
testData <- datos[-trainIndex, ]

# Entrenamos los modelos con método correspondiente
modelo_rl <- train(Salario ~ ., data = trainData, method = "lm")       # Regresión Lineal
modelo_dt <- train(Salario ~ ., data = trainData, method = "rpart")    # Árbol de decisión

modelo_rf <- train(Salario ~ ., data = trainData, method = "rf")       # Random Forest
modelo_knn <- train(Salario ~ ., data = trainData, method = "knn")     # K-Nearest Neighbors

# Guardamos todos los modelos entrenados en una lista
modelos <- list(
  "Regresión Lineal" = modelo_rl,
  "Árbol de Decisión" = modelo_dt,
  "Random Forest" = modelo_rf,
  "KNN" = modelo_knn
)

# Creamos un data.frame vacío para guardar los resultados
resultados <- data.frame(Modelo = character(), R2 = double(), MAE = double(), RMSE = double())

# Evaluamos los modelos ya entrenados con los datos de testeo
for (nombre in names(modelos)) {
  
  modelo <- modelos[[nombre]]  # extrae el modelo desde la lista
  
  pred <- predict(modelo, newdata = testData)
  r2 <- R2(pred, testData$Salario)
  mae <- MAE(pred, testData$Salario)
  rmse <- RMSE(pred, testData$Salario)
  
  resultados <- rbind(resultados, data.frame(Modelo = nombre, R2 = r2, MAE = mae, RMSE = rmse))
  
}

# Mostramos los resultados
print(resultados)



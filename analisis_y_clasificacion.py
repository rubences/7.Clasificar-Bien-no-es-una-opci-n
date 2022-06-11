#-----------------------------------------------------------------------------------------
#
# Módulos necesarios:
#   PANDAS 0.24.2
#   NUMPY 1.16.3
#   MATPLOTLIB 3.0.3
#   SCIKIT-LEARN : 0.21.0
#   JMPEstadisticas (copiar el archivo en su proyecto al mismo nivel que este archivo)
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------

#Adquisición de los datos
import pandas as pnd
observaciones = pnd.read_csv("datas/sonar.all-data.csv")

#Informaciones
print(observaciones.columns.values)

#Cantidad de observaciones
print (observaciones.shape)

#Añadir un nombre para cada característica
observaciones = pnd.read_csv("datas/sonar.all-data.csv", names=["F1","F2","F3","F4","F5","F6","F7","F8","F9",
                                                      "F10","F11","F12","F13","F14","F15","F16","F17","F18","F19",
                                                      "F20","F21","F22","F23","F24","F25","F26","F27","F28","F29",
                                                      "F30","F31","F32","F33","F34","F35","F36","F37","F38","F39",
                                                      "F40","F41","F42","F43","F44","F45","F46","F47","F48","F49",
                                                      "F50","F51","F52","F53","F54","F55","F56","F57","F58","F59",
                                                      "F60","OBJETO"])

#Desactivación de la cantidad máxima de columnas del DataFrame a mostrar
pnd.set_option('display.max_columns',None)

#Visualización de las 10 primeras observaciones
print(observaciones.head(10))

#Si el objeto es una mina tomará el valor 1, sino el
#valor 0
observaciones['OBJETO'] = (observaciones['OBJETO']=='M').astype(int)

#¿Faltan datos?
print(observaciones.info())

#Reparto entre mina y roca
print(observaciones.groupby("OBJETO").size())


#Visualización de los distintos indicadores
print (observaciones.describe())

#Búsqueda de los valores extremos con la ayuda de la biblioteca JMPEstadisticas
import JMPEstadisticas as jmp
stats = jmp.JMPEstadisticas(observaciones['F1'])
stats.analisisCaracteristica()


#Uso del módulo Scikit-Learn
from sklearn.model_selection import train_test_split
array = observaciones.values

#Conversión de los datos en tipo decimal
X = array[:,0:-1].astype(float)

#Se elige la última columna como característica de predicción
Y = array[:,-1]

#Creación de los conjuntos de aprendizaje y de pruebas
porcentaje_datos_test = 0.2
X_APRENDIZAJE, X_VALIDACION, Y_APRENDIZAJE, Y_VALIDACION = train_test_split(X, Y, test_size=porcentaje_datos_test, random_state=42)

#Importación de los algoritmos y de la función de cálculo de precisión
#accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Eliminación de los errores de tipo warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#REGRESION LOGISTICA
regresion_logistica = LogisticRegression()
regresion_logistica.fit(X_APRENDIZAJE, Y_APRENDIZAJE)
predicciones = regresion_logistica.predict(X_VALIDACION)
print("Regresión logística: "+str(accuracy_score(predicciones, Y_VALIDACION)))

#ARBOL DE DECISION
arbol_decision = DecisionTreeClassifier()
arbol_decision.fit(X_APRENDIZAJE, Y_APRENDIZAJE)
predicciones = arbol_decision.predict(X_VALIDACION)
print("Arbol de decisión:  "+str(accuracy_score(predicciones, Y_VALIDACION)))


#BOSQUES ALEATORIOS
bosque_aleatorio= RandomForestClassifier()
bosque_aleatorio.fit(X_APRENDIZAJE, Y_APRENDIZAJE)
predicciones = bosque_aleatorio.predict(X_VALIDACION)
print("Bosque aleatorio: "+str(accuracy_score(predicciones, Y_VALIDACION)))


#K VECINOS MÁS CERCANOS
knn = KNeighborsClassifier()
knn.fit(X_APRENDIZAJE, Y_APRENDIZAJE)
predicciones = knn.predict(X_VALIDACION)
print("K vecinos más cercanos: "+str(accuracy_score(predicciones, Y_VALIDACION)))


#MAQUINA DE VECTORES DE SOPORTE
SVM = SVC(gamma='auto')
SVM.fit(X_APRENDIZAJE, Y_APRENDIZAJE)
predicciones = SVM.predict(X_VALIDACION)
print("Máquina de vectores de soporte: "+str(accuracy_score(predicciones, Y_VALIDACION)))


from sklearn.model_selection import GridSearchCV

#Définición de un rango de valores a probar
penalizacion = [{'C': range(1,100)}]


#Pruebas con 5 muestras de Validación Cruzada
busqueda_optimizaciones = GridSearchCV(SVC(), penalizacion, cv=5)
busqueda_optimizaciones.fit(X_APRENDIZAJE, Y_APRENDIZAJE)

print("El mejor parámetro es:")
print()
print(busqueda_optimizaciones.best_params_)
print()


#MAQUINA DE VECTORES DE SOPORTE OPTIMIZADA
SVM = SVC(C=65, gamma='auto')
SVM.fit(X_APRENDIZAJE, Y_APRENDIZAJE)
predicciones = SVM.predict(X_VALIDACION)
print("Máquina de vectores de soporte optimizada: "+str(accuracy_score(predicciones, Y_VALIDACION)))


#Importación de los algoritmos de "boosting"
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier


#GRADIENT BOOSTING
gradientBoosting = GradientBoostingClassifier()
gradientBoosting.fit(X_APRENDIZAJE, Y_APRENDIZAJE)
predicciones = SVM.predict(X_VALIDACION)
print("GRADIENT BOOSTING: "+str(accuracy_score(predicciones, Y_VALIDACION)))



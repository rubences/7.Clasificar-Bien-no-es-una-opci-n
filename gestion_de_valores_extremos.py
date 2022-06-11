#-----------------------------------------------------------------------------------------
#
# Módulos necesarios:
#   PANDAS 0.24.2
#   NUMPY 1.16.3
#   MATPLOTLIB 3.0.3
#   SCIKIT-LEARN: 0.21.0
#   JMPEstadísticas (copiar el archivo en su proyecto al mismo nivel que este archivo)
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


import pandas as pnd
observaciones = pnd.read_csv("datas/sonar.all-data.csv", names=["F1","F2","F3","F4","F5","F6","F7","F8","F9",
                                                      "F10","F11","F12","F13","F14","F15","F16","F17","F18","F19",
                                                      "F20","F21","F22","F23","F24","F25","F26","F27","F28","F29",
                                                      "F30","F31","F32","F33","F34","F35","F36","F37","F38","F39",
                                                      "F40","F41","F42","F43","F44","F45","F46","F47","F48","F49",
                                                      "F50","F51","F52","F53","F54","F55","F56","F57","F58","F59",
                                                      "F60","OBJETO"])

observaciones['OBJETO'] = (observaciones['OBJETO']=='M').astype(int)

#Para cada característica se buscan los números de línea correspondientes
#a un dato extremo
import numpy as np

#Se crea una lista encargada de contener los números de líneas correspondientes
#a un valor extremo
num_lineas = []

#Miramos las 60 características
for caracteristica in observaciones.columns.tolist():
    #Para una caracteristica: cálculo de los percentilee
    Q1 = np.percentile(observaciones[caracteristica],25)
    Q3 = np.percentile(observaciones[caracteristica],75)
    #Cálculo del límite
    dato_extremo = 1.5*(Q3-Q1)
    #Si el dato es inferior o superior al límite se recupera su número de línea y se añade a la lista
    lista_datos_extremos = observaciones[(observaciones[caracteristica]<Q1-dato_extremo) | (observaciones[caracteristica]>Q3+dato_extremo)].index
    num_lineas.extend(lista_datos_extremos)



#Se ordena la lista de menor a mayor
num_lineas.sort()


#Se crea una lista que contiene los números de líneas a eliminar
num_lineas_a_eliminar=[]


#Se mira el conjunto de los números de líneas
for linea in num_lineas :
    #Para una línea, se recupera su número
    num_linea = linea
    #Se calcula la cantidad de veces donde aparece este número de línea
    #en el conjunto de los números de líneas
    n_valores_extremos = num_lineas.count(num_linea)

    #Si la cantidad de errores es superior a 7, entonces se añade el número de la
    #línea a la lista de las líneas a eliminar
    if (n_valores_extremos>7):
        num_lineas_a_eliminar.append(num_linea)



#Se eliminan los duplicados
num_lineas_a_eliminar = list(set(num_lineas_a_eliminar))


#A continuuación se eliminan las líneas en el dataframe
print(num_lineas_a_eliminar)
print("Cantidad de líneas à eliminar = "+str(len(num_lineas_a_eliminar)))
observaciones = observaciones.drop(num_lineas_a_eliminar,axis=0)
print()
print()



#Uso del módulo Scikit-Learn
from sklearn.model_selection import train_test_split
array = observaciones.values

#Conversión de los datos a tipo decimal
X = array[:,0:-1].astype(float)

#Elegimos la última columna como característica de predicción
Y = array[:,-1]

#Creación de los conjuntos de aprendizaje y de pruebas
porcentaje_datos_test = 0.2
X_APRENDIZAJE, X_VALIDACION, Y_APRENDIZAJE, Y_VALIDACION = train_test_split(X, Y, test_size=porcentaje_datos_test, random_state=42)

#Importar algoritmos y la funcón del cálculo de precisión
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

#ÁRBOL DE DECISIÓN
arbol_decision = DecisionTreeClassifier()
arbol_decision.fit(X_APRENDIZAJE, Y_APRENDIZAJE)
predicciones = arbol_decision.predict(X_VALIDACION)
print("Árbol de decisión:  "+str(accuracy_score(predicciones, Y_VALIDACION)))


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

#Definición de un rango de valores a probar
penalizacion = [{'C': range(1,100)}]


#Pruebas con 5 muestras de Validación Cruzada
busqueda_optimizaciones = GridSearchCV(SVC(), penalizacion, cv=5)
busqueda_optimizaciones.fit(X_APRENDIZAJE, Y_APRENDIZAJE)

print("El mejor parámetro es:")
print()
print(busqueda_optimizaciones.best_params_)
print()


#MÁQUINA DE VECTORES DE SOPORTE OPTIMIZADA
SVM = SVC(C=98, gamma='auto')
SVM.fit(X_APRENDIZAJE, Y_APRENDIZAJE)
predicciones = SVM.predict(X_VALIDACION)
print("Máquina de vectores de soporte optimizada: "+str(accuracy_score(predicciones, Y_VALIDACION)))
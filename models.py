# %% [markdown]
# # Parcial 1

# %% [markdown]
# Al entregar la solución de este parcial, yo, Carlos Eduardo Figueredo Triana con código 201813445 me comprometo a no conversar durante el desarrollo de este examen con ninguna persona que no sea el profesor del curso, sobre aspectos relacionados con el parcial; tampoco utilizaré algún medio de comunicación por voz, texto o intercambio de archivos, para consultar o compartir con otros, información sobre el tema del parcial. Soy consciente y acepto las consecuencias que acarreará para mi desempeño académico cometer fraude en este parcial

# %% [markdown]
# ### Objetivo del negocio

# %% [markdown]
# Tecnologías Alpes busca que mediante algoritmos de Machine Learning se pueda saber cuáles son los factores más importantes que generan que un empleado deje de estar en la empresa.
#
# Variable objetivo: Permanece en empresa.
#
# Variables de decisión: Las demás variables que afecten si el empleado permanece en la empresa o no.
#
# Teniendo esto en cuenta, se usará el algoritmo de clasificación árboles de decisión.
#

# %% [markdown]
# ### 1. Perfilamiento y preparación de datos

# %% [markdown]
# Se realiza importación de librerías

# %%
# Librerías para manejo de datos
import pandas as pd

pd.set_option("display.max_columns", 25)  # Número máximo de columnas a mostrar
pd.set_option("display.max_rows", 50)  # Numero máximo de filas a mostar
import numpy as np

np.random.seed(3301)
import pandas as pd

# Para preparar los datos
from sklearn.preprocessing import LabelEncoder

# Para crear el arbol de decisión
from sklearn.tree import DecisionTreeClassifier

# Para usar KNN como clasificador
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Para realizar la separación del conjunto de aprendizaje en entrenamiento y test.
from sklearn.model_selection import train_test_split

# Para evaluar el modelo
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.metrics import ConfusionMatrixDisplay

# Para búsqueda de hiperparámetros
from sklearn.model_selection import GridSearchCV

# Para la validación cruzada
from sklearn.model_selection import KFold

# Librerías para la visualización
import matplotlib.pyplot as plt

# Seaborn
import seaborn as sns
from sklearn import tree

# %% [markdown]
# Luego importamos los datos

# %%
df = pd.read_csv(
    "Datos_EmpleadosAlpes.csv", sep=";", encoding="utf-8", index_col=None
)

# %% [markdown]
# Vemos características de los datos

# %%
df.shape

# %%
df.head()

# %%
df.dtypes

# %%
df.describe()

# %% [markdown]
# Podemos ver que la mayoría se toman como valores numéricos enteros a excepción de Compromiso y Años rol actual, que se toman como flotantes.
# Entonces veamos estas variables en profundidad.

# %%
pd.value_counts(df["Compromiso"])

# %%
pd.value_counts(df["Años_Rol_Actual"])

# %%
df.isnull().sum()

# %% [markdown]
# Procedemos a eliminar los pocos valores nulos que hay y los posibles repetidos

# %%
# Eliminación registros con ausencias
df = df.dropna()
# Eliminación de registros duplicados.
df = df.drop_duplicates()

# %% [markdown]
# Procedemos a eliminar los ids

# %%
df = df.drop(["ID_empleado"], axis=1)

# %%
df.head()

# %% [markdown]
# Revisamos cada variable en busca de anomalías

# %%
pd.value_counts(df["ViajesNegocio"])

# %% [markdown]
# Ninguno no está en el diccionario, pero se interpretará como si fuera nunca

# %%
df["ViajesNegocio"].replace("Ninguno", "Nunca", inplace=True)
df["ViajesNegocio"].value_counts().to_frame()

# %% [markdown]
# Ahora quedó acorde con el diccionario

# %%
pd.value_counts(df["Edad"])

# %% [markdown]
# Ninguna anomalía en edad.

# %%
pd.value_counts(df["PermaneceEnEmpresa"])

# %% [markdown]
# Ninguna anomalía en PermaneceEnEmpresa.

# %%
pd.value_counts(df["Distancia_casa"])

# %% [markdown]
# Ninguna anomalía en DistanciaCasa.

# %%
pd.value_counts(df["Satisfacción_ambiente"])

# %% [markdown]
# Hay un valor fuera de rango (5), entonces se interpretará como que es 4 debido a que es el valor más cercano.

# %%
df["Satisfacción_ambiente"].replace(5, 4, inplace=True)
df["Satisfacción_ambiente"].value_counts().to_frame()

# %%
pd.value_counts(df["Genero"])

# %% [markdown]
# Se entiende que M se refiere a Hombre y F a Mujer, entonces se realiza el reemplazo.

# %%
df["Genero"].replace("M", "Hombre", inplace=True)
df["Genero"].replace("F", "Mujer", inplace=True)
df["Genero"].value_counts().to_frame()

# %%
pd.value_counts(df["Satisfaccion_trabajo"])

# %% [markdown]
# Satisfaccion_trabajo está correcto.

# %%
pd.value_counts(df["Ingreso_mensual"])

# %% [markdown]
# Se entiende que Ingreso_mensual está lleno de valores numéricos, lo cual es correcto

# %%
pd.value_counts(df["Estado_civil"])

# %% [markdown]
# Estado_civil está correcto.

# %%
pd.value_counts(df["SobreTiempo"])

# %% [markdown]
# SobreTiempo está correcto.

# %%
pd.value_counts(df["Horas_Produccion"])

# %% [markdown]
# Puesto que en Horas_Produccion solo hay un valor, se procede a eliminarlo.

# %%
df = df.drop(["Horas_Produccion"], axis=1)

# %%
pd.value_counts(df["Bonos"])

# %% [markdown]
# Bonos está correcto.

# %%
pd.value_counts(df["Años_trabajando"])

# %% [markdown]
# Años_trabajando está correcto

# %%
pd.value_counts(df["Años_Compañia"])

# %% [markdown]
# Años_Compañia está correcto.

# %%
pd.value_counts(df["Años_Actual_Jefe"])

# %% [markdown]
# Años_Actual_Jefe está correcto.

# %% [markdown]
# Ahora procedemos a convertir las variables categóricas en numéricas.

# %%
df.dtypes

# %% [markdown]
# Para ViajesNegocio interpretamos a Nunca como 0, Pocos como 1 y Frecuentes como 2.

# %%
atributo = "ViajesNegocio"


def label_categorias(row):
    if row[atributo] == "Nunca":
        return 0
    elif row[atributo] == "Pocos":
        return 1
    elif row[atributo] == "Frecuentes":
        return 2
    else:
        return None


df[atributo] = df.apply(lambda row: label_categorias(row), axis=1)
pd.value_counts(df["ViajesNegocio"])

# %% [markdown]
# Para PermaneceEnEmpresa interpretaremos el Sí con 1 y el No con 0

# %%
atributo = "PermaneceEnEmpresa"


def label_categorias(row):
    if row[atributo] == "SI":
        return 1
    elif row[atributo] == "NO":
        return 0
    else:
        return None


df[atributo] = df.apply(lambda row: label_categorias(row), axis=1)
pd.value_counts(df["PermaneceEnEmpresa"])

# %% [markdown]
# Para Genero identificaremos Hombre con 0 y Mujer con 1

# %%
atributo = "Genero"


def label_categorias(row):
    if row[atributo] == "Mujer":
        return 1
    elif row[atributo] == "Hombre":
        return 0
    else:
        return None


df[atributo] = df.apply(lambda row: label_categorias(row), axis=1)
pd.value_counts(df["Genero"])

# %% [markdown]
# Para Estado_civil identificaremos Soltero con 0, Casado con 1 y Divorciado con 2

# %%
atributo = "Estado_civil"


def label_categorias(row):
    if row[atributo] == "Soltero":
        return 0
    elif row[atributo] == "Casado":
        return 1
    elif row[atributo] == "Divorciado":
        return 2
    else:
        return None


df[atributo] = df.apply(lambda row: label_categorias(row), axis=1)
pd.value_counts(df["Estado_civil"])

# %% [markdown]
# Para SobreTiempo identificaremos Si con 1 y No con 0

# %%
atributo = "SobreTiempo"


def label_categorias(row):
    if row[atributo] == "Si":
        return 1
    elif row[atributo] == "No":
        return 0
    else:
        return None


df[atributo] = df.apply(lambda row: label_categorias(row), axis=1)
pd.value_counts(df["SobreTiempo"])

# %%
df.dtypes

# %%
df.shape

# %%
# Vamos a seleccionar de nuestro conjunto solo los atributos numéricos.
number_cols = df.dtypes[
    (df.dtypes == np.int64) | (df.dtypes == np.float64)
].index
number_cols = df.select_dtypes(include=["int64", "float"]).columns
number_cols

# %%
df = df[number_cols]

# %% [markdown]
# ### 2. Modelo (árboles de decisión)

# %%
# Se selecciona la variable objetivo, en este caso "PermaneceEnEmpresa".
Y = df["PermaneceEnEmpresa"]
# Del conjunto de datos se elimina la variable "PermaneceEnEmpresa"
X = df.drop(["PermaneceEnEmpresa"], axis=1)

# %%
# Dividir los datos en entrenamiento y test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

# %%
# Fijemos el número de particiones. Utilizaremos K = 10.
particiones = KFold(n_splits=10, shuffle=True, random_state=0)

# %%
# Establecemos el espacio de búsqueda para los hiperparámetros que deseamos ajustar.
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [4, 6, 8, 10, 20],
    "min_samples_split": [2, 3, 4, 5],
}

# %%
# Definimos el modelo sin ningún valor de estos hiperparámetros
arbol = DecisionTreeClassifier(random_state=0)

# %%
# Ahora utilizamos GridSearch sobre el grid definido y con 10 particiones en la validación cruzada.
mejor_modelo = GridSearchCV(arbol, param_grid, cv=particiones)
# Ajuste del modelo
mejor_modelo.fit(X_train, Y_train)

# %%
# Podemos ver cuál fue el resultado de la búsqueda (mejores valores de hiperparámetros)
mejor_modelo.best_params_

# %%
# Obtener el mejor modelo.
arbol_final = mejor_modelo.best_estimator_
# Probemos ahora este modelo sobre test.
y_pred_train = arbol_final.predict(X_train)
y_pred_test = arbol_final.predict(X_test)
print(
    "Exactitud sobre entrenamiento: %.2f"
    % accuracy_score(Y_train, y_pred_train)
)
print("Exactitud sobre test: %.2f" % accuracy_score(Y_test, y_pred_test))

# %% [markdown]
# ### 3. Evaluación

# %%
# Mostrar reporte de clasificación
print(classification_report(Y_test, y_pred_test))
# Se puede visualizar la matriz de confusión
ConfusionMatrixDisplay.from_estimator(arbol_final, X_test, Y_test)
plt.show()

# %%
# Se imprime el informe de rendimiento
print("Datos de prueba")
print(classification_report(Y_train, y_pred_train))
# Se puede visualizar la matriz de confusión
ConfusionMatrixDisplay.from_estimator(arbol_final, X_train, Y_train)
plt.show()

# %% [markdown]
# Podemos ver que tanto en conjunto entrenamiento como en test el modelo dio un desempeño alto.

# %% [markdown]
# ### 4. Interpretación del modelo

# %%
# Obtener la importancia de las variables. Mientras mayor el coeficiente, más la importancia.
importancia_atributo = pd.DataFrame(
    data={
        "Atributo": X_train.columns,
        "Importancia": arbol_final.feature_importances_,
    }
)
importancia_atributo = importancia_atributo.sort_values(
    by="Importancia", ascending=False
).reset_index(drop=True)
importancia_atributo

# %% [markdown]
# Como podemos ver, las variables más importantes que afectan si el empleado permanece o no en la empresa son "Ingreso mensual", "SobreTiempo" y "Años en la compañía". Con base en esto, la compañía puede tratar estas variables para poder disminuir la rotación de empleados.

# %%
# Gráfico de los primeros 3 niveles del árbol
fig = plt.figure(figsize=(25, 10))
_ = tree.plot_tree(
    arbol_final,
    max_depth=3,
    feature_names=X.columns,
    class_names=["0", "1", "2"],
    filled=True,
    fontsize=9,
)

# %% [markdown]
# Ahora podemos probar el modelo con un valor del conjunto:

# %%
# Se calcula la probabilidad de que un dato cualquiera sea puesto en cada categoría
print(arbol_final.predict_proba(X_test.iloc[[13]]))

# %% [markdown]
# En este caso, el valor 13 tiene una probabilidad del 2% de dejar la empresa.

# %% [markdown]
# ### 5. Valor para el negocio

# %% [markdown]
# Concluimos que el modelo sirve para poder predecir si un empleado va a dejar la compañía o no, esto permitiría a la compañía tomar decisiones respecto al empleado y las variables más importantes que afectan si se queda o no en la compañía.

# %% [markdown]
# ## Parcial 1 - parte 2

# %% [markdown]
# La empresa no ha decidido usar la tarea que se propuso previamente, entonces se hará uso de una nueva tarea con el algoritmo KNN.

# %% [markdown]
# ### Construcción del modelo con KNN:

# %% [markdown]
# Para usar el algoritmo KNN tenemos que establecer sus hiperparámetros, para esto establecemos el espacio de búsqueda y usaremos GridSearchCV para iterar en los valores y hallar los mejores hiperparámetros.

# %%
# Primero definamos el espacio de búsqueda
n_vecinos = list(range(1, 15))

# %%
param_grid = {"n_neighbors": n_vecinos, "p": [1, 2, 3, 4]}

# %%
clasificadorKNN = KNeighborsClassifier()
modelo_Knn = GridSearchCV(clasificadorKNN, param_grid, cv=particiones)
modelo_Knn.fit(X_train, Y_train)
print("Mejor parámetro: {}".format(modelo_Knn.best_params_))
print("Mejor cross-validation score: {:.2f}".format(modelo_Knn.best_score_))

# %% [markdown]
# Los mejores hiperparámetros que nos arrojó fueron n_neighbors igual a 7, p igual a 1 y cross-validation score igual a 0.63

# %% [markdown]
# ### Evaluación

# %%
# Obtener el mejor modelo.
modelo_final = modelo_Knn.best_estimator_
# Probemos ahora este modelo sobre test.
y_pred_train = modelo_final.predict(X_train)
y_pred_test = modelo_final.predict(X_test)
print(
    "Exactitud sobre entrenamiento: %.2f"
    % accuracy_score(Y_train, y_pred_train)
)
print("Exactitud sobre test: %.2f" % accuracy_score(Y_test, y_pred_test))

# %%
# Mostrar reporte de clasificación sobre entrenamiento
print(classification_report(Y_train, y_pred_train))
# Se puede visualizar la matriz de confusión
ConfusionMatrixDisplay.from_estimator(modelo_final, X_train, Y_train)
plt.show()

# %%
# Mostrar reporte de clasificación sobre test
print(classification_report(Y_test, y_pred_test))
# Se puede visualizar la matriz de confusión
ConfusionMatrixDisplay.from_estimator(modelo_final, X_test, Y_test)
plt.show()

# %% [markdown]
# En el caso de uso del modelo con algoritmo KNN obtuvimos un desempeño en conjunto de entrenamiento del 72% y en el conjunto test del 54%, pero resulta que el algoritmo de árboles de decisión obtuvo un desempeño en conjunto de entrenamiento del 79% y en el conjunto test del 72%.

# %% [markdown]
# ### Interpretación del modelo:

# %%
# Se calcula la probabilidad de que un dato cualquiera sea puesto en cada categoría
print(modelo_final.predict_proba(X_test.iloc[[13]]))


class Models:
    tree = arbol_final
    knn = modelo_final


# %% [markdown]
# Usando el algoritmo KNN para el registro 13 nos muestra una probabilidad del 42% de que el empleado no permanezca en la empresa, mientras que cuando usamos el algoritmo árboles de decisión con el mismo registro, este nos dio un valor del 2% de que el empleado no permanezca en la empresa.

# %% [markdown]
# ### Valor para el negocio:

# %% [markdown]
# Viendo los resultados del uso de ambos algoritmos como tareas de aprendizaje, se le recomendaría a la empresa el aceptar usar el algoritmo de árboles de decisión sobre el de KNN. No solo el desempeño es más alto en árboles de decisión sino que también muestra las variables más importantes que afectan si el empleado permanece en la empresa o no.

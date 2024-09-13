import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_cleaning import *
from sklearn.utils import shuffle

# Random_state de la aleatorización de los datos
shuffle_random_state = 127
split_random_state = 88

# Leer datos
df = pd.read_csv("train1.csv")

# Aleatorizar datos para evitar el desbalanceo de clases
df_shuffled = shuffle(df, random_state=shuffle_random_state)

print("df_shuffled random_state: ", shuffle_random_state)

# Separar train y test
x_train, x_test, y_train, y_test = train_test_split(
    df_shuffled.drop(columns=["Name", "Transported"]),
    df_shuffled["Transported"],
    test_size=0.2,                    # El test será el 20% del dataset de entrenamiento
    random_state=split_random_state
)

print("split random_state: ", split_random_state)

# Separar datos numéricos y categóricos
df_cat, df_num = num_cat_separation(x_train)

# Codifica los datos categóricos
df_cat = encode_dataframe(df_cat)

# Crear los diccionarios de conteo de clases y valores faltantes
class_counts_num, missing_values_num = storeMS(df_num)
class_counts_cat, missing_values_cat = storeMS(df_cat)

# Imputación de datos
imputation(df_num, class_counts_num, missing_values_num)
imputation(df_cat, class_counts_cat, missing_values_cat)

# Combina los DataFrames de datos numéricos y categóricos
df_new = combine_num_cat(df_cat, df_num)

# Separar datos numéricos y categóricos para el conjunto de prueba
df_cat_test, df_num_test = num_cat_separation(x_test)

# Codifica los datos categóricos
df_cat_test = encode_dataframe(df_cat_test)

# Crear los diccionarios de conteo de clases y valores faltantes
class_counts_num_test, missing_values_num_test = storeMS(df_num_test)
class_counts_cat_test, missing_values_cat_test = storeMS(df_cat_test)

# Imputación de datos
imputation(df_num_test, class_counts_num_test, missing_values_num_test)
imputation(df_cat_test, class_counts_cat_test, missing_values_cat_test)

df_new_test = combine_num_cat(df_cat_test, df_num_test)

# Definir el modelo
model = CatBoostClassifier(verbose=0)

# Definir los hiperparámetros a buscar
param_dist = {
    'iterations': np.arange(1150, 1500, 5),         # Iteraciones
    'learning_rate': np.arange(0.01, 0.2, 0.05),   # Tasa de aprendizaje
    'depth': np.arange(3, 9, 1),                    # Profundidad del árbol
    'l2_leaf_reg': np.arange(4, 9, 1),              # Regularización L2
}

# Configurar RandomizedSearch
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,  # Número de combinaciones a probar
    scoring='accuracy',  # Métrica
    cv=3,  # Validación cruzada
    random_state=42,
    verbose=1,  # Muestra el progreso
    n_jobs=-1   # Usa todos los núcleos disponibles
)

# Ejecutar la búsqueda aleatoria
random_search.fit(df_new, y_train)

# Mostrar los mejores hiperparámetros
print("Best parameters found: ", random_search.best_params_)
print("Best accuracy: ", random_search.best_score_)


print("\n*************** TRAIN *********************")

# Realizar predicciones
y_pred_train = random_search.predict(df_new)

# Métricas
accuracy = accuracy_score(y_train, y_pred_train)
precision = precision_score(y_train, y_pred_train)
recall = recall_score(y_train, y_pred_train)
f1 = f1_score(y_train, y_pred_train)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'F1: {f1 * 100:.2f}%')

# TEST

print("\n*************** TEST *********************")

# Realizar predicciones
y_pred = random_search.predict(df_new_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'F1: {f1 * 100:.2f}%')


#joblib.dump(random_search, 'catboost_model_V1.joblib')

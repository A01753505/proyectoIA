# 1. Importar bibliotecas -------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split   #type: ignore
from sklearn.metrics import  make_scorer, accuracy_score, recall_score, precision_score, f1_score    #type: ignore
from sklearn.linear_model import LogisticRegressor   #type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor   #type: ignore
from sklearn.svm import SVR #type: ignore
from sklearn.neighbors import KNeighborsRegressor   #type: ignore
from sklearn.tree import DecisionTreeRegressor  #type: ignore
from xgboost import XGBRegressor    #type: ignore
import matplotlib.pyplot as plt #type: ignore
import scikit_posthocs as sp    #type: ignore
import baycomp  #type: ignore
from tqdm import tqdm   #type: ignore


# 2. Cargar el dataset y preparar datos -------------------------------------------------------------------------------------------------------------
df = pd.read_csv("Modelo/train.csv")
y = df["Transported"]
X = pd.read_csv("newDF.csv")

# 3. Definición, entrenamiento y evaluación de regresores --------------------------------------------------------------------------------------------
# Definir los regresores
regressors = {
    #'Random Forest': RandomForestRegressor(random_state=42),
    'Logistic Regression': LogisticRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'KNN Regressor': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(random_state=42)
}

# Definir las métricas para la validación cruzada
scoring = {
    'Accuracy': make_scorer(accuracy_score),
    'Recall':  make_scorer(recall_score, average='macro'),
    'Precision': make_scorer(precision_score, average='macro'),
    'F1': make_scorer(f1_score, average='macro')
}

# Evaluar los regresores utilizando 5-fold cross-validation y guardar los resultados por métrica
results:dict = {}
for name, model in tqdm(regressors.items(), desc="Evaluando Modelos"):
    results[name] = {}
    for metric_name, metric in tqdm(scoring.items(), desc=f"Evaluando métricas para {name}", leave=False):
        scores = cross_val_score(model, X, y, cv=5, scoring=metric)
        results[name][metric_name] = scores  # Guardar los 5 resultados individuales

# Convertir los resultados a un DataFrame para cada métrica
acc_df = pd.DataFrame({model: results[model]['Accuracy'] for model in regressors.keys()}).abs()  # Convertir a valores positivos
rec_df = pd.DataFrame({model: results[model]['Recall'] for model in regressors.keys()}).abs()  # Convertir a valores positivos
pres_df = pd.DataFrame({model: results[model]['Precision'] for model in regressors.keys()})  # Mantener R² sin cambios
f1_df = pd.DataFrame({model: results[model]['F1'] for model in regressors.keys()}).abs()  # Convertir a valores positivos

# Imprimir cada DataFrame por métrica

# Resultados para Accuracy
print("DataFrame de Accuracy (10 resultados por modelo):")
print(acc_df)
print("\n")

# Resultados para Recall
print("DataFrame de Recall (10 resultados por modelo):")
print(rec_df)
print("\n")

# Resultados para Precision
print("DataFrame de Precision (10 resultados por modelo):")
print(pres_df)
print("\n")

# Resultados para F1
print("DataFrame de F1 (10 resultados por modelo):")
print(f1_df)
print("\n")

# 4. Análisis y visualización de métricas individuales ------------------------------------------------------------------------------------------------------------
# Calcular promedios
acc_mean = acc_df.mean()
rec_mean = rec_df.mean()
pres_mean = pres_df.mean()
f1_mean = f1_df.mean()

# Visualizar los resultados
acc_mean.plot(kind='bar', title='Accuracy Promedio por Modelo')
plt.ylabel('Accuracy')
plt.show()

rec_mean.plot(kind='bar', title='Recall Promedio por Modelo')
plt.ylabel('Recall')
plt.show()

pres_mean.plot(kind='bar', title='Precision Promedio por Modelo')
plt.ylabel('Precision')
plt.show()

f1_mean.plot(kind='bar', title='F1 score Promedio por Modelo')
plt.ylabel('F1 score')
plt.show()

# 5. Análisis de diferencias críticas -------------------------------------------------------------------------------------------------------------
# Function to calculate and visualize critical differences using scikit-posthocs
def calcular_diferencias_criticas(df, metric_name):
    try:
        # Calculate ranks of the models
        avg_rank = df.rank(axis=1).mean(axis=0)
        print(avg_rank)

        # Perform the Nemenyi post-hoc test
        cd_result = sp.posthoc_nemenyi_friedman(df.values)

        # Plot the critical difference diagram
        plt.figure(figsize=(10, 6), dpi=100)
        plt.title(f'Diagrama de Diferencias Críticas ({metric_name})')
        sp.critical_difference_diagram(avg_rank, cd_result)
        plt.show()

    except ValueError as e:
        print(f"Error al calcular diferencias críticas para {metric_name}: {e}")

# Calculate and plot critical differences for each metric
calcular_diferencias_criticas(acc_df, 'Accuracy')
calcular_diferencias_criticas(rec_df, 'Recall')
calcular_diferencias_criticas(pres_df, 'Precision')
calcular_diferencias_criticas(f1_df, 'F1 score')
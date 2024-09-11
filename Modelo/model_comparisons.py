# 1. Importar bibliotecas -------------------------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split   #type: ignore
from sklearn.metrics import  make_scorer, accuracy_score, recall_score, precision_score, f1_score    #type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier   #type: ignore
from sklearn.svm import SVC #type: ignore
from sklearn.neighbors import KNeighborsClassifier   #type: ignore
from sklearn.tree import DecisionTreeClassifier  #type: ignore
from xgboost import XGBClassifier    #type: ignore
import matplotlib.pyplot as plt #type: ignore
import scikit_posthocs as sp    #type: ignore
from catboost import CatBoostClassifier  #type: ignore
import baycomp  #type: ignore
from tqdm import tqdm   #type: ignore


# 2. Cargar el dataset y preparar datos -------------------------------------------------------------------------------------------------------------
df = pd.read_csv("Modelo/train.csv")
X = pd.read_csv("newDF.csv")
# Separar train y test
x_train, x_test, y_train, y_test = train_test_split(
    df.drop(columns = ["Name", "Transported"]),
    df["Transported"],
    test_size = 0.2,                    # El test será el 20% del dataset de entrenamiento
    random_state = 42
)

# 3. Definición, entrenamiento y evaluación de clasificadores --------------------------------------------------------------------------------------------
# Definir los clasificadores
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Classifier': SVC(),
    'KNN Classifier': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42)
}

# Definir las métricas para la validación cruzada
scoring = {
    'Accuracy': make_scorer(accuracy_score),
    'Recall':  make_scorer(recall_score, average='macro'),
    'Precision': make_scorer(precision_score, average='macro'),
    'F1': make_scorer(f1_score, average='macro')
}

# Evaluar los clasificadores utilizando 5-fold cross-validation y guardar los resultados por métrica
results:dict = {}
for name, model in tqdm(classifiers.items(), desc="Evaluando Modelos"):
    results[name] = {}
    for metric_name, metric in tqdm(scoring.items(), desc=f"Evaluando métricas para {name}", leave=False):
        scores = cross_val_score(model, X, y_train, cv=5, scoring=metric)
        results[name][metric_name] = scores  # Guardar los 5 resultados individuales

# Convertir los resultados a un DataFrame para cada métrica
acc_df = pd.DataFrame({model: results[model]['Accuracy'] for model in classifiers.keys()}).abs()  # Convertir a valores positivos
rec_df = pd.DataFrame({model: results[model]['Recall'] for model in classifiers.keys()}).abs()  # Convertir a valores positivos
pres_df = pd.DataFrame({model: results[model]['Precision'] for model in classifiers.keys()})  # Mantener R² sin cambios
f1_df = pd.DataFrame({model: results[model]['F1'] for model in classifiers.keys()}).abs()  # Convertir a valores positivos

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

# 6. Comparación bayesiana de todos los clasificadores -------------------------------------------------------------------------------------------------------------
# Función para realizar comparaciones bayesianas entre todos los pares de modelos y generar gráficos
def comparaciones_bayesianas(df, nombre_metrica):
    rope = 0.0  # Región de Equivalencia Práctica (ROPE)
    bayes_comparison_results = {}

    # Comparar todos los pares únicos de modelos
    for i, modelo_1 in enumerate(df.columns):
        for j, modelo_2 in enumerate(df.columns):
            if i < j:  # Evitar comparaciones duplicadas y autocomparaciones
                # Realizar la comparación bayesiana entre los dos modelos seleccionados usando two_on_single
                probs, fig = baycomp.two_on_single(
                    df[modelo_1].values,  # Resultados del primer modelo
                    df[modelo_2].values,  # Resultados del segundo modelo
                    rope=rope,
                    plot=True,  # Generar el gráfico
                    names=(modelo_1, modelo_2)  # Nombres para los modelos
                )
                bayes_comparison_results[(modelo_1, modelo_2)] = probs
                # Imprimir los resultados
                print(f"Comparación bayesiana entre {modelo_1} y {modelo_2} en {nombre_metrica}: {probs}")

                # Guardar el gráfico
                fig.savefig(f"Comparacion_Bayesiana_{modelo_1}_vs_{modelo_2}_{nombre_metrica}.png")
                plt.show()

    return bayes_comparison_results

# Realizar comparaciones bayesianas para cada métrica y generar los gráficos
resultados_bayesianos_acc = comparaciones_bayesianas(acc_df, 'Accuracy')
resultados_bayesianos_rec = comparaciones_bayesianas(rec_df, 'Recall')
resultados_bayesianos_pres = comparaciones_bayesianas(pres_df, 'Precision')
resultados_bayesianos_f1 = comparaciones_bayesianas(f1_df, 'F1 score')
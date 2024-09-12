# ENTRENAMIENTO Y BUSQUEDA DE HIPERPARÁMETROS PARA EL MODELO

import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate   #type: ignore
from catboost import CatBoostClassifier  #type: ignore
from sklearn.model_selection import GridSearchCV

# Cargar el dataset y preparar datos
df = pd.read_csv("Modelo/train.csv")
X = pd.read_csv("DF.csv")
# Separar train y test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    df["Transported"],
    test_size = 0.2,        # El conjunto de validación será el 20% del dataset de entrenamiento
    random_state = 42
)

# Definir y entrenar el modelo
cat = CatBoostClassifier(verbose=0)
cat.fit(X_train, y_train)

# Evaluar el modelo con validación cruzada
scoring = ['accuracy', 'precision', 'recall', 'f1']
results = cross_validate(cat, X_train, y_train, cv=5, scoring=scoring, return_train_score=False)

print(f"Accuracy: {results['test_accuracy'].mean()} ± {results['test_accuracy'].std()}")
print(f"Precision: {results['test_precision'].mean()} ± {results['test_precision'].std()}")
print(f"Recall: {results['test_recall'].mean()} ± {results['test_recall'].std()}")
print(f"F1 Score: {results['test_f1'].mean()} ± {results['test_f1'].std()}")

# Búsqueda de hiperparámetros con GridSearch
# Los valores de los hiperparámetros fueron escogidos y probados aleatoriamente
# param_grid = [{'iterations': [1000, 1250, 1500],
#                'learning_rate': [0.01, 0.03],
#                'max_depth': [4, 6, 8],
#                'l2_leaf_reg': [3, 5, 9]}]

# cat2 = CatBoostClassifier(verbose=0)
# grid_search = GridSearchCV(cat2, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X, y_train)

# print("\nGrid mejor estimador: ",grid_search.best_estimator_)
# print("Grid mejores parámetros: ",grid_search.best_params_)

# Modelo mejorado con los hiperparámetros encontrados
cat_new = CatBoostClassifier(iterations=1250, learning_rate=0.03, depth=4, l2_leaf_reg=4, verbose=0)
cat_new.fit(X_train, y_train)

# Evaluar el modelo con validación cruzada
scoring = ['accuracy', 'precision', 'recall', 'f1']
results = cross_validate(cat, X_train, y_train, cv=5, scoring=scoring, return_train_score=False)

print(f"Accuracy: {results['test_accuracy'].mean()} ± {results['test_accuracy'].std()}")
print(f"Precision: {results['test_precision'].mean()} ± {results['test_precision'].std()}")
print(f"Recall: {results['test_recall'].mean()} ± {results['test_recall'].std()}")
print(f"F1 Score: {results['test_f1'].mean()} ± {results['test_f1'].std()}")
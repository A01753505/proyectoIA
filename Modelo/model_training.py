import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score   #type: ignore
from sklearn.ensemble import GradientBoostingClassifier  #type: ignore
from sklearn.model_selection import GridSearchCV

# Cargar el dataset y preparar datos
df = pd.read_csv("Modelo/train.csv")
X = pd.read_csv("newDF.csv")
# Separar train y test
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns = ["Name", "Transported"]),
    df["Transported"],
    test_size = 0.2,                    # El test será el 20% del dataset de entrenamiento
    random_state = 42
)

# Definir y entrenar el modelo
# boost_reg = GradientBoostingClassifier()
# boost_reg.fit(X, y_train)

# boost_scores = cross_val_score(boost_reg, X, y_train, scoring="accuracy", cv=10)
# boost_rmse_scores = np.sqrt(-boost_scores)

# def display_scores(scores):
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())

# display_scores(boost_rmse_scores)


# Búsqueda de hiperparámetros
param_grid = [{'n_estimators': [100, 115, 125],
               'learning_rate': [0.1, 0.01],
               'max_depth': [2, 3, 4, 5]}]

boost_reg2 = GradientBoostingClassifier()
grid_search = GridSearchCV(boost_reg2, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X, y_train)

print()
print("Grid mejores parámetros: ",grid_search.best_params_)
print("Grid mejor estimador: ",grid_search.best_estimator_)

# boost_reg_new = GradientBoostingClassifier(max_depth=4, n_estimators=125)
# boost_reg_new.fit(X, y_train)

# scores = cross_val_score(boost_reg_new, X, y_train, scoring="accuracy", cv=10)
# rmse_scores = np.sqrt(-scores)
# print()
# display_scores(rmse_scores)

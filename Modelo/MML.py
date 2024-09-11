from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #type: ignore
from catboost import CatBoostClassifier  #type: ignore
import joblib   #type: ignore
import pandas as pd

# ------------------------------------------------- DATAFRAME DE TRAIN -------------------------------------------------

# Leer datos
df = pd.read_csv("Modelo/train.csv")
X = pd.read_csv("DF.csv")
y = df["Transported"]

cat = CatBoostClassifier(iterations=1250, learning_rate=0.03, depth=6, l2_leaf_reg=9, verbose=0)
cat.fit(X, y)

# Realizar predicciones
predictions = cat.predict(X)

# Evaluar precisión
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'F1: {f1 * 100:.2f}%')

# Exportar modelo
# joblib.dump(cat,'Model.joblib')

# ------------------------------------------------- DATAFRAME DE TEST -------------------------------------------------

# Leer datos
# X = pd.read_csv("DF_test.csv")

# Importar modelo
# modelo = joblib.load('Model.joblib')

# Realizar predicciones
# predictions = modelo.predict(X)

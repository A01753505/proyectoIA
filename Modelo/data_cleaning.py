# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  #type: ignore
from collections import Counter
from sklearn.preprocessing import MinMaxScaler  #type: ignore
from sklearn.impute import KNNImputer #type: ignore
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Separar datos numéricos y categóricos
def num_cat_separation(X):
    df_num = X.select_dtypes(include = ["int64", "float64"])
    df_cat = X.select_dtypes(include = ["object"])

    return df_cat, df_num

# Crear los diccionarios de conteo de clases y valores faltantes
def storeMS(df):
  class_counts = {}
  missing_values = {}

  for col in df.columns:
    class_count = Counter(df[col].dropna())  # Excluir valores faltantes al contar clases
    class_counts[col] = class_count
    missing_values[col] = df[col].isnull().sum()

  return class_counts, missing_values

# Tratamiento de datos CATEGÓRICOS
def cat_KNN_cleaning(df_cat, col):
    # Imputación predictiva o basada en KNN
    df_cat[col] = df_cat[col].fillna("NotApply")

    return df_cat

# Tratamiento de datos NUMÉRICOS (KNN)
def num_KNN_cleaning(df_num, col):
    imputer = KNNImputer(n_neighbors=3)
    df_imputed = imputer.fit_transform(df_num[[col]])
    df_num[col] = df_imputed

    return df_num

# Rellenar datos numéricos y faltantes usando imputación
def fill_data(DF, llave, column):
  if llave == "Moda":
    DF[column] = DF[column].fillna(DF[column].mode()[0])
  elif llave == "NewCat":
    if pd.api.types.is_numeric_dtype(DF[column]):
      DF[column] = DF[column].fillna(0.0)
    else:
      DF[column] = DF[column].fillna(False)
  else:
    if pd.api.types.is_numeric_dtype(DF[column]):
      num_KNN_cleaning(DF, column)
    else:
      cat_KNN_cleaning(DF, column)

#Hacer la imputación de cada columna dependiendo la sugerencia
def imputation(df, class_counts, missing_values):
  for col, counts in class_counts.items():
      if missing_values[col] != 0:
        most_common_class, most_common_count = counts.most_common(1)[0]
        total_non_null = sum(counts.values())

        if most_common_count / total_non_null < 0.5:
            fill_data(df, "Moda", col)
        elif missing_values[col] / len(df) < 0.05:
            fill_data(df, "NewCat", col)
        else:
            fill_data(df, "KNN", col)

# Realiza label encoding
def label_encoding(df, col):
  le = LabelEncoder()
  df[col] = le.fit_transform(df[col])
  return df

# Realiza one-hot encoding
def one_hot_encoding(df, cols):
  one_hot = pd.get_dummies(df, columns=cols)
  return one_hot

# Realiza una codificación dependiendo del tipo de columna
def encode_dataframe(df):
  for col in df:
    if col == "PassengerId":
      df['group'] = df['PassengerId'].str.split('_').str[0]
      group_counts = df['group'].value_counts()
      df['Has_family'] = df['group'].map(lambda x: True if group_counts[x] > 1 else False)
      df.drop(columns=['PassengerId', 'group'], inplace=True)
    elif col == "Cabin":
      df[['Deck', 'Side']] = df['Cabin'].str.split('/', expand=True).iloc[:, [0, 2]]
      df = label_encoding(df, 'Deck')
      df = pd.get_dummies(df, columns=['Side'])
      df.drop(columns=['Cabin'], inplace=True)
    elif col == "HomePlanet" or col == "Destination":
      df = one_hot_encoding(df, [col])
    else:
      pass
  return df

# Combina los DataFrames de datos numéricos y categóricos
def combine_num_cat(df_cat, df_num):
    # Aseguramos que los índices coinciden
    if not df_cat.index.equals(df_num.index):
        df_cat = df_cat.reset_index(drop=True)
        df_num = df_num.reset_index(drop=True)

    # Combina los DataFrames por columnas
    df_combined = pd.concat([df_cat, df_num], axis=1)

    return df_combined


#----------------------------------------------------------------------DATAFRAME DE PRUEBA----------------------------------------------------------------------

# Leer datos
df = pd.read_csv("Modelo/train.csv")

# Separar train y test
x_train, x_test, y_train, y_test = train_test_split(
    df.drop(columns = ["Name", "Transported"]),
    df["Transported"],
    test_size = 0.2,                    # El test será el 20% del dataset de entrenamiento
    random_state = 42
)

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

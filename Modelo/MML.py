from catboost import CatBoostClassifier  #type: ignore
import joblib   #type: ignore
import pandas as pd

# ------------------------------------------------- DATAFRAME DE TRAIN -------------------------------------------------

# Leer datos
df = pd.read_csv("Modelo/train.csv")
X = pd.read_csv("newDF.csv")
y = df["Transported"]

cat = CatBoostClassifier(iterations= 1500, l2_leaf_reg= 3, learning_rate= 0.01, max_depth= 4, verbose= 0)
cat.fit(X, y)

# Exportar modelo
# joblib.dump(cat,'pruebaModelo.joblib')

# Predicciones
# predictions = cat.predict(df)

# Calculo del error
# error_scores = cross_val_score(cat, df, predictions, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
# print(rmse_scores)


# ------------------------------------------------- DATAFRAME DE TEST -------------------------------------------------

# Leer datos
# df_test = pd.read_csv("test.csv")

# X_test = df_test.drop(columns = ["Id"])

# # Separar datos numéricos y categóricos
# df_catt, df_numt = num_cat_separation(X_test)

# df_numt = outliers(df_numt, factor=1.5)
# df_numt = normalize_data(df_numt)

# class_counts_numt, missing_values_numt = storeMS(df_numt)
# class_counts_catt, missing_values_catt = storeMS(df_catt)

# imputation(df_numt, class_counts_numt, missing_values_numt)
# imputation(df_catt, class_counts_catt, missing_values_catt)

# df_catt = encode_dataframe(df_catt)

# df_test = combine_num_cat(df_catt, df_numt)

# def columnas_faltantes(df1, df2):
#     columnas_df1 = set(df1.columns)
#     columnas_df2 = set(df2.columns)

#     # Encuentra columnas en df1 que no están en df2
#     columnas_faltantes = columnas_df1 - columnas_df2

#     return list(columnas_faltantes)

# Ejemplo de uso
# faltantes = columnas_faltantes(df_new, df_test)
# print("Columnas en df_new que faltan en df_test:", faltantes)

# def agregar_columnas_faltantes(df1, df2, valor_default=0):
#     columnas_df1 = set(df1.columns)
#     columnas_df2 = set(df2.columns)

#     # Encuentra columnas en df1 que no están en df2
#     columnas_faltantes = columnas_df1 - columnas_df2

#     # Agrega las columnas faltantes a df2 con el valor predeterminado
#     for columna in columnas_faltantes:
#         df2[columna] = valor_default

#     return df2

# # Reordenar las columnas del DataFrame objetivo según el orden del DataFrame de referencia
# def ordenar_columnas(df1, df2):
#     columnas_referencia = df2.columns.tolist()

#     columnas_target = [col for col in columnas_referencia if col in df1.columns]

#     # Asegurarse de que todas las columnas de df1 estén presentes en df2
#     if len(columnas_target) != len(df1.columns):
#         raise ValueError("El DataFrame objetivo tiene columnas que no están en el DataFrame de referencia o viceversa.")

#     df1_ordenado = df1[columnas_target]

#     return df1_ordenado

# df_test_actualizado = agregar_columnas_faltantes(df_new, df_test)
# df_new_actualizado = agregar_columnas_faltantes(df_test, df_new)

# df_test_order = ordenar_columnas(df_test_actualizado, df_new_actualizado)

# boost_reg = GradientBoostingRegressor(max_depth=4, n_estimators=125)
# boost_reg.fit(df_new_actualizado, y)

# # Predicciones
# predictions = boost_reg.predict(df_test_order)

# # Calculo del error
# error_scores = cross_val_score(boost_reg, df_test, predictions,
#                                 scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
# display_scores(rmse_scores)
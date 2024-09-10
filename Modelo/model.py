# #----------------------------------------------------------------------PRUEBAS----------------------------------------------------------------------

# # Regresión lineal

# from sklearn.linear_model import LinearRegression #type: ignore
# lin_reg = LinearRegression()
# lin_reg.fit(df_new, y_train)

# from sklearn.metrics import mean_squared_error  #type: ignore

# predictions = lin_reg.predict(df_new)
# lin_mse = mean_squared_error(y_train,predictions)
# lin_rmse = np.sqrt(lin_mse)
# lin_rmse

# # Árboles de decisión

# from sklearn.tree import DecisionTreeRegressor  #type: ignore

# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(df_new, y_train)

# predictions = tree_reg.predict(df_new)
# tree_mse = mean_squared_error(y_train, predictions)
# tree_rmse = np.sqrt(tree_mse)
# tree_rmse

# # Validación cruzada

# from sklearn.model_selection import cross_val_score

# def display_scores(scores):
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())

# scores = cross_val_score(lin_reg, df_new, y_train, scoring="neg_mean_squared_error", cv=10)
# linear_rmse_scores = np.sqrt(-scores)

# display_scores(linear_rmse_scores)

# scores = cross_val_score(tree_reg, df_new, y_train, scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)

# display_scores(tree_rmse_scores)

# from sklearn.ensemble import RandomForestRegressor  #type: ignore

# forest_reg = RandomForestRegressor()
# forest_reg.fit(df_new, y_train)

# forest_scores = cross_val_score(forest_reg, df_new, y_train,
#                                 scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)

# from sklearn.svm import SVR #type: ignore

# svr_reg = SVR()
# svr_reg.fit(df_new, y_train)

# svr_scores = cross_val_score(svr_reg, df_new, y_train,
#                                 scoring="neg_mean_squared_error", cv=10)
# svr_rmse_scores = np.sqrt(-svr_scores)
# display_scores(svr_rmse_scores)

# from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

# boost_reg = GradientBoostingRegressor()
# boost_reg.fit(df_new, y_train)

# boost_scores = cross_val_score(boost_reg, df_new, y_train,
#                                 scoring="neg_mean_squared_error", cv=10)
# boost_rmse_scores = np.sqrt(-boost_scores)
# display_scores(boost_rmse_scores)

# extra_reg = ExtraTreesRegressor()
# extra_reg.fit(df_new, y_train)

# extra_scores = cross_val_score(extra_reg, df_new, y_train,
#                                 scoring="neg_mean_squared_error", cv=10)
# extra_rmse_scores = np.sqrt(-extra_scores)
# display_scores(extra_rmse_scores)

# """# Búsqueda de hiperparámetros"""

# from sklearn.model_selection import GridSearchCV

# #Búsqueda de parámetros
# param_grid = [{'n_estimators': [100, 115, 125], 'learning_rate': [0.1, 0.01], 'max_depth': [2, 3, 4, 5]}]

# boost_reg = GradientBoostingRegressor()
# grid_search = GridSearchCV(boost_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

# grid_search.fit(df_new, y_train)

# print("Grid mejores parámetros: ",grid_search.best_params_)
# print("Grid mejor estimador: ",grid_search.best_estimator_)

# boost_reg = GradientBoostingRegressor(max_depth=4, n_estimators=125)
# boost_reg.fit(df_new, y_train)

# scores = cross_val_score(boost_reg, df_new, y_train,
#                                 scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
# display_scores(rmse_scores)

# # cvres = grid_search.cv_results_
# # print("Resultados de Grid search")
# # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
# #   print(np.sqrt(-mean_score), params)

# """# Resultados"""

# # DATAFRAME DE TRAIN

# # Leer datos
# df = pd.read_csv("train.csv")

# X = df.drop(columns = ["Id", "SalePrice"])
# y = df["SalePrice"]

# # Separar datos numéricos y categóricos
# df_cat, df_num = num_cat_separation(X)

# df_num = outliers(df_num, factor=1.5)
# df_num = normalize_data(df_num)

# class_counts_num, missing_values_num = storeMS(df_num)
# class_counts_cat, missing_values_cat = storeMS(df_cat)

# imputation(df_num, class_counts_num, missing_values_num)
# imputation(df_cat, class_counts_cat, missing_values_cat)

# df_cat = encode_dataframe(df_cat)

# df_new = combine_num_cat(df_cat, df_num)

# # DATAFRAME DE TEST

# # Leer datos
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

# # Ejemplo de uso
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
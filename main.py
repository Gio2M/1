from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from typing import Optional
import numpy as np
import sklearn


# Ruta al archivo JSON
file_path = '../PI MLOps - STEAM/steam_games.json'

# Leer el archivo JSON línea por línea y cargar los datos en una lista
data_list = []
with open(file_path, 'r') as f:
    for line in f:
        data_list.append(eval(line.strip()))

# Crear DataFrame a partir de la lista de diccionarios
df = pd.DataFrame(data_list)

df1 = df.copy()

df2 = df.copy()


# Función para obtener juegos por año
def juegos_por_año(df1, año):

    # Convertir el año a entero
    #año = int(año)

    # Filtrar el DataFrame para obtener solo los juegos del año proporcionado
    juegos_del_año = df1[df1['release_date'].str.startswith((año)) & pd.notna(df1['release_date'])]

    # Validar si se encontraron juegos para el año dado
    if juegos_del_año.empty:
        return f"No se encontraron juegos para el año {año}"

    # Obtener la lista de nombres de juegos y ordenarla alfabéticamente
    lista_juegos = sorted(juegos_del_año['app_name'].tolist())

    # Devolver la lista de juegos ordenada alfabéticamente
    return lista_juegos




# Convertir la columna "price" a valores numéricos
df2['price'] = pd.to_numeric(df2['price'], errors='coerce')

# Crear un nuevo DataFrame eliminando las columnas no deseadas
columns_to_drop = ['url', 'discount_price', 'reviews_url', 'id', 'metascore']
df2 = df2.drop(columns=columns_to_drop)

# Eliminar todas las filas con valores faltantes en cualquier columna
df2 = df2.dropna()

# Convertir la columna de fechas a formato datetime
df2['release_date'] = pd.to_datetime(df2['release_date'], format='%Y-%m-%d', errors='coerce')

selected_columns = ['genres', 'specs', 'price', 'early_access', 'sentiment', 'publisher']
df2 = df2[selected_columns].copy()

# Crear columnas nuevas en el DataFrame para cada categoría única de sentiment
unique_sentiments = ['Mixed', 'Very Positive', 'Positive', 'Mostly Positive', '1 user reviews', '2 user reviews', '3 user reviews', '4 user reviews', '5 user reviews', 'Mostly Negative']

for sentiment in unique_sentiments:
    df2[sentiment] = df2['sentiment'].apply(lambda x: sentiment in x)

# Eliminar la columna original "sentiment"
df2.drop(columns=['sentiment'], inplace=True)

# Crear una copia del DataFrame original
encoded_df = df2.copy()

# Lista de géneros seleccionados
selected_genres = ['Indie', 'Action', 'Casual', 'Adventure', 'Strategy', 'Simulation', 'RPG', 'Free to Play', 'Early Access', 'Sports']

# Crear columnas para los géneros seleccionados y llenar con valores 0
for genre in selected_genres:
    encoded_df[genre] = encoded_df['genres'].apply(lambda x: 1 if genre in x else 0)

# Eliminar la columna original de géneros
encoded_df.drop(columns=['genres'], inplace=True)
df2 = encoded_df

# Crear una copia del DataFrame original
encoded_df = df2.copy()

# Lista de especificaciones seleccionadas
selected_specs = ['Single-player', 'Steam Achievements', 'Downloadable Content', 'Steam Trading Cards', 'Steam Cloud', 'Multi-player', 'Full controller support', 'Partial Controller Support', 'Steam Leaderboards', 'Co-op', 'Sheared/Split Screen']

# Crear columnas para las especificaciones seleccionadas y llenar con valores 0
for spec in selected_specs:
    encoded_df[spec] = encoded_df['specs'].apply(lambda x: 1 if spec in x else 0)

# Eliminar la columna original de especificaciones
encoded_df.drop(columns=['specs'], inplace=True)
df2 = encoded_df

# Crear una copia del DataFrame original
encoded_df = df2.copy()

# Lista de publishers seleccionados
selected_publishers = ['Ubisoft', 'Dovetali Games – Trains', 'Degica', 'Paradox Interacticve', 'SEGA', 'Dovetail Games - Flight', 'KOEI TECMO GAMES CO., LTD.', 'Activision', 'Big Fish Games', 'KISS ltd']

# Crear columnas para los publishers seleccionados y llenar con valores 0
for publisher in selected_publishers:
    encoded_df[publisher] = encoded_df['publisher'].apply(lambda x: 1 if publisher in x else 0)

# Eliminar la columna original de publishers
encoded_df.drop(columns=['publisher'], inplace=True)
df2 = encoded_df

from sklearn.model_selection import train_test_split

# Dividir los datos en características (X) y variable objetivo (y)
X = df2.drop(columns=['price'])
y = df2['price']




# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

# Crear el modelo de regresión utilizando Random Forest con parámetros específicos
model1 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
model1.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

# Obtener las predicciones del modelo en el conjunto de prueba
predictions = model1.predict(X_test)

# Calcular el RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Mostrar el precio predicho y el RMSE
print("Precio predicho:", predictions)
print("RMSE:", rmse)



app = FastAPI()

# Endpoint para obtener juegos por año
@app.get("/juegos/{Anio}")
def juegos(Anio: str):
    return juegos_por_año(df, Anio)

@app.get("/genero/{Anio}")
def genero( Anio: str ): #Se ingresa un año y devuelve una lista con los 5 géneros más ofrecidos en el orden correspondiente.
    return generos_mas_comunes(df,Anio)

@app.get("/specs/{Anio}")
def specs( Anio: str ): #Se ingresa un año y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente.
    return specs_comun_año(df,Anio)

@app.get("/earlyacces/{Anio}")
def earlyacces( Anio: str ): #Cantidad de juegos lanzados en un año con early access.
    return num_early_año(df,Anio)

@app.get("/sentim/{Anio}")
def sentim( Anio: str ): #Según el año de lanzamiento, se devuelve una lista con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento.
    return num_juegos_sentim(df,Anio)
                    #Ejemplo de retorno: {Mixed = 182, Very Positive = 120, Positive = 278}

@app.get("/metascore/{Anio}")
def metascore( Anio: str ): #Top 5 juegos según año con mayor metascore.
    return mayor_metascore_año(df,Anio)



# Definir la estructura de datos para la entrada de predicción
class PredictionInput(BaseModel):
    Mixed : Optional [bool] 
    Very_Positive : Optional [bool]
    Positive : Optional [bool]
    Mostly_Positive : Optional [bool]
    user1_reviews : Optional [bool]
    user2_reviews : Optional [bool]
    user3_reviews : Optional [bool]
    user4_reviews : Optional [bool]
    user5_reviews : Optional [bool]
    Mostly_Negative : Optional [bool]
    Indie : Optional [bool]
    Action : Optional [bool]
    Casual : Optional [bool]
    Adventure : Optional [bool]
    Strategy : Optional [bool]
    Simulation : Optional [bool]
    RPG : Optional [bool]
    Free_to_Play : Optional [bool]
    Early_Access : Optional [bool]
    Sports: Optional [bool]
    Single_player : Optional [bool]
    Steam_Achievements : Optional [bool]
    Downloadable_Content : Optional [bool]
    Steam_Trading_Cards : Optional [bool]
    Steam_Cloud : Optional [bool]
    Multi_player : Optional [bool]
    Full_controller_support : Optional [bool]
    Partial_Controller_Support : Optional [bool]
    Steam_Leaderboards : Optional [bool]
    Co_op : Optional [bool]
    Sheared_Split_Screen : Optional [bool]
    Ubisoft : Optional [bool]
    Dovetali_Games_Trains: Optional [bool]
    Degica: Optional [bool]
    Paradox_Interacticve: Optional [bool]
    SEGA: Optional [bool]
    Dovetail_Games_Flight: Optional [bool]
    KOEI_TECMO_GAMES_CO_LTD: Optional [bool]
    Activision: Optional [bool]
    Big_Fish_Games: Optional [bool]
    KISS_ltd: Optional [bool]
    Early1_Access: Optional [bool]
    # ... Otras características aquí ...

# Definir una ruta para hacer predicciones
@app.post("/predict")
def predict_price(data: PredictionInput):
    prediction_data = [
    data.Mixed, 
    data.Very_Positive, 
    data.Positive, 
    data.Mostly_Positive,
    data.user1_reviews, 
    data.user2_reviews,
    data.user3_reviews,
    data.user4_reviews,
    data.user5_reviews,
    data.Mostly_Negative,
    data.Indie,
    data.Action,
    data.Casual,
    data.Adventure,
    data.Strategy,
    data.Simulation,
    data.RPG,
    data.Free_to_Play,
    data.Early_Access,
    data.Sports,
    data.Single_player,
    data.Steam_Achievements,
    data.Downloadable_Content,
    data.Steam_Trading_Cards,
    data.Steam_Cloud,
    data.Multi_player,
    data.Full_controller_support,
    data.Partial_Controller_Support,
    data.Steam_Leaderboards,
    data.Co_op,
    data.Sheared_Split_Screen,
    data.Ubisoft,
    data.Dovetali_Games_Trains,
    data.Degica,
    data.Paradox_Interacticve,
    data.SEGA,
    data.Dovetail_Games_Flight,
    data.KOEI_TECMO_GAMES_CO_LTD,
    data.Activision,
    data.Big_Fish_Games,
    data.KISS_ltd,
    data.Early1_Access,

        # ... Otras características aquí ...
    ]
    predictions = model1.predict([prediction_data])
    return {"Precio predicho": predictions[0],"RMSE": rmse}

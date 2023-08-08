## Proyecto de Predicción de Precios de Juegos en Steam
Este repositorio contiene un proyecto que utiliza el framework FastAPI para construir una API que predice los precios de los juegos en Steam en función de varias características. El proyecto también incluye la limpieza y preparación de los datos, la creación y entrenamiento de un modelo de regresión de Random Forest y la implementación de la API para hacer predicciones en tiempo real.

## Descripción
Este proyecto tiene como objetivo crear una API que permita a los usuarios obtener predicciones de precios de juegos en Steam en función de las características seleccionadas. Además, proporciona una ruta para obtener una lista de juegos lanzados en un año específico.

## Estructura del Repositorio
Api_PI_1: Es la carpeta raiz que contiene todos los archivos.
main.py: Contiene el código principal de la API utilizando FastAPI y el del modelado y entrenamiento del modelo de Random Forest.
EDA.ipynb: Notebook en el que encontraras todos los analisis que se hicieron a los datos. 
Transformacion_Modelado.ipynb: Notebook que contiene toda las tranformaciones que fueron necesarias para dejar los datos consumibles por el modelo.
README.md: Este archivo que proporciona información sobre el proyecto.
PI MLOps-STEAM: Carpeta que continen el archivo steam_games.json fuente de los datos de los juegos en Steam y su diccionario de datos.

## Requisitos
Python 3.7 o superior
Bibliotecas: FastAPI, pandas, numpy, pydantic, sklearn, typing.

## Instalación
Clona este repositorio: git clone https://github.com/Gio2M/Api_PI1.git
Navega al directorio del proyecto: cd Api_PI_1
Instala las dependencias: pip install -r requirements.txt

## Uso
Ejecuta la API desde la terminal: uvicorn main:app --reload
Accede a la documentación de la API en: http://127.0.0.1:8000/docs

Tambien puedes acceder a este modelo atraves del servicio web Render en la direcion ip:  https://api1-zzzl.onrender.com/docs

## Endpoints
GET /juegos/{Anio}: Obtiene una lista de juegos lanzados en el año especificado.
GEt /genero/{Anio}: Se ingresa un año y devuelve una lista con los 5 géneros más ofrecidos en el orden correspondiente.
GET /specs/{Anio}: Se ingresa un año y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente.
GET /earlyacces/{Anio}:Cantidad de juegos lanzados en un año con early access.
GET /sentim/{Anio}:Según el año de lanzamiento, se devuelve una lista con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento.
GET /metascore/{Anio}:Top 5 juegos según año con mayor metascore.
POST /predict: Realiza una predicción de precio de juego en Steam en función de las características proporcionadas.

## Contribuciones
¡Las contribuciones son bienvenidas! Si deseas contribuir a este proyecto, por favor crea un pull request.

## Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

## Contacto
Si tienes alguna pregunta o comentario, puedes contactarme en gio75388@gmail.com

## Créditos
FastAPI
pandas
scikit-learn
Steam API
soyhenry.com
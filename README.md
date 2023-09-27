**Airflow Weather Data Pipeline******

Overview
This Airflow project constructs a Directed Acyclic Graph (DAG) designed to fetch weather information from an online weather data API. The data is then stored, transformed, and used to train a prediction algorithm. This data feeds into a dashboard hosted in a dedicated docker-compose.yml file accessible on port 8050.

The DAG is set to run every minute, ensuring the dashboard and prediction model are consistently updated.

Initial Setup
Before running the DAG:

Execute the docker-compose.yaml file.
Create directories named clean_data and raw_files for data storage.
Project Phases:
Data Retrieval from OpenWeatherMap API

Fetches weather data for multiple cities (default: ['paris', 'london', 'washington']). Users can customize the cities by altering the cities variable.
Data is stored as a JSON file in the /app/raw_files directory. Filename format: YYYY-MM-DD HH:MM.json.
& 3. Data Transformation

Task 2: Concatenates the 20 latest files from /app/raw_files, converts them into a CSV named data.csv for the dashboard's latest observations.
Task 3: Transforms all files in the /app/raw_files directory into a fulldata.csv, which will later be used for model training.
& 5. Model Training and Selection

Tasks 4', 4'', and 4''' involve training various regression models: LinearRegression, DecisionTreeRegressor, and RandomForestRegressor.
Models are evaluated using cross-validation, and performance scores are shared using XCom.
Task 5 selects the best-performing model, retrains it on all data, and then saves it.
Execution
The DAG is defined in weather_data_pipeline.py within the Dag directory and can be run using Airflow.

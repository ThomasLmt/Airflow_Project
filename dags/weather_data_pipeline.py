# Importing packages and libraries
import requests
import json
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# OpenWeatherMap API key and link
API_KEY = "72b3a96a804778439aee6e16556e5472"
URL_TEMPLATE = "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"

def fetch_weather_data():
    cities = ['paris', 'london', 'washington']
    data = {}
    for city in cities:
        response = requests.get(URL_TEMPLATE.format(city=city, api_key=API_KEY))
        if response.status_code == 200:
            data[city] = response.json()

    now = datetime.now()
    file_name = now.strftime('%Y-%m-%d %H:%M') + ".json"
    with open(f"/app/raw_files/{file_name}", "w") as f:
        json.dump(data, f)

    return True

def transform_data_into_csv(n_files=None, filename='data.csv'):
    parent_folder = '/app/raw_files'
    files = sorted(os.listdir(parent_folder), reverse=True)
    if n_files:
        files = files[:n_files]

    dfs = []

    for f in files:
        # Extra code because I received an error like it's not a dict
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_temp = json.load(file)

        if isinstance(data_temp, list):
            print(f"File {f} contains a list, not a dict!")
            continue  # skip processing this file

        for city, data_city in data_temp.items():
            dfs.append(
                {
                    'temperature': data_city['main']['temp'],
                    'city': data_city['name'],
                    'pression': data_city['main']['pressure'],
                    'date': f.split('.')[0]
                }
            )

    df = pd.DataFrame(dfs)

    print('\n', df.head(10))

    df.to_csv(os.path.join('/app/clean_data', filename), index=False)


# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 9, 19),  # use your desired start date
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# DAG definition
dag = DAG(
    'weather_data_pipeline',
    default_args=default_args,
    description='A DAG to fetch data from OpenWeatherMap',
    schedule_interval=timedelta(minutes=1),
    catchup=False
)

# Task1: calling the API and fetch some data for Paris, Lisbon, Washington
task1 = PythonOperator(
    task_id='fetch_weather_data',
    python_callable=fetch_weather_data,
    dag=dag,
)

# Task2
def task2_func():
    transform_data_into_csv(n_files=20, filename='data.csv')

task2 = PythonOperator(
    task_id='transform_recent_20_files',
    python_callable=task2_func,
    dag=dag,
)

# Task3
def task3_func():
    transform_data_into_csv(filename='fulldata.csv')

task3 = PythonOperator(
    task_id='transform_all_files',
    python_callable=task3_func,
    dag=dag,
)

def compute_model_score(model, X, y):
    # computing cross val
    cross_validation = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring='neg_mean_squared_error')

    model_score = cross_validation.mean()

    return model_score

def train_and_save_model(model, X, y, path_to_model='./app/model.pckl'):
    # training the model
    model.fit(X, y)
    # saving model
    print(str(model), 'saved at ', path_to_model)
    dump(model, path_to_model)


def prepare_data(path_to_data='/app/clean_data/fulldata.csv'):
    
    df = pd.read_csv(path_to_data)
    
    df = df.sort_values(['city', 'date'], ascending=True)

    dfs = []

    for c in df['city'].unique():
        df_temp = df[df['city'] == c]

        df_temp.loc[:, 'target'] = df_temp['temperature'].shift(1)

        for i in range(1, 10):
            df_temp.loc[:, 'temp_m-{}'.format(i)
                        ] = df_temp['temperature'].shift(-i)

        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    df_final = pd.concat(
        dfs,
        axis=0,
        ignore_index=False
    )

    df_final = df_final.drop(['date'], axis=1)

    df_final = pd.get_dummies(df_final)

    features = df_final.drop(['target'], axis=1)
    target = df_final['target']

    return features, target


def train_model_task(model_class, **kwargs):
    X, y = prepare_data('/app/clean_data/fulldata.csv')
    score = compute_model_score(model_class(), X, y)
    
    task_instance = kwargs['task_instance']
    task_instance.xcom_push(key=f'score_{model_class.__name__}', value=score)

def select_and_retrain_best_model(**kwargs):
    task_instance = kwargs['task_instance']
    score_lr = task_instance.xcom_pull(task_ids='train_linear_regression', key='score_LinearRegression')
    score_dt = task_instance.xcom_pull(task_ids='train_decision_tree', key='score_DecisionTreeRegressor')
    score_rf = task_instance.xcom_pull(task_ids='train_random_forest', key='score_RandomForestRegressor')
    
    scores = {
        LinearRegression: score_lr,
        DecisionTreeRegressor: score_dt,
        RandomForestRegressor: score_rf
    }
    best_model_class = min(scores, key=scores.get)

    X, y = prepare_data('/app/clean_data/fulldata.csv')
    model = best_model_class()
    train_and_save_model(model, X, y, '/app/clean_data/best_model_final.pickle')

    print(f"Retrained and saved the best model: {best_model_class}")

# Task4: LinearRegression
task4 = PythonOperator(
    task_id='train_linear_regression',
    python_callable=train_model_task,
    op_args=[LinearRegression],
    provide_context=True,
    dag=dag
)

# Task5: DecisionTreeRegressor
task5 = PythonOperator(
    task_id='train_decision_tree',
    python_callable=train_model_task,
    op_args=[DecisionTreeRegressor],
    provide_context=True,
    dag=dag
)

# Task6: RandomForestRegressor
task6 = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_model_task,
    op_args=[RandomForestRegressor],
    provide_context=True,
    dag=dag
)

# Task7: Select and retrain the best model
task7 = PythonOperator(
    task_id='select_and_retrain_best_model',
    python_callable=select_and_retrain_best_model,
    provide_context=True,
    dag=dag
)

# Task sequence for Airflow
task1 >> [task2, task3]
task3 >> [task4, task5, task6] >> task7
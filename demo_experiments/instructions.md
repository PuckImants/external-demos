# Vertex Experiments Demo

This demo showcases the following Vertex AI features:

- Pipelines
- Experiments

In this demo, you compile a simple pipeline training a small neural network. You specify a number of runs with different hyperparameters, and submit the metrics from these runs into Vertex Experiments.

## Instructions

### 1 - Specify constant variables

Define the following variables in your notebook:

```
PROJECT_ID
BUCKET_URI
VERSION_NAME # The version of your model
REGION
```

And also a timestamp variable `TIMESTAMP` to separate your experiments from anyone else running the notebooks.

### 2 - Copy the dataset into your own bucket

The dataset can be found here:

```
DATASET_URI = "gs://cloud-samples-data/vertex-ai/structured_data/player_data" 
```

### 3 - Import yoru libraries and define further constants

```
import logging
import os
import time

logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)

import kfp.v2.compiler as compiler
# Pipeline Experiments
import kfp.v2.dsl as dsl
# Vertex AI
from google.cloud import aiplatform as vertex_ai
from kfp.dsl import Artifact, Input, Metrics, Model, Output, component
from typing import NamedTuple

TASK = "regression"
MODEL_TYPE = "tensorflow"
EXPERIMENT_NAME = f"{PROJECT_ID}-{TASK}-{MODEL_TYPE}-{TIMESTAMP}"

# Pipeline
PIPELINE_URI = f"{BUCKET_URI}/pipelines"
TRAIN_URI = f"{BUCKET_URI}/player_data/data.csv"
LABEL_URI = f"{BUCKET_URI}/player_data/labels.csv"
MODEL_URI = f"{BUCKET_URI}/model"
DISPLAY_NAME = "experiments-demo-gaming-data" 
PIPELINE_JSON_PKG_PATH = "experiments_demo_gaming_data.json"
PIPELINE_ROOT = f"gs://{BUCKET_URI}/pipeline_root"
```

### 4 - Define your pipeline functions

You should write the `get_predictions`, `evaluate_model_mae` and `evaluate_model_rmse` functions. 

```
@component(
    packages_to_install=[
        "numpy==1.21.0",
        "pandas==1.3.5", 
        "scikit-learn==1.0.2",
        "tensorflow==2.9.0",
    ]
)
def custom_trainer(
    train_uri: str,
    label_uri: str,
    dropout_rate: float,
    learning_rate: float,
    epochs: int,
    model_uri: str,
    metrics: Output[Metrics], 
    model_metadata: Output[Model], 
    

):

    # import libraries
    import logging
    import uuid
    from pathlib import Path as path

    import pandas as pd
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.metrics import Metric 
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    import numpy as np
    from math import sqrt
    import os
    import tempfile

    # set variables and use gcsfuse to update prefixes
    gs_prefix = "gs://"
    gcsfuse_prefix = "/gcs/"
    train_path = train_uri.replace(gs_prefix, gcsfuse_prefix)
    label_path = label_uri.replace(gs_prefix, gcsfuse_prefix)
    model_path = model_uri.replace(gs_prefix, gcsfuse_prefix)

    def get_logger():

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        return logger

    def get_data(
        train_path: str, 
        label_path: str
    ) -> (pd.DataFrame): 
        
        
        #load data into pandas dataframe
        data_0 = pd.read_csv(train_path)
        labels_0 = pd.read_csv(label_path)
        
        #drop unnecessary leading columns
        
        data = data_0.drop('Unnamed: 0', axis=1)
        labels = labels_0.drop('Unnamed: 0', axis=1)
        
        #save as numpy array for reshaping of data 
        
        labels = labels.values
        data = data.values
    
        # Split the data
        labels = labels.reshape((labels.size,))
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=7)
    
        #Convert data back to pandas dataframe for scaling
        
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
        train_labels = pd.DataFrame(train_labels)
        test_labels = pd.DataFrame(test_labels)
        
        #Scale and normalize the training dataset
        
        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data = pd.DataFrame(scaler.transform(train_data), index=train_data.index, columns=train_data.columns)
        test_data = pd.DataFrame(scaler.transform(test_data), index=test_data.index, columns=test_data.columns)
        
        return train_data,train_labels, test_data, test_labels 
    
        """ Train your Keras model passing in the training data and values for learning rate, dropout rate,and the number of epochs """

    def train_model(
        learning_rate: float, 
        dropout_rate: float,
        epochs: float,
        train_data: pd.DataFrame,
        train_labels: pd.DataFrame):
 
        # Train tensorflow model
        param = {"learning_rate": learning_rate, "dropout_rate": dropout_rate, "epochs": epochs}
        model = Sequential()

        # ToDO: define your model. 
        # The model should have an input layer, 3 hidden layers [500,100,50], the first one should have a dropout layer with rate param['dropout_rate']
        # The model is a regression model!
            
        model.compile(
        tf.keras.optimizers.Adam(learning_rate= param['learning_rate']),
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
        
        model.fit(train_data, train_labels, epochs= param['epochs'])
        
        return model

    # Get Predictions
    def get_predictions(model, test_data):
        # ToDo: return predictions
        return pred

    # Evaluate predictions with MAE
    def evaluate_model_mae(pred, test_labels):
        # ToDo: return mean absolute error
        return mae
    
    # Evaluate predictions with RMSE
    def evaluate_model_rmse(pred, test_labels):
        # ToDo: return root mean squared error
        return rmse    
 
    
    #Save your trained model in GCS     
    def save_model(model, model_path):

        model_id = str(uuid.uuid1())
        model_path = f"{model_path}/{model_id}"        
        path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path + '/model_tensorflow')
    
            
    # Main ----------------------------------------------
    
    train_data, train_labels, test_data, test_labels = get_data(train_path, label_path)
    model = train_model(learning_rate, dropout_rate, epochs, train_data,train_labels )
    pred = get_predictions(model, test_data)
    mae = evaluate_model_mae(pred, test_labels)
    rmse = evaluate_model_rmse(pred, test_labels)
    save_model(model, model_path)

    # Metadata ------------------------------------------

    #convert numpy array to pandas series
    mae = pd.Series(mae)
    rmse = pd.Series(rmse)

    #log metrics and model artifacts with ML Metadata. Save metrics as a list. 
    metrics.log_metric("mae", mae.to_list()) 
    metrics.log_metric("rmse", rmse.to_list()) 
    model_metadata.uri = model_uri

```

### 5 - Write pipeline and compile it.

```
# define our workflow

@dsl.pipeline(name="gaming-custom-training-pipeline")
def pipeline(
    train_uri: str,
    label_uri: str,
    dropout_rate: float,
    learning_rate: float,
    epochs: int,
    model_uri: str,
):

    custom_trainer(
        train_uri = train_uri,
        label_uri = label_uri, 
        dropout_rate = dropout_rate,
        learning_rate = learning_rate,
        epochs = epochs, 
        model_uri = model_uri
    )
```

### 6 - Define your runs

```
runs = [
    {"dropout_rate": 0.001, "learning_rate": 0.001,"epochs": 20},
    {"dropout_rate": 0.002, "learning_rate": 0.002,"epochs": 25},
    {"dropout_rate": 0.003, "learning_rate": 0.003,"epochs": 30},
    {"dropout_rate": 0.004, "learning_rate": 0.004,"epochs": 35},
    {"dropout_rate": 0.005, "learning_rate": 0.005,"epochs": 40},
]
```

### 7 - Submit your pipeline jobs

You should start with 

`for i,run in enumerate(runs):`

and create an `aiplatform.PipelineJob`. 

Make sure to include the experiment name when you submit!

### 8 - Review your experiment results

Use `aiplatform.get_experiment_df()`.
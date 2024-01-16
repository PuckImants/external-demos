In this demo, you will build a very simple Vertex Pipeline with three python components. The idea here is to understand how to define Kubeflow pipelines in python, as well as submitting pipeline jobs to vertex. 


## 0. Prepare data

You should create a bigquery table for the dataset we will be using, uploading the `dataset.csv` to a dataset in the correct region.


## 1. Environment setup

1. Authenticate with gcloud
2. Create a new notebook
3. Install the `google-cloud-pipeline-components` package. 
4. Define the following constants:

```
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

PROJECT_ID="dt-tu-sandbox-dev"
BUCKET_NAME="gs://ovo-demos"

DISPLAY_NAME=f"beans_model_{TIMESTAMP}"
ENDPOINT_NAME=f"train-automl-beans-{TIMESTAMP}"

BQ_SOURCE="bq://dt-tu-sandbox-dev.ml_pipeline.beans"

REGION="europe-west2"

PIPELINE_ROOT = f"{BUCKET_NAME}/auto_ml_pipeline_root/"
PIPELINE_ROOT
```
5. Imports:

```
import kfp

from kfp import compiler, dsl
from kfp.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics

from google.cloud import aiplatform
from typing import NamedTuple

from google_cloud_pipeline_components.v1.dataset import TabularDatasetCreateOp
from google_cloud_pipeline_components.v1.automl.training_job import AutoMLTabularTrainingJobRunOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
```

## 2. Create your pipeline

In this example, we will not use any custom components, instead makign use of the google_cloud_pipeline_components library, which provides components for common Vertex AI tasks. You can find documentation [here](https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-2.8.0/api/v1/automl/training_job.html).

Look at the imports, and you can see pipeline components for dataset creation, automl training, and model deployment. 

Start with this function:

```
@pipeline(name="automl-tab-beans-training-v2",pipeline_root=PIPELINE_ROOT)
def pipeline(
    bq_source: str = BQ_SOURCE,
    display_name: str = DISPLAY_NAME,
    project: str = PROJECT_ID,
    gcp_region: str = "europe-west2",
):
```

you should then:
1. create the dataset from the `bq_source`
2. Train the model, with the following parameters:

```
project=project,
display_name=display_name,
optimization_prediction_type="classification",
budget_milli_node_hours=1000,
column_transformations=[
    {"numeric": {"column_name": "Area"}},
    {"numeric": {"column_name": "Perimeter"}},
    {"numeric": {"column_name": "MajorAxisLength"}},
    {"numeric": {"column_name": "MinorAxisLength"}},
    {"numeric": {"column_name": "AspectRation"}},
    {"numeric": {"column_name": "Eccentricity"}},
    {"numeric": {"column_name": "ConvexArea"}},
    {"numeric": {"column_name": "EquivDiameter"}},
    {"numeric": {"column_name": "Extent"}},
    {"numeric": {"column_name": "Solidity"}},
    {"numeric": {"column_name": "roundness"}},
    {"numeric": {"column_name": "Compactness"}},
    {"numeric": {"column_name": "ShapeFactor1"}},
    {"numeric": {"column_name": "ShapeFactor2"}},
    {"numeric": {"column_name": "ShapeFactor3"}},
    {"numeric": {"column_name": "ShapeFactor4"}},
    {"categorical": {"column_name": "Class"}},
],
dataset=dataset_create_op.outputs["dataset"],
target_column="Class",
location=gcp_region
```
3. create your endpoint
4. deploy your model, using the following:
```
model=,
endpoint=,
dedicated_resources_min_replica_count=1,
dedicated_resources_max_replica_count=1,
dedicated_resources_machine_type="n1-standard-4",
```

model and endpoint will need to point to the relevent outputs of the training and endpoint ops. 

## 3. Create and submit your pipeline job. 

This is done in the same way as the intro hands-on.
# Scikit Learn Hands-on

In this hands-on, you will practice what you have learnt about the sklearn library to build a local model and then deploy to Vertex AI. There are solutions in the `sklearn.ipynb` file. Try to refer to the solutions as little as possible, instead making use of the [sklearn](https://scikit-learn.org/stable/) and [Vertex AI](https://cloud.google.com/vertex-ai/docs) Documentation.

Before you start, create a new notebook in this directory, along with a `scr_dir` folder and a `model_artifacts` folder. 

## 0. Environment Variables

It is useful to define a few constants which you will need throughout the hands-on. 

```
PROJECT_ID = ""
REGION = "us-central1"

BUCKET_NAME = "" # name of bucket where we will store our model artifacts
BUCKET_URI = f"gs://{BUCKET_NAME}"
MODEL_ARTIFACT_DIR = "" # directory where you will save your model in your bucket

IMAGE = "" # name of docker image where you will save for model deployment
REPOSITORY = "" # repository where you will save your docker image

MODEL_DISPLAY_NAME = "" # name of model in vertex AI Model registry
```

## 1. Requirements

run the following code in a notebook cell to create your requirements.txt. You should then install the requirements in your notebook. 

```
%%writefile requirements.txt
fastapi
uvicorn==0.17.6
joblib~=1.1.1
numpy>=1.17.3, <1.24.0
scikit-learn~=1.0.0
pandas
google-cloud-storage>=2.2.1,<3.0.0dev
google-cloud-aiplatform[prediction]>=1.18.2
```

Copy the requirements file to the `scr_dir` folder. 

## 2. Create a bucket for saving model artifacts

Please do so using the `google.cloud.storage` module, and create the bucket in the `us-central1` region.

## 3. Load the dataset

We will be using the `diamonds` dataset from the `seaborn` library. 

## 4. Create the column transforms

The `price` will be our target variable for this exercise, so make a copy of this column and remove it from the dataset to create a pandas dataframe of our independent variables. 

One-hot the categorical variables and scale the numerical variables using the standard scaler. You can use the `make_column_transformer` from `sklearn.compose` to combine transformations. 

## 5. Create the model

We will use a random forest regressor to predict the price of the diamonds. 

## 6. Create the sklearn pipeline

You can use `make_pipeline` from `sklearn.pipeline` to create a pipeline which combines the column transformer with the model. 

## 7. Train the model

Train the model using `pipeline.fit` and test that you can generate predictions on the following example:

```
input_example = [[0.23, 'Ideal', 'E', 'SI2', 61.5, 55.0, 3.95, 3.98, 2.43]]
```

## 8. Save the model to your bucket

Use `joblib.dump` to save your pipeline to the `model_artifacts` folder, and then copy it to your `MODEL_ARTIFACT_DIR` directory in the bucket you made in task 1. 

## 9. Create the CPR Predictor

In the XGBoost example, we simple used the model and a pre-built Vertex container to deploy our model. In this one, we aim to go a step further, by including some pre-processing of data. We will take a simple example, looking at the `clarity` variable in our dataset. Sometimes we will be given data which has long-form versions of the categories, such as `Flawless` for `FL`.

Use the following code to create a dictionary and save it locally:

```
clarity_dict={"Flawless": "FL",
              "Internally Flawless": "IF",
              "Very Very Slightly Included": "VVS1",
              "Very Slightly Included": "VS2",
              "Slightly Included": "S12",
              "Included": "I3"}

import json
with open("model_artifacts/preprocessor.json", "w") as f:
    json.dump(clarity_dict, f)
```

Copy the `proprocessor.json` to the same directory in GCS as your pipeline file.


Then we need to create a class which can read this preprocessor and apply the logic before running predictions. You can run the following in a new cell:

```
%%writefile scr_dir/predictor.py

import joblib
import numpy as np
import json

from google.cloud import storage
from google.cloud.aiplatform.prediction.sklearn.predictor import SklearnPredictor


class CprPredictor(SklearnPredictor):

    def __init__(self):
        return

    def load(self, artifacts_uri: str) -> None:
        """Loads the sklearn pipeline and preprocessing artifact."""

        super().load(artifacts_uri)

        # open preprocessing artifact
        with open("preprocessor.json", "rb") as f:
            self._preprocessor = json.load(f)


    def preprocess(self, prediction_input: np.ndarray) -> np.ndarray:
        """Performs preprocessing by checking if clarity feature is in abbreviated form."""

        inputs = super().preprocess(prediction_input)

        for sample in inputs:
            if sample[3] not in self._preprocessor.values():
                sample[3] = self._preprocessor[sample[3]]
        return inputs

    def postprocess(self, prediction_results: np.ndarray) -> dict:
        """Performs postprocessing by rounding predictions and converting to str."""

        return {"predictions": [f"${value}" for value in np.round(prediction_results)]}
```

Take a minute to understand what is happening in each function here. 


## 10. Create a local docker image using the aiplatform library

We would like to test our model locally before deploying it to Vertex AI endpoints. We can do so using the following:

```
from google.cloud.aiplatform.prediction import LocalModel
from scr_dir.predictor import CprPredictor
```

Use the `LocalModel.build_cpr_model` to create a model which makes use of the preprocessing steps outlined in the `CprPredictor` class. You can find documentation [here](https://cloud.google.com/python/docs/reference/aiplatform/1.19.1/google.cloud.aiplatform.prediction.LocalModel#google_cloud_aiplatform_prediction_LocalModel_build_cpr_model). 

Your `output_image_uri` should point to `f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}"`. 

Make sure to reference your requirements file in the `scr_dir` folder!

## 11. Test local deployment

Create a request file using the following:

```
import json

sample = {"instances": [
  [0.23, 'Ideal', 'E', 'VS2', 61.5, 55.0, 3.95, 3.98, 2.43],
  [0.29, 'Premium', 'J', 'Internally Flawless', 52.5, 49.0, 4.00, 2.13, 3.11]]}

with open('instances.json', 'w') as fp:
    json.dump(sample, fp)
```

Notice how some examples have extended forms for the clarity variable. Our preprocessing should cope with this. 

Deploy your model to a local endpoint, you can find guidance [here](https://cloud.google.com/python/docs/reference/aiplatform/1.19.1/google.cloud.aiplatform.prediction.LocalModel#google_cloud_aiplatform_prediction_LocalModel_deploy_to_local_endpoint). Use example 1 in the documentation as a guide, but using a `request_file` instead of a request. 

Once you are happy that your image is working locally, you can deploy it to Vertex Endpoints.

## 12. Push your docker image to the artifact registry

use `gcloud` to create a repository with the name you specified in your constants, in the correct region and with repository format of `docker`. Use the following to configure authorisation for your docker in workbench:

```
!gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
```

Finally, you can use `local_model.push_image()` to push your local image to the repository. 


## 13. Add your model to the model registry and deploy

You can use `aiplatform.Model.upload` with your local_model, pointing to the artifact directory in your bucket with your model artifacts, and the display name you defined in the constants. You will be returned a aiplatform model variable, `model`.

Use `endpoint = model.deploy(machine_type="n1-standard-2")` to deploy your model to an endpoint. If you navigate to Vertex AI, you will be able to see your model in the model registry, and your endpoint. This can take a few minutes!


## 14. Test your prediction and clean up:

Use the following to test your endpoint:

```endpoint.predict(instances=[[0.23, 'Ideal', 'E', 'VS2', 61.5, 55.0, 3.95, 3.98, 2.43]])```

and clean up using the following:

```
endpoint.delete(force=True)
model.delete()
```

## Congratulations!

You have finished this hands-on with more complex deployments!
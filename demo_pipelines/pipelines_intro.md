# Introduction to pipelines

In this demo, you will build a very simple Vertex Pipeline with three python components. The idea here is to understand how to define Kubeflow pipelines in python, as well as submitting pipeline jobs to vertex. 


## 1. Environment setup

1. Authenticate with gcloud
2. Create a new notebook
3. Install the `google-cloud-pipeline-components` package. 
4. Define the following constants:

```
PROJECT_ID=""
BUCKET_NAME=""

REGION="europe-west2"

PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root/"
PIPELINE_ROOT
```
5. Imports:

```
import kfp

from kfp import compiler, dsl
from kfp.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics

from google.cloud import aiplatform
from typing import NamedTuple
```

## 2. Create your first components:

Components can be defined as python functions using the `kfp.dsl.component` decorator. This will use a default python image to run our function in. Add this example to a new cell:

```
@component()
def product_name(text: str) -> str:
    return text
```

Next, turn the following python function into a component. You will need to add an additional package `emoji` to your component. See how to do so in the documentation [here](https://kubeflow-pipelines.readthedocs.io/en/stable/source/dsl.html#kfp.dsl.component).

```
def emoji(
    text: str,
) -> NamedTuple(
    "Outputs",
    [
        ("emoji_text", str),  # Return parameters
        ("emoji", str),
    ],
):
    import emoji

    emoji_text = text
    emoji_str = emoji.emojize(':' + emoji_text + ':', language='alias')
    print("output one: {}; output_two: {}".format(emoji_text, emoji_str))
    return (emoji_text, emoji_str)
```

Create your third and final component using the function below, this time using the `python:3.9` base image:

```
def build_sentence(
    product: str,
    emoji: str,
    emojitext: str
) -> str:
    print("We completed the pipeline, hooray!")
    end_str = product + " is "
    if len(emoji) > 0:
        end_str += emoji
    else:
        end_str += emojitext
    return(end_str)
```

## 3. Create your pipeline

Having defined our components, we now need to tie them together. We can do this using a python function with the `@pipeline` decorator. 

Create your pipeline, using the following function definition:

``` 
def intro_pipeline(text: str = "Vertex Pipelines", emoji_str: str = "sparkles"):
```

You should pass:
1. the `text` input into the `product_name` component
2. the `emoji_str` input into the `emoji` component
3. the outputs of those two components into the `build_sentence` component.

Hint: you can get the outputs of one component using `component.output` for single output, or `component.outputs["emoji_text"]` for components with multiple outputs. For example:

```
product_task = product_name(text=text)
product = product_task.output
```

You will then add the `@pipeline` decorator, with documentation [here](https://kubeflow-pipelines.readthedocs.io/en/stable/source/dsl.html#kfp.dsl.pipeline). Make sure to add the `name`, `description` and `pipeline_root` parameters to your decorator. 


## 4. Compile your pipeline

Now you have defined your pipeline, you can compile it. You can use the `compiler.Compiler().compile()` function, passing in your pipeline function, and a package path to save it (this can be `intro_pipeline_job.json` for example). You should then see a json file appear in this directory, with the pipeline defined inside. Take a look, this contains lots of information about your pipeline. 

## 5. Run your pipeline on Vertex Pipelines

We can now run our pipeline in a serverless manner in Vertex. To do so, create an `aiplatform.PipelineJob`, documentation [here](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.PipelineJob). You can then run `job.submit()` to submit this job. 

Check your pipeline in the Vertex UI. Any questions, just shout!

## 6. Bonus - Add experiments

add an `experiment` parameter to your `job.submit`, and resubmit to see your parameters get automatically logged in an experiment!
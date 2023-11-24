# XGBoost Demo

In this demo, you train an xgboost model locally in the notebook instance, and then deploy the model using Vertex AI Endpoints.

Please open a new jupyter notebook to begin.

## 1 - Install dependencies

Please install the `xgboost` python package.

## 2 - Specificy Environment Variables

In your jupyter notebook, define the following variables:

```
PROJECT_ID
REGION
MODEL_NAME
VERSION_NAME
```

## 3 - Initialise aiplatform

## 4 - Download the dataset

You can find the dataset at `gs://mortgage_dataset_files/mortgage-small.csv`.

## 5 - Load the data using pandas

The dtypes should be as follows:

```
COLUMN_NAMES = collections.OrderedDict({
 'as_of_year': np.int16,
 'agency_code': 'category',
 'loan_type': 'category',
 'property_type': 'category',
 'loan_purpose': 'category',
 'occupancy': np.int8,
 'loan_amt_thousands': np.float64,
 'preapproval': 'category',
 'county_code': np.float64,
 'applicant_income_thousands': np.float64,
 'purchaser_type': 'category',
 'hoepa_status': 'category',
 'lien_status': 'category',
 'population': np.float64,
 'ffiec_median_fam_income': np.float64,
 'tract_to_msa_income_pct': np.float64,
 'num_owner_occupied_units': np.float64,
 'num_1_to_4_family_units': np.float64,
 'approved': np.int8
})
```

## 6 - Data Validation

Check that the approved column only contains 1 or 0.

## 7 - Feature Engineering

For the category type variables one-hot encode your independent variables.

<details>
    <summary>hint</summary>
use `pd.get_dummies`   
</details>

## 8 - Split your data

Use split your data 

<details>
    <summary>hint</summary>
use the sklearn `train_test_split` function   
</details>

## 9 - Fit the xgboost classifier model

## 10 - Generate accuracy metric against your test set

## 11 - Save your model locally

## 12 - Upload the xgboost model to Vertex AI

## 13 - Deploy your model to Vertex AI Endpoints

## 14 - Generate a prediction

Use the following example input.

```
example_input = [
    [2016.0, 1.0, 346.0, 27.0, 211.0, 4530.0, 86700.0, 132.13, 1289.0, 1408.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
  ]
```

## 15 -  Clean up

Remove your model and endpoint using the following commands:

```
endpoint.delete(force=True) 
model.delete()
```
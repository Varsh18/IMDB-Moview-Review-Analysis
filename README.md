# IMDB Movie Review Analysis

## Summary
In this project, I have used IMDB Movie review dataset. This dataset will be used for training the model using 
1) an Automated ML 
2) Hyperdrive.

After that ,we will compare their performance and deploy the best model from both Automated ML and hyperdrive using python SDK. Finally , deployed model is consumed via an HTTP API using model endpoint. An HTTP API is a URL that is exposed over the network so that interaction with a trained model can happen via HTTP requests.

## Microsoft Azure ML
   The [Azure](https://ml.azure.com) Machine Learning service empowers developers and data scientists with a wide range of productive experiences for building, training, and deploying machine learning models faster. It also accelerates time to market and foster team collaboration with industry-leading MLOps—DevOps for machine learning.It Innovates on a secure, trusted platform, designed for responsible machine learning.
   
## Project Overview
Sentiment analysis (or opinion mining) is a natural language processing technique used to determine whether data is positive, negative or neutral. Sentiment analysis is often performed on textual data to help businesses monitor brand and product sentiment in customer feedback, and understand customer needs.Sentiment analysis is extremely important because it helps businesses quickly understand the overall opinions of their customers. 

This project was aimed at providing sentiment analysis (positive , negative) using [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle.IMDB dataset having 50K movie reviews for natural language processing or Text analytics.This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms.

Hyperparameters are adjustable parameters you choose for model training that guide the training process. The HyperDrive package helps you automate choosing these parameters. For example, you can define the parameter search space as discrete or continuous, and a sampling method over the search space as random, grid, or Bayesian. Also, you can specify a primary metric to optimize in the hyperparameter tuning experiment, and whether to minimize or maximize that metric.

AutoML involves the application of DevOps principles to machine learning, in order to automate all aspects of the process. For example, we can automate feature engineering, hyperparameter selection, model training, and tuning. With AutoML, we can create hundreds of models a day,get better model accuracy
deploy models faster.

The best performing model from the AutoML experiment and Hyperdrive was deployed as a webservice using Azure Container Instance (ACI).HTTP post requests were sent to test the Model Endpoints.

## Project Set Up and Installation

The project was performed using the Microsoft Azure Machine Learning Studio that provides resources for training machine learning models.Below are the steps performed to train and deploy the model and consume it as a web service.

Sign in to [Azure Portal](https://portal.azure.com/).

Search for Machine Learning Service and launch it

In Azure ML dashboard, navigate to Cluster > Create a Compute cluster. This cluster is used for executing Jupyter notebooks.

In the Notebook section, upload the necessary files needed for the project(hyperparameter_tuning.ipynb, automl.ipynb, train.py and score.py).

hyperparameter_tuning.ipynb file contains the code to perform model training using Hyperdrive and deploy the best model as a webservice. Finally , the deployed service is consumed throught HTTP requests and the response is recorded.

automl.ipynb file contains the code to perform model training using AutoML with Deep Learning and deploy the best trained model as a webservice. Finally ,the deployed service is consumed throught HTTP requests and the response is recorded.

After successful completion of project execution , webservice and the compute cluster is deleted.

## Dataset

### Overview
The dataset used in this project is [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from [kaggle](https://www.kaggle.com/).This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.This dataset contains 50k movie reviews which 49582 unique values. The attributes in the dataset are review and sentiment. The attribute review contains the user review and sentiment attribute contains two values (positive , negative) corresponding to the user review.

### Task
The main idea of this project is analyse the binary sentiments for the IMDB Movie review. The features in the dataset are review column which has the user reviews.Text preprocesing is challenging one. Based on the goal of the project, we need to remove the unwanted information from the data.Machine Learning can produce more accurate results when we process the text data properly.
Below are the steps used in preprocessing the text data.

1. Remove HTML tags
2. Remove extra whitespaces
3. Convert accented characters to ASCII characters
4. Expand contractions
5. Remove special characters
6. Lowercase all texts
7. Lemmatization 
8. Remove numbers
9. Remove stopwords
10. Convert words to numeric form

### Access
The dataset is uploaded to the Github repository and accessed in the Jupyter notebook using TabularDatasetFactory class
   
	from azureml.data.dataset_factory import TabularDatasetFactory
	ds=TabularDatasetFactory.from_delimited_files(path="https://raw.githubusercontent.com/Varsh18/IMDB-Moview-Review-Analysis/master/IMDB-Dataset.csv")

## Automated ML
AutoML involves the application of DevOps principles to machine learning, in order to automate all aspects of the process. For example, we can automate feature engineering, hyperparameter selection, model training, and tuning. With AutoML, we can create hundreds of models a day,get better model accuracy
deploy models faster.

	automl_settings = {
	    "experiment_timeout_minutes": 20,
	    "primary_metric": 'accuracy',
	    "max_concurrent_iterations": max_nodes, 
	    "max_cores_per_iteration": -1,
	    "enable_dnn": True,
	    "enable_early_stopping": True,
	    "validation_size": 0.3,
	    "verbosity": logging.INFO,
	    "enable_voting_ensemble": False,
	    "enable_stack_ensemble": False,
	}

	automl_config = AutoMLConfig(task = 'classification',
                             debug_log = 'automl_errors.log',
                             compute_target=compute_cluster_name,
                             training_data=train_data,
                             label_column_name='sentiment',
                             blocked_models = ['LightGBM', 'XGBoostClassifier'],
                             **automl_settings
                            )
			  
Automated ML Settings:
| Name        | Description           | Value  |
| ------------- |:-------------:| -----:|
|experiment_timeout_minutes     | It defines how long experement will run | 20 |
| primary_metric    | metric that Automated ML will optimize for model selection      |   accuracy |
| max_concurrent_iterations | Represents the maximum number of iterations that would be executed in parallel. The default value is 1.|    8 |
|max_cores_per_iteration     | The maximum number of threads to use for a given training iteration. Equal to -1, which means to use all the possible cores per iteration per child-run. | -1 |
| enable_dnn    | Whether to include DNN based models during model selection.      |   True |
| enable_early_stopping | Whether to enable early termination if the score is not improving in the short term.      |    True |
| validation_size | What fraction of the data to hold out for validation when user validation data is not specified      |    0.3 |
|verbosity     | The verbosity level for writing to the log file | logging.INFO |
| enable_voting_ensemble    | Whether to enable/disable VotingEnsemble iteration      |   False |
| enable_stack_ensemble | Whether to enable/disable StackEnsemble iteration.      |    False |


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

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

![Compute instance](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/compute_instance.png)

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


Automated ML configuration:

| Name        | Description           | Value  |
| ------------- |:-------------:| -----:|
|debug_log     | The log file to write debug information to. | automl_errors.log |
| compute_target    | compute target which is used to run Automated ML experiments      |   compute_cluster_name |
| training_data | The training data to be used within the experiment |    train_data |
|label_column_name     | The name of the label column | sentiment |
| blocked_models    | A list of algorithms to ignore for an experiment.      |   'LightGBM', 'XGBoostClassifier' |

![AutoML Run submitted](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/automl/run_submitted_jupyter1.png)

![AutoML run details](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/automl/run_completed_jupyter1.png)

![AutoML run details](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/automl/run_completed_jupyter2.png)

### Results
The best model got from Automated ML is  StandardScalerWrapper LogisticRegression with accuracy of 0.8370

### Future Work
1. To do more preprocessing and compare the results
2. Increasing the experiment timeout to improve the primary metric i.e., Accuracy

![Best Run details Jupyter](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/automl/best_model_jupyter.png)

![Auto ML Best Run Details](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/automl/best_run_azure.png)

![Best Run properties](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/automl/best_run_properties.png)


## Hyperparameter Tuning
The scikit-learn pipeline consists of tuning the hyperparameter of a logistic regression binary classification model using HyperDrive.Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable (y) is a binary variable.HyperDrive is a package that helps you automate choosing parameters.

Main Steps for Tuning with HyperDrive:

	1.Define the parameter search space. This could be a discrete/categorical variable (e.g., apple, banana, pair) or it can be a continuous value (e.g., a time series value).
	2.Define the sampling method over the search space. This is a question of the method you want to use to find the values. For example, you can use a random, grid, or Bayesian search strategy.
	3.Specify the primary metric to optimize. For example, the Area Under the Curve (AUC) is a common optimization metric.
	4.Define an early termination policy. An early termination policy specifies that if you have a certain number of failures, HyperDrive will stop looking for the answer.

   We start by setting up a training script "train.py". It contains dataset creation, train and evaluate Logistic regression
model from Scikit learn.The dataset is tabular which is imported from a URL in the training script. Then by using python sdk, Compute cluster is created and configuring  the training run by creating a HyperDriveConfig and AutoMLConfig for comparison.After that ,run is submitted and best model was saved and registered.

Sampling the hyperparameter space
 Azure Machine Learning supports the following methods to use over the hyperparameter space:

	1.Random sampling
	2.Grid sampling
	3.Bayesian sampling

This project uses Bayesian sampling as parameter sampling.

Bayesian sampling tries to intelligently pick the next sample of hyperparameters, based on how the previous samples performed, such that the new sample improves the reported primary metric.when using Bayesian sampling, the number of concurrent runs has an impact on the effectiveness of the tuning process. Typically, a smaller number of concurrent runs leads to better sampling convergence. That is because some runs start without fully benefiting from runs that are still running.

	ps = BayesianParameterSampling ({
	    "--C":choice(0.00001,0.0001,0.001,0.01,0.1,1,10,100,200,500,1000),
	    "--max_iter":choice(50,100,200,300,500,1000)
	})

This will define a search space with two parameters, C and max_iter. The C can have a a choice of [0.00001,0.0001,0.001,0.01,0.1,1,10,100,200,500,1000] , and the max_iter will be a choice of [50,100,200,300,500,1000].

![experiment running](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/hyperdrive/exp_running_jupyter1.png)

![experiment completed_azure](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/hyperdrive/exp_completed_azure.png)

![Run completed](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/hyperdrive/rundetails2.png)

![Hyperdrive run details](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/hyperdrive/rundetails1.png)

![Run details](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/hyperdrive/rundetails3.png)

![Run properties](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/hyperdrive/rundetails4.png)

### Results
The highest accuracy of the Logistic Regression Model acheived was 0.8878.The best run metrics for this model are:
Best Run Metrics: {'Regularization Strength:': 0.01, 'Max iterations:': 1000, 'Accuracy': 0.8878}

![Best run details](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/hyperdrive/exp_running_jupyter4.png)

![Best run details azure](https://github.com/Varsh18/IMDB-Moview-Review-Analysis/blob/master/images/hyperdrive/bestmodel_azure.png)
### Future Work
1. Need to try different parameter sampling with early termination policy and compare the results.
2. Convert the best model to ONNX Format.
3. Increase the maximum total run count by different hyperparameters and check the results

## Model Deployment
Deployment is about delivering a trained model into production so that it can be consumed by others.Configuring deployment settings means making choices on cluster settings and other types of interaction with a deployment.
### ACI and AKS: 
Both ACI and AKS are available in the Azure ML platform as deployment options for models.

ACI is a container offering from Azure, which uses container technology to quickly deploy compute instances. The flexibility of ACI is reduced as to what AKS offers, but it is far simpler to use.For this project, I have used Azure Container Instanced to deploy the best model.

AKS, on the other hand, is a Kubernetes offering. The Kubernetes service is a cluster that can expand and contract given on demand, and it does take more effort than the container instance to configure and setup.

To gain knowledge about both Hyperdrive and AutoML deployed ,I have deployed best runs from both AutoML and hyperdrive and consumed the model endpoints.


## Hyperdrive Deployement
The best run model is saved and registered.For deploying hyperdrive model, we need to write environment file and scoring file. By using environment file and deployment details , I have deployed the hyperdrive best model is Azure Container Instance as it is simple and easier to deploy.

## Automated ML Deployment
Firstly ,the best model is saved and registered.The best run contains environment and scoring file.With the help of that, we can deploy the best model.I have used Azure Container Instance.ACI has more flexible benefits than Azure Kubernates Services. I have enabled logging before deploying the model

## Screen Recording
Click [here](https://drive.google.com/file/d/1VRmS_4KrbmPj_pBOKeGoYvKderx0lxeq/view?usp=sharing) to view the detailed implementation

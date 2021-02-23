# Project: Titanic Survival Prediction

In this project we are going to train the same dataset but using two different approaches:

- Approach No. 1 will be AutoML API from Azure 
- Approach No. 2 will be HyperDrive API (also from Azure) 

For both approaches we will retrieve the best model and compare the best models among them. The winner of this comparison will be registered and deployed. 
Finally and after the deployement has taken place we are going to test the best model end point by sending a request.

The chart below should visualize the above explanation:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/Diagram.PNG)

## Dataset
As you probably have guessed from the project title we will be working with the "Titanic Dataset" which is already a classical dataset to learn Machine Learning.

### Overview
This data is an open source set which is available online through the "OpenML" Organization ( https://www.openml.org/ ) 
If you are interested on checking the dataset by yourself you can click on the following link: https://www.openml.org/data/get_csv/16826755/phpMYEkMl

### Task
The main task for this project will be to build a predictive model that answers the question: “what sorts of people were more likely to survive?” 
To answer the above stated question we are going to give the model different input variables such as age, type of cabin the passanger had, etc.

More specifically, we are going to concentrate on the following features:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/features.PNG)

The above features will be pre-processed to facilitate computation during training.If you are interested in this step you can take a look to the following files:

- pre-process.py:
This file contains different pre-process functions which will be then imported from the train.py file

- train.py:
This file imports the functions from the pre-process file and wraps all of them into a function which is then call the "clean_data" fuction.
In addition to the clean data function the script applies the following steps:
  - Scales the independent variables
  - Splits the data into train and test
  - Performs a logistic regression

If you would like to take a look to the scripts here you will find the link to them:

- Pre-process file: \
https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/starter_file/pre_preprocess.py

- Train file: \
https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/starter_file/train.py



### Access
During the project two differents ways were implemented in order access the data set.

For the experiment using AutoML we just simply imported it to the notebook by using the TabularDatasetFactory class and the method ".fromDelimetedFiles"
Afterwards the dataset will be registered

The following pictures displays the steps mentioneds above:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/dataset_AutoML.PNG)


For the experiment using the HyperDrive we did the same as with the AutoMl; the only difference is that it was performend within the "train.py" script (rows 31-33 from script: https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/starter_file/train.py )


## Automated ML
First of all it is important to give a generic view of the AutoML settings and configuration that were used for this experiment.

The first step is to create the AutoML settings which will be afterwards passed to the AutoML Configuration as part of the parameters needed within the configuration.
Let's take briefly a look into the AutoML settings parameters used in this experiment:

- experiment_timeout_minutes:\
Refers to the maximum amount of time in minutes that all iterations combined can take before the experiment terminates. For this project the timeout was set to 20 Minutes.

- max_concurrent_iterations: \
Refers to the maximum number of iterations that would be executed in parallel. Important to mention is that values for all experiments should be less than or equal to the maximum number of nodes of the compute cluster.

- primary_metric: \
Refers to the metric that AutoML Process will optimize for model selection.For this project the primary metric selected was the accuracy. Other possibilities would have been the   following: \
  AUC_weighted \
  average_precision_score_weighted \
  norm_macro_recall \
  precision_score_weighted

Now let's move on with the AutoML Configuration and explain briefly those parameters:

- compute_target: \
Refers to the computer target resource name (notebook138164) on which we will let the model training run.

- task: \
Refers to the type of task that we want to solve. In this case we are interested in classifiying (as accurate as possible) if someone would survive (1) or not (0). Therefore the type of taks entered is "classification".

- training_data: \
Refers to the dataset that we will use for training the model.Remeber that in this case we have stored the dataset in a variable call training_dataset.

- label_column_name: \
Refers to name of the "dependent variable" of interest. In this case the name is "survived".

- path: \
Refers to the path in which our experiment is or will take place.

- enable_early_stopping: \
It supports early termination of low-performance runs.

- featurization: \
Setup as 'Auto'which is the default setting and specifies that, as part of preprocessing, so called data guardrails and featurization steps are to be done automatically. \
The guardrails help to identify potential issues with the data (for example, missing values or class imbalance). They also help to take corrective actions for improved results.\
Featurization steps provide techniques that are automatically applied to the input data as part of the pre-process.\
Plese for more details checked on the documentation,it is definetely worth it: 
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-features#featurization \
In our project I decided to pre-process the data in advance to reduce computation and be able to save some ressources.\
The picture below shows the data guardrail states that were proved during trainig of the dataset. As you can see and due to the pre-process mentioned above all the states were flagged as "Passed" indicating that no data problems were detected, thus no additional action was required either from side nor from the automatic featurization process. \
![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/Data_Guardrails.PNG)

- debug_log: \
Refers to the name of log file to write debug information to.If not specified, 'automl.log' is used.

- automl_settings: \
Kwargs allowing us to pass keyworded variable length of arguments to the configuaration. In this case the AutoML Settings.

In the below picture you could see the code snippet which captures the AutoML Settings and Configuration:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/AutoML_Config.PNG)


### Results
After having submitted the experiment run (based on the AutoML Configuration). This is what happened:

1) We were able to see the run details of the experiment as shown above:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/0-Experiment_Running_AutoML.PNG)

2) By using the RunDetails widget we were able to find out that the experiment was done.

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/1-Run_Details_AutoML_Finished.PNG)

3) By checking the experiment results we were able to see the a list containing all the child runs as well as the list of all models tested:

Child Runs:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/2-Run_Details_AutoML_Various_Exp.PNG)

List of all Models:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/3-List_of_All_Model_AutoML.PNG)

4) Out of the list of all models we were able to see which one was the one presenting the best results. From the picture above you can see that the Run 51 gave the best model with the following metrics: 

Accuracy:  0.82521 \
AUC Macro: 0.86622 \
AUC Micro: 0.88014 

Below you can see the picture confirming the above metrics:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/4-Best_Model_AutoML_Run_ID.PNG)

5) Best Model:
The results of the AutoML gave as a winner a "VotingEnsemble" model with an accuracy of 0.82521.The voting ensemble method combines conceptually different machine learning classifiers and uses a majority vote or the average predicted probabilities (soft vote) to predict the class labels. Such a classifier can be useful for a set of equally well performing model in order to balance out their individual weaknesses. \
By retrieving the properties and/or outputs of the model we are able to see the model parameters as well:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/best_model_params.PNG)

6) Room for improvement: 
As we saw it earlier the data input given was very solid pre-processed. You can see this as well on the results of the data guardrails where every staged was "passed".
Thus there is my opinion not much additional if something at all to do to improve the model. I guess you could argue that you could test some of the "blocked" models to see if the output would be better. Nevertheless I will remain with the opionion of sticking to the selected "VotingEnsemble" method, then in real life or real projects you do not have the time to concentrate in "too depth" in improving for let's say 0.01 points. 

## Hyperparameter Tuning
For the Hyperparameter Tuning we will have a "go" with a Logistic regression since this is a simple and very efficient method for binary and linear classification problems. In addition to that the model is very easy to realize and achieves very good performance with linearly separable classes. Due to this it has become an extensively employed algorithm for classification problems within the industry and scientific analysis projects.

With respect to the paremeters that will be used for the Hyperparameter Tuning experiment we are going to concentrate on the following:

- Inverse of Regularization Strength ('--C'):
We will use this parameter to train the model with the goal of having a better generalization of the model.This would lead to better performance on unseen data, by preventing the algorithm from overfitting the training dataset.Concerning the values it is important to know that smaller values would specify stronger regularization.

- Maximum Number of Iterations to solve to converge ('max_iter'):
Refers to the maximum number of iterations taken for the solvers to converge.

In the picture below we can see the HyperDrive Configuration which has included the above mentioned parameters:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/9-HyperDrive_Configuration.PNG)

In addition to the Logistic Regression parameters notice please the additional parameters that were passed to the HyperDrive Configuration. 

- Early Stopping Policy:\
This policy allows to automatically terminate poorly performing runs with an early termination policy. The result of this early termination is to improve computational efficiency. In this specif case the setup of the policy was the following: Bandit Policy which based on slack factor/slack amount and evaluation interval terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.

- cpu_cluster:\
Here we used the already existing cpu cluster that we created when performing the AutoMl Experiment.

- primary_metric: \
Since we want to compare the performance of both experiments (AutoML & HyperParameterTuning) the primary metric remains the same (accuracy) and the goal is to maximize the primary metric (PrimaryMetricGoal.MAXIMIZE).

- max_total_runs & max_concurrent_runs :\
These would be the same as the values that we used to perform the AutoML Experiment.

- estimator: \
We are going to use the estimator for training in Scikit-learn experiments.

### Results
After having submitted the experiment run (based on the hyperdrive_config ).This is what happened:

1) We were able to see the run details of the experiment as shown above:

Experiment Name:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/8-HyperDrive_Experiment.PNG)

Run Details:
![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/10-HyperDrive_Experiment_Run.PNG)

As a side note I have added an additional picture where you can see the registration of both experiments (AutoML & HyperDrive) that took place:
![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/11-Visualizing_Both_Experiments.PNG)


2) After the experiment was finished we were able to take a look to the best model:

![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/12-HyperDrive_Best_Run.PNG)

3) Improvement room: \
The Logistic Regression Parameters concentrated only on the inverse of regularization strength ('--C') and the maximum number of iterations to solve to converge ('max_iter'). Nevetheless it would interesting to add additional parameters to observe how the model performs. Some proposals could be the type of penalty, different class weights or different type of solver.

4) Deciding on which model to deployed: \
Due to the fact that the metrics were higher on the model retrieved from the AutoML instead of the model from the Hyperparameter tuning we decided to deploy the AutoML model

## Model Deployment
For Model Deployment you can see below the steps that we took:

1) Registering the model: \
![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/5-Registering_Best_Model_AutoML.PNG)

2) Define inference configuration: \
Due to the fact that we did not deploy using the GUI is important to define the environment used to run the model \
The inference configuration references the following entities, which are used to run the model when it's deployed:

- An entry script, named score.py, loads the model when the deployed service starts. This script is also responsible for receiving data, passing it to the model, and then returning a response.

- An Azure Machine Learning environment. An environment defines the software dependencies needed to run the model and entry script.

If you whish to take a look to the score.py file you can access to it via this link:
(https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/starter_file/score.py)


3) Deploying the model (and inference config): \
![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/Deploying_Model.PNG)

4) Deployment Verification: \
Verifiying URI, Endpoint and Application Insights:
![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/6-Best_Model_AutoML_Deployed.PNG)

5) Querying the endpoint with a sample input
![alt text](https://github.com/MarceloLandaverde/udacity-capstone-project/blob/master/Pictures/7-Requests_Best_Model_AutoML_Deployed.PNG)


## Screen Recording
Please find below a short video link focusing on the following points:

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

https://www.youtube.com/watch?hd=1&v=Njvp20OCG84&feature=youtu.be



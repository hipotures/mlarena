## Goal: Develop a High-Performing model using data from a mental health survey

You are tasked with developing a machine learning model. Your goal is to use data from a mental health survey to explore factors that may cause individuals to experience depression. Target variable is 'Depression'. Your goal is to create a high-performing model using various algorithms and ensemble methods.

### Steps to Complete the Task

#### 1. Preprocessing
- Handle missing values appropriately.
- Encode categorical variables.
- Scale numerical features if necessary.
- Split the data into training and testing sets.

#### 2. Model Development
Use the following algorithms to create individual models:
- XGBoost
- LightGBM
- CatBoost

For each model:
- Implement cross-validation for hyperparameter tuning.
- Train the model on the training data.
- Make predictions on the test set.
- Calculate the Accuracy score for each model.

#### 3. Ensemble Methods
Implement the following ensemble techniques:
- StackingClassifier: Use the above models as base learners and a logistic regression as the meta-learner.
- VotingClassifier: Combine the above models using soft voting.

#### 4. GPU Utilization
- For XGBoost, use the following parameters to enable GPU acceleration:
  - `params["device"] = "cuda"`
  - `params["tree_method"] = "hist"`
- For other algorithms, utilize GPU acceleration if available.

#### 5. Final Evaluation
- Evaluate all models (including individual and ensemble models) using Accuracy metric on the test set.
- Compare the performance of all models.

Remember to focus on maximizing the Accuracy score, as this is the primary evaluation metric for this task.

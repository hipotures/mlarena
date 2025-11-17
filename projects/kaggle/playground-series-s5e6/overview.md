# Playground Series - Season 5, Episode 6: Predicting Optimal Fertilizers

## Competition Overview

### Description
The Playground Series is a series of fun, beginner-friendly competitions designed to provide an approachable environment for learning and practicing machine learning skills. 

Season 5, Episode 6 focuses on **"Predicting Optimal Fertilizers"** - a multi-class classification problem where participants need to predict the appropriate fertilizer type based on various soil and crop characteristics. This competition addresses a real-world agricultural challenge of recommending the right fertilizer based on environmental and soil conditions.

### Goal
Build a machine learning model to predict the most suitable fertilizer for a given set of agricultural conditions. The model should recommend the optimal fertilizer type based on:
- Environmental factors (temperature, humidity)
- Soil characteristics (type, moisture content)
- Crop type being grown
- Current nutrient levels in the soil (NPK values)

### Evaluation Metric
The competition uses **Mean Average Precision @ 3 (MAP@3)** as the evaluation metric.

**MAP@3 Formula:**
For each observation, you can predict up to 3 `Fertilizer Name` values. The metric is calculated as:

```
MAP@3 = (1/U) × Σ(u=1 to U) Σ(k=1 to min(n,3)) P(k) × rel(k)
```

Where:
- `U` is the number of observations
- `P(k)` is the precision at cutoff k
- `n` is the number of predictions per observation
- `rel(k)` is an indicator function equaling 1 if the item at rank k is a relevant (correct) label, zero otherwise

**Important:** Once a correct label has been scored for an observation, that label is no longer considered relevant for that observation, and additional predictions of that label are skipped in the calculation.

**Example:** If the correct label is `A` for an observation, all these predictions score an average precision of 1.0:
- `[A, B, C, D, E]`
- `[A, A, A, A, A]` 
- `[A, B, A, C, A]`

### Submission Format
For each `id` in the test set, you may predict up to 3 `Fertilizer Name` values, with the predictions space delimited. The file should contain a header and have the following format:

```
id,Fertilizer Name
750000,14-35-14 10-26-26 Urea
750001,14-35-14 10-26-26 Urea
...
```

### Key Features
- Beginner-friendly dataset with clear features
- Tabular data format suitable for traditional ML algorithms
- Multi-class classification problem
- Real-world agricultural application
- Balanced focus on feature engineering and model selection

### Agricultural Context
The competition involves predicting recommendations for fertilizers containing the three main macronutrients:
- **Nitrogen (N)** - Essential for leaf growth and chlorophyll production
- **Phosphorus (P)** - Critical for flower, fruit, and root development  
- **Potassium (K)** - Important for overall plant health and disease resistance

Different crops and soil conditions require different NPK ratios, making this a practical problem in precision agriculture.

### Dataset Overview
The dataset contains information about:
- **Environmental conditions**: Temperature, Humidity
- **Soil properties**: Moisture level, Soil Type (Sandy, Clayey, etc.)
- **Crop information**: Type of crop being grown
- **Nutrient levels**: Current Nitrogen, Phosphorous, and Potassium levels
- **Target variable**: Fertilizer Name (the optimal fertilizer recommendation)

### Popular Approaches
Based on community notebooks:
- XGBoost and other gradient boosting methods
- Random Forest classifiers
- Feature engineering focusing on NPK ratios
- Ensemble methods combining multiple models
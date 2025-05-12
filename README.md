# Titanic Survival Prediction

**Written by Lily Gates**  
*May 2025*

## Description
This project predicts the survival outcomes of passengers aboard the Titanic using machine learning. It investigates the influence of various factors such as age, gender, passenger class, and fare on survival chances. The analysis utilizes a Random Forest classifier, which outperforms other models like Logistic Regression and Decision Tree in predicting survival based on historical data.

## Methodology
The analysis uses supervised learning, employing three classification models: Logistic Regression, Decision Trees, and Random Forest. 

The methodology includes:
* **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
* **Model Training**: The models are trained on the dataset, which is split into training and test sets.
* **Model Evaluation**: The models are evaluated using performance metrics like accuracy, precision, recall, and F1-score. A confusion matrix is also used to assess model performance.
* **Feature Importance**: The models rank features based on their contribution to the survival prediction.

## Required Dependencies
To run the project, the following Python libraries are required:
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`

## Output
The script generates:
* **Feature Importance Plots**: Visualizations showing the most influential factors in predicting survival (e.g., age, gender, fare).
* **Confusion Matrix**: For each model, visualizing true positives, false positives, true negatives, and false negatives.
* **Model Performance Metrics**: Including accuracy, precision, recall, and F1-score for each model.

## Limitations
Despite the Random Forest model outperforming the other models in key metrics, there are several limitations:
1. **Limited Feature Set**: The model was trained on a limited set of features, excluding potentially important variables like cabin location, family identifiers, or group ticket information. This simplification may have overlooked crucial survival patterns.
2. **Overfitting and Bias**: Random Forest models are prone to overfitting, especially when there are many distinct values in the features. The model could also be biased toward features with many categories, such as class or fare, rather than accounting for more nuanced factors.
3. **Contextual Factors**: The analysis does not include critical contextual factors such as proximity to lifeboats, crew behavior, or personal connections, all of which were likely influential during the Titanic disaster.
4. **Generalizability**: The model was validated using a holdout portion of the same dataset, so its performance on unseen data or in different scenarios remains untested.

## Future Improvements
* Experiment with different scaling methods for numerical features.
* Test different tree depth levels for the Decision Tree model to avoid overfitting.
* Explore alternative methods for addressing missing "Age" values.
* Use the Kaggle `test.csv` file to compare the performance of the Random Forest model trained on `train.csv`.

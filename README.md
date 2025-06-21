# Titanic Machine Learning 🛳️💻

Welcome to the Titanic Machine Learning repository! This project predicts the survival of Titanic passengers using various machine learning techniques. It explores key factors such as age, gender, and fare to identify what influences survival rates. 

[![Releases](https://img.shields.io/github/release/gamy703/titanic_machine_learning.svg)](https://github.com/gamy703/titanic_machine_learning/releases)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Data Exploration](#data-exploration)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The Titanic disaster remains one of the most discussed maritime tragedies. In this project, we aim to analyze the Titanic dataset to predict passenger survival. By applying machine learning algorithms, we can identify which factors played a significant role in survival. This project uses Logistic Regression, Decision Trees, and Random Forest algorithms to perform classification.

## Features

- Predicts passenger survival based on various features.
- Utilizes Logistic Regression, Decision Tree, and Random Forest algorithms.
- Analyzes key factors like age, gender, and fare.
- Visualizes data for better understanding.
- Easy to use and modify.

## Technologies Used

This project employs several technologies and libraries:

- **Python**: The primary programming language.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib**: For data visualization.
- **Seaborn**: For statistical data visualization.
- **Scikit-learn**: For implementing machine learning algorithms.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gamy703/titanic_machine_learning.git
   cd titanic_machine_learning
   ```

2. **Install Required Libraries**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   You can find the Titanic dataset on [Kaggle](https://www.kaggle.com/c/titanic/data). Download the dataset and place it in the project directory.

4. **Run the Project**:
   Execute the main script to see the predictions:
   ```bash
   python main.py
   ```

5. **Check Releases**:
   For the latest updates and releases, visit [Releases](https://github.com/gamy703/titanic_machine_learning/releases).

## Data Exploration

Before diving into machine learning, it’s crucial to explore the dataset. The Titanic dataset contains various features that can influence survival:

- **PassengerId**: Unique identifier for each passenger.
- **Survived**: Survival status (0 = No, 1 = Yes).
- **Pclass**: Ticket class (1st, 2nd, 3rd).
- **Name**: Passenger name.
- **Sex**: Gender of the passenger.
- **Age**: Age in years.
- **SibSp**: Number of siblings or spouses aboard.
- **Parch**: Number of parents or children aboard.
- **Ticket**: Ticket number.
- **Fare**: Fare paid for the ticket.
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

### Visualizing Data

We utilize Matplotlib and Seaborn to visualize relationships between different features and survival rates. Some key visualizations include:

- **Survival by Gender**: Understanding how gender affects survival rates.
- **Age Distribution**: Analyzing age groups and their survival rates.
- **Fare Distribution**: Exploring how fare correlates with survival.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example visualization
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Survival Count by Gender')
plt.show()
```

## Machine Learning Models

This project implements three primary machine learning models:

### 1. Logistic Regression

Logistic Regression is a statistical method for predicting binary classes. It estimates the probability that a given input point belongs to a certain class.

### 2. Decision Tree

A Decision Tree uses a tree-like model to make decisions based on feature values. It splits the data into subsets based on the value of features.

### 3. Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees and merges them to improve accuracy and control overfitting.

### Model Evaluation

Each model is evaluated using metrics such as accuracy, precision, recall, and F1-score. We utilize cross-validation to ensure our models generalize well to unseen data.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example code for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy}')
```

## Results

After training and evaluating the models, we compare their performance. The Random Forest model often yields the best accuracy, followed by Decision Trees and Logistic Regression. 

### Feature Importance

Understanding which features contribute most to survival can guide future decisions. We can visualize feature importance using:

```python
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
```

## Contributing

Contributions are welcome! If you want to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, feel free to reach out:

- GitHub: [gamy703](https://github.com/gamy703)
- Email: gamy703@example.com

Explore the Titanic dataset and enhance your machine learning skills! For updates, check the [Releases](https://github.com/gamy703/titanic_machine_learning/releases).
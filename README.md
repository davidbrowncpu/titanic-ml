# Titanic - Machine Learning from Disaster

My first Kaggle competition submission, built as I started learning machine learning fundamentals in Python.

## The Problem
Predict whether a passenger survived the Titanic disaster based on features like class, sex, age, and fare — the classic beginner ML classification task.

## Approach
- Cleaned missing data (Age, Fare) using median imputation
- Selected six features: Pclass, Sex, SibSp, Parch, Fare, Age
- Trained a Random Forest Classifier (scikit-learn)
- Generated predictions and submitted to the Kaggle leaderboard

## Result
0.78708

## Next Steps
- Engineer additional features (e.g. family size, title extracted from name)
- Try alternative models (Gradient Boosting, Logistic Regression) for comparison
- Apply similar workflow to a dataset more directly tied to my research interest in AI for underserved/low-resource regions

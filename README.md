# Titanic - Machine Learning from Disaster

My first Kaggle competition, built while learning machine learning fundamentals in Python. Documenting three iterations here on purpose — the middle one didn't work, and that's the most useful part of the story.

## The Problem
Predict whether a passenger survived the Titanic disaster using features like class, sex, age, and fare.

## Attempt 1 — Baseline (Score: 0.78708)
- Random Forest Classifier
- Features: Pclass, Sex, SibSp, Parch, Fare, Age
- Simple median imputation for missing values
- Result: a solid baseline score on my first-ever Kaggle submission

## Attempt 2 — "Improving" it (Score: 0.75598)
Tried to improve on the baseline by:
- Engineering new features: FamilySize, Title (extracted from passenger names)
- Adding Gradient Boosting and comparing it against Random Forest via cross-validation
- Cross-validation accuracy looked better (0.8418 vs 0.8305 for Random Forest)

**This scored worse on the actual leaderboard than my simpler first attempt.**

That was the real lesson: cross-validation accuracy and leaderboard accuracy aren't the same thing, especially on a small dataset (891 training rows). The Gradient Boosting model had learned patterns in my training folds that didn't generalize — classic overfitting.

## Attempt 3 — Fixing the overfit (Score: 0.78229)
Instead of abandoning the new features, I fixed the actual problem:
- Went back to Random Forest, but constrained it (`max_depth=4`, `min_samples_leaf=5`) so it couldn't memorize noise
- Dropped raw Fare (noisy, redundant with Pclass)
- Used Stratified K-Fold cross-validation for a more reliable estimate
- Kept FamilySize and Title, since the *features* weren't the problem — the model complexity was

Result: back above my baseline, and this time the improvement is trustworthy because I understand why it works.

## Takeaway
A better cross-validation score doesn't guarantee a better real-world result. I'd rather ship a model I understand than one that just tests well.

## Next Steps
- Try hyperparameter tuning (GridSearchCV) on the constrained Random Forest
- Apply the same "verify, don't just trust the metric" discipline to a dataset tied to my actual research interest: AI performance for underserved/low-resource regions

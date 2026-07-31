import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import kagglehub

path = kagglehub.competition_download('titanic')

train = pd.read_csv(f"{path}/train.csv")
test = pd.read_csv(f"{path}/test.csv")


for df in [train, test]:

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1


    df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)

    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
         "Rev", "Sir", "Jonkheer", "Dona"], "Rare"
    )
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")


    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["Title"] = df["Title"].map(
        {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
    ).fillna(4)

features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]

X_train = train[features]
y_train = train["Survived"]
X_test = test[features]

rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=1)

rf_score = cross_val_score(rf, X_train, y_train, cv=5).mean()
gb_score = cross_val_score(gb, X_train, y_train, cv=5).mean()

print(f"Random Forest CV accuracy: {rf_score:.4f}")
print(f"Gradient Boosting CV accuracy: {gb_score:.4f}")

best_model = gb if gb_score > rf_score else rf
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission.csv", index=False)
print("Saved improved submission.csv")
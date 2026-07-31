import kagglehub
kagglehub.login()
import kagglehub

path = kagglehub.competition_download('titanic')
print(path)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv(f"{path}/train.csv")
test = pd.read_csv(f"{path}/test.csv")
features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
for df in [train, test]:
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
X_train = train[features]
y_train = train["Survived"]
X_test = test[features]
model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
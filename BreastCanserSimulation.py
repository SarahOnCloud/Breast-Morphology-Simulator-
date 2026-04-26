#here is the final copy of the code:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset in from CSV
    # 1st line of the csv file is column names, all following are data entries. 
df = pd.read_csv("wdbc.csv")

# Define the database columns we want the model to view, it can ONLY see these
# columns. df stands for define. For y, we defined a mapping which converts the
# character value from "Diagnosis" into a number useable in calculations.
features = ["radius_mean", "texture_mean", "smoothness_mean", "concavity_mean"]
X = df[features]
y = df["Diagnosis"].map({"M": 1, "B": 0})

# Split the data into a testing set and a testing set.
  # 80% of the data is reserved to use in training the model while the remaining
  # 20% is keep to test the models functionallity. The random_state=42 keep this
  # split identical across multiple iterations of the code.
    # (Same data is used for testing no matter how many times we run script)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale each columns data to give each equal weight in the system
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construct a Logical Regession Model using our calculated data.
  # Each feature we're analysising is give some wieght in determining the 
  # probability that the tumor is malignant; Larger positive weight → increases 
  # malignancy probability, negative weight → decreases probability.
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

#***
# Check the Accurassy for the model by checking given only 4 variables, how often
# does the model correctly classifies a tumor.
print("Accuracy:", model.score(X_test_scaled, y_test))

# Check the model's coefficients. each coefficient tells us, If this variable 
# increases by 1 standard deviation, how much does it increase the odds of
# malignancy?
for feature, coef in zip(features, model.coef_[0]):
    print(feature, coef)

#everything after this is the plotting data you (jules) gave me
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Predict probabilities using scaled test data
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Compute AUC
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

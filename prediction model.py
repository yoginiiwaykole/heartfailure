import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv(r"C:\Users\yogin\Downloads\HEARTFAILUREDATASET_project.csv")

print(df.head())

# Select independent and dependent variable
X = df[["age","Sex","Depression","Smoking","trestbps","cholesterol","fasting blood sugar","thalach"]]
y = df["target"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
model = RandomForestClassifier()

# Fit the model
model.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(model, open("model.pkl", "wb"))
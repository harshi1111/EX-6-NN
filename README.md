<H3>NAME - HARSHITHA V</H3>
<H3>REG NO - 212223230074</H3>

<H1 ALIGN =CENTER>EX. NO.6 Heart attack prediction using MLP</H1>
<H3>Aim:</H3>  To construct a  Multi-Layer Perceptron to predict heart attack using Python
<H3>Algorithm:</H3>
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<BR>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>
Step 4:Split the dataset into training and testing sets using train_test_split().<BR>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<BR>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<BR>
Step 10:Print the accuracy of the model.<BR>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<BR>
<H3>Program: </H3>

```
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

try:
    data = pd.read_csv('heart.csv')
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {data.shape}")
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please upload the file or adjust the path.")
    exit()

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print(f"Features (X) shape: {X.shape}")
print(f"Labels (y) shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data normalized.")

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    activation='relu',
    solver='adam',
    random_state=42,
    tol=1e-4
)

print("Starting MLP training...")
mlp.fit(X_train, y_train)
print("MLP training finished.")

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print("------------------------")

plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_, color='dodgerblue', linewidth=2)
plt.title('MLP Training Error Convergence', fontsize=16)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

<H3>Output:</H3>

<img width="478" height="498" alt="image" src="https://github.com/user-attachments/assets/6d01382c-0f09-48fb-8b0b-3f38cafbcb15" />


<H3>Results:</H3>
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.

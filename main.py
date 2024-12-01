import json
import numpy as np
import pandas as pd
from collections import Counter
import logging


# Dataset Preparation
def load_train_data():
    with open("play_tennis_data.json", "r") as file:
        data = json.load(file)
    return pd.DataFrame(data)


def load_test_data():
    with open("test_data.json", "r") as file:
        test_data = json.load(file)
    return pd.DataFrame(test_data)


def encode_data(df, reference_columns=None):
    # Exclude the 'Day' and 'PlayTennis' columns as they are not relevant for the model
    df = df.drop(columns=["Day", "PlayTennis"], errors="ignore")

    # Perform one-hot encoding for categorical features
    encoded_features = pd.get_dummies(df, columns=["Outlook", "Temperature", "Humidity", "Wind"], drop_first=True)
    # Convert all feature columns to float
    encoded_features = encoded_features.astype(float)

    # Ensure the columns of the test data match the training data columns
    if reference_columns is not None:
        # Align the columns of the test data with the reference (training) columns
        encoded_features = encoded_features.reindex(columns=reference_columns, fill_value=0)

    # Drop the target column from the encoded test instance if it's present
    encoded_features = encoded_features.drop(columns=["PlayTennis"], errors="ignore")

    return encoded_features


def encode_df(df):
    # Exclude the 'Day' column as it is not relevant for the model
    df = df.drop(columns=["Day"], errors="ignore")

    # Exclude the target column during encoding
    features = df.drop(columns=["PlayTennis"])
    encoded_features = pd.get_dummies(features, columns=["Outlook", "Temperature", "Humidity", "Wind"], drop_first=True)
    # Convert all feature columns to float
    encoded_features = encoded_features.astype(float)
    # Reattach the target column
    encoded_features["PlayTennis"] = df["PlayTennis"]
    return encoded_features


# Distance Calculation
def calculate_distance(instance1, instance2, metric="euclidean"):
    instance1 = np.array(instance1)  # Convert to numpy array
    instance2 = np.array(instance2)  # Convert to numpy array

    if metric == "euclidean":
        return np.sqrt(np.sum((instance1 - instance2) ** 2))
    elif metric == "manhattan":
        return np.sum(np.abs(instance1 - instance2))


# k-NN Classifier
class KNearestNeighbors:
    def __init__(self, k, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.data = None

    def train(self, data):
        self.data = data
        # Persist the data for lazy learning
        self.data.to_json("knn_model.json", orient="records")

    def predict(self, instance):
        # Flatten the test instance into a Series to match the training data
        instance = instance.values.flatten()  # Convert to a 1D numpy array
        distances = []

        # Ensure the instance matches the training data shape
        if len(instance) != len(self.data.columns) - 1:  # Excluding "PlayTennis"
            print(f"Shape mismatch: instance ({len(instance)}) and data ({len(self.data.columns) - 1})")
            return None  # Exit early if the shapes don't match

        # Calculate distance for each instance
        for _, row in self.data.iterrows():
            # Drop the target column and flatten the row values to match the instance's shape
            row_values = row.drop("PlayTennis").values.flatten()  # Flatten the row
            dist = calculate_distance(instance, row_values, self.metric)
            distances.append((dist, row["PlayTennis"]))

        # Sort by distance and get top-k
        distances.sort(key=lambda x: x[0])

        print(f"Distances: {distances[:self.k]}")  # Debugging distances and neighbors
        neighbors = [label for _, label in distances[:self.k]]

        # Majority vote
        prediction = Counter(neighbors).most_common(1)[0][0]
        print(f"Predicted class: {prediction}")  # Debugging the predicted class
        return prediction

    def evaluate(self, data, log_file):
        # Confusion Matrix initialization
        tp = fp = tn = fn = 0
        correct = 0
        predictions = []

        for i in range(len(data)):
            test_instance = data.iloc[i].drop("PlayTennis")
            actual = data.iloc[i]["PlayTennis"]
            predicted = self.predict(test_instance)
            predictions.append((actual, predicted))

            if actual == predicted:
                correct += 1
                if actual == "Yes":
                    tp += 1  # True Positive
                else:
                    tn += 1  # True Negative
            else:
                if actual == "Yes":
                    fn += 1  # False Negative
                else:
                    fp += 1  # False Positive

            # Log the actual and predicted values
            log_file.write(f"Actual: {actual}, Predicted: {predicted}\n")

        accuracy = correct / len(data)

        # Confusion Matrix Output
        log_file.write(f"\nConfusion Matrix:\n")
        log_file.write(f"True Positives (TP): {tp}\n")
        log_file.write(f"False Positives (FP): {fp}\n")
        log_file.write(f"True Negatives (TN): {tn}\n")
        log_file.write(f"False Negatives (FN): {fn}\n")
        log_file.write("------------------------\n")


        return accuracy, predictions


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(filename="classification_log.txt", level=logging.INFO)

    # Get user input for k and distance metric
    k = int(input("Enter the number of nearest neighbors (k): "))
    metric = input("Choose the distance metric (euclidean/manhattan): ").strip().lower()

    # Ensure valid metric input
    while metric not in ["euclidean", "manhattan"]:
        print("Invalid input! Please choose either 'euclidean' or 'manhattan'.")
        metric = input("Choose the distance metric (euclidean/manhattan): ").strip().lower()

    train_data = load_train_data()
    test_data = load_test_data()
    encoded_train_data = encode_df(train_data)
    encoded_test_data = encode_df(test_data)

    knn = KNearestNeighbors(k=k, metric=metric)

    # Train the model
    knn.train(encoded_train_data)

    if not test_data.empty:
        print("-----------Testing-----------")
        # Open log file and evaluate on test data
        with open("classification_log.txt", "a") as log_file:
            accuracy, predictions = knn.evaluate(encoded_test_data, log_file)
            print(f"Accuracy on test data: {accuracy}")
    else:
        print("No test data")

    # Test a single instance
    test_instance = {
        "Outlook": "Rain",
        "Temperature": "Cool",
        "Humidity": "Normal",
        "Wind": "Strong"
    }
    test_instance = pd.DataFrame([test_instance])  # Wrap it in a list
    encoded_test_instance = encode_data(test_instance, reference_columns=encoded_train_data.columns)
    print(f"---------Test Instance---------\n")
    prediction = knn.predict(encoded_test_instance)
    if prediction is not None:
        print(f"Predicted class for the test instance: {prediction}")
    else:
        print("Prediction could not be made due to shape mismatch.")

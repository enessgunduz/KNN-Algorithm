# K-Nearest Neighbors (KNN) Classifier

This project implements a K-Nearest Neighbors (KNN) classifier with options for **Euclidean** or **Manhattan** distance metrics. It includes functionality for training the model on a provided dataset, evaluating the model on a test dataset, and generating confusion matrix metrics (True Positives, False Positives, True Negatives, False Negatives).

## Features:
- K-Nearest Neighbors (KNN) classifier with user-defined `k` and distance metric.
- Supports **Euclidean** and **Manhattan** distance metrics.
- Evaluation on test data with accuracy and confusion matrix calculation.
- Logs classification results and confusion matrix to a log file.
- Can predict labels for single instances based on the trained model.

## Requirements:

Before running the code, ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `json`
- `logging`

## Dataset:
Training Data:

You need to prepare a JSON file (play_tennis_data.json) for training data that contains information on weather conditions and whether people play tennis under those conditions.
Test Data:

Similarly, provide another JSON file (test_data.json) with test instances for evaluation.

## How to Run the Code:
### Step 1: Prepare the Data
Ensure that your training and test data are saved as play_tennis_data.json and test_data.json, respectively, in the same directory as the Python script.
### Step 2: Choose the Model Parameters
When running the code, you'll be prompted to input the following parameters:

k: The number of nearest neighbors to consider for classification.
distance metric: Choose either euclidean or manhattan for the distance metric.
### Step 3: Run the Code
After preparing the data and entering the required parameters, simply run the script:
```
python main.py
```

The script will:

Train the model on the training dataset.
Evaluate it on the test dataset and print the accuracy.
Log the confusion matrix results in classification_log.txt.
### Step 4: View Results
The accuracy of the model on the test dataset will be displayed in the terminal.
The confusion matrix (True Positives, False Positives, True Negatives, False Negatives) will be written to the classification_log.txt file.

Example:
```
Enter the number of nearest neighbors (k): 3
Choose the distance metric (euclidean/manhattan): euclidean
```

The log file classification_log.txt will contain an output like:
```
Actual: Yes, Predicted: Yes
Actual: No, Predicted: No
Actual: Yes, Predicted: No

Confusion Matrix:
True Positives (TP): 2
False Positives (FP): 1
True Negatives (TN): 3
False Negatives (FN): 0
```

## Test a Single Instance:
After evaluating the test data, the model will predict a class for a sample test instance provided in the code. You can modify the instance's values to test other cases.

Logging:
The script logs the classification results (actual vs. predicted) for each test instance and confusion matrix values to a file called classification_log.txt.
## Troubleshooting:
Shape Mismatch: If the test instance does not match the shape of the training data, the prediction will fail. Ensure that the instance data contains the same features as the training data.

Invalid Metric: If the user provides an invalid distance metric, the program will prompt the user to choose either euclidean or manhattan.




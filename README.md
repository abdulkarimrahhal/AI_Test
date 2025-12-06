md
# AI_Test

This repository contains a simple AI project, likely focused on the Iris dataset, using Python.

## Key Features & Benefits

*   Utilizes the Iris dataset for classification.
*   Implements a Random Forest Classifier for prediction.
*   Includes data preprocessing steps for handling missing values and scaling.
*   Provides basic evaluation metrics like accuracy score and confusion matrix.

## Prerequisites & Dependencies

Before running the code, ensure you have the following installed:

*   **Python 3.6+**
*   **Required Python Libraries:**
    *   pandas
    *   scikit-learn (sklearn)
    *   matplotlib
    *   seaborn

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/abdulkarimrahhal/AI_Test.git
    cd AI_Test
    ```

2.  **Ensure all dependencies are installed (see Prerequisites & Dependencies).**

3.  **The `IrisDataset.csv` file should be in the same directory as `Iris_ai.py`.**

## Usage Examples & Code Explanation

To run the Iris classification model, execute the `Iris_ai.py` script:

```bash
python Iris_ai.py
```

**Code Overview (`Iris_ai.py`):**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris_df = pd.read_csv("IrisDataset.csv")
print(iris_df.head())
print()

# معالجة القيم الفارغة
print(iris_df.info())
for attr in iris_df.columns:
    if not iris_df[attr].isna().sum() == 0:
        iris_df[attr...
```

The script performs the following steps:

1.  **Data Loading:** Loads the Iris dataset from `IrisDataset.csv` using pandas.
2.  **Data Preprocessing:** Handles missing values (the provided snippet is incomplete, but would typically fill NaN values) and scales the features using `StandardScaler`.
3.  **Data Splitting:** Splits the data into training and testing sets using `train_test_split`.
4.  **Model Training:** Trains a Random Forest Classifier (`RandomForestClassifier`) on the training data.
5.  **Model Evaluation:** Predicts on the test set and calculates the accuracy score and confusion matrix using `accuracy_score` and `confusion_matrix`.
6.  **Visualization:**  The snippet does not fully show the visualization implementation, but is expected to generate visualizations using matplotlib and seaborn.

## Configuration Options

Currently, the script doesn't offer extensive configuration options. However, you can modify the following parameters directly within the `Iris_ai.py` file:

*   **`test_size` in `train_test_split`:**  Adjust the size of the test set.
*   **Parameters of `RandomForestClassifier`:**  Experiment with different parameters for the Random Forest model, such as `n_estimators` (number of trees), `max_depth`, `min_samples_split`, etc.

## Contributing Guidelines

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request.

## License Information

License not specified. All rights reserved by the repository owner.

## Acknowledgments

*   The Iris dataset is a widely used dataset in machine learning and is publicly available.
*   This project utilizes popular Python libraries such as pandas, scikit-learn, matplotlib, and seaborn.

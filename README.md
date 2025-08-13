# Predicting Churn Risk via In-App Behavior Clustering for Enhanced User Retention

## Overview

This project analyzes in-app user behavior data to identify distinct user segments and predict churn risk.  The analysis employs unsupervised learning techniques (clustering) to group users with similar behavior patterns.  A predictive model is then developed to classify users as high or low churn risk, enabling proactive interventions to improve user retention. This allows for targeted retention strategies based on specific user segments and their individual needs.

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## How to Run

1. **Install Dependencies:**  Ensure you have Python 3.x installed. Navigate to the project directory in your terminal and install the required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script:** Execute the main script using:

   ```bash
   python main.py
   ```

## Example Output

The script will print key findings of the analysis to the console, including details about the identified user clusters and the performance metrics of the churn prediction model.  Additionally, the script generates several visualizations, including:

* **Cluster visualizations:** Plots illustrating the distribution of users across different clusters based on their in-app behavior.  These will be saved as PNG files (e.g., `cluster_visualization_1.png`, `cluster_visualization_2.png`).
* **Model performance plots:**  Plots showing the performance of the churn prediction model (e.g., ROC curve, precision-recall curve). These will also be saved as PNG files (e.g., `roc_curve.png`, `precision_recall_curve.png`).


The specific output files and their names might vary slightly depending on the data and the model used.  Consult the code for details on the generated output.
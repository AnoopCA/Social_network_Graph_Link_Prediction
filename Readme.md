Social Network Graph Link Prediction - Facebook Challenge:
This GitHub repository contains code and analysis for the Social Network Graph Link Prediction project, specifically focused on the Facebook Challenge. The goal of this project is to predict whether a link between two nodes (users) will be established in a social network graph. The project utilizes features derived from preferential attachment and singular value decomposition (SVD), and employs the XGBoost algorithm for modeling and prediction.

Dependencies:
To run the code in this repository, we need the following dependencies:
networkx
numpy
pandas
seaborn
matplotlib
scipy
xgboost
sklearn
prettytable
These dependencies can be installed using pip or conda.

Data Pre-processing:
Importing Libraries
The necessary Python libraries are imported at the beginning of the code.

Importing Data
Data from the Facebook Challenge is imported, including both training and testing data, stored in HDF files.

Feature Generation
Two types of features are generated: Preferential Attachment and SVD Dot features. These features are calculated for both training and testing data and saved to HDF files for further use.

Train-Test Split
The data is split into training and testing sets for model evaluation using the train_test_split function from sklearn.

Modeling and Hyperparameter Tuning
XGBoost Classifier
The XGBoost algorithm is employed for this classification problem. The XGBoost classifier is set up with the objective of binary logistic regression.

Hyperparameter Tuning
Hyperparameter tuning is performed using Randomized Search Cross-Validation. The model is trained and evaluated for different combinations of hyperparameters.

Best Model Selection
The best-performing model is selected based on the cross-validation results. Model parameters are saved to a pickle file for future use.

Performance Analysis
Cross Validation Results
Cross-validation results are analyzed, showing the impact of different hyperparameters on the model's performance. The best-performing parameter combination is highlighted.

F1 Score
F1 scores are calculated for both the training and testing sets to evaluate model performance.

Confusion Matrix
Confusion matrices are generated for the training and testing sets. These matrices provide insights into the model's predictions.

ROC Curve
The Receiver Operating Characteristic (ROC) curve is plotted to visualize the true positive rate against the false positive rate for different thresholds.

Conclusion:
The project concludes with an analysis of the improved model's performance. The F1 score has increased significantly, indicating better predictive accuracy in link prediction within the social network graph. The repository contains code, analysis, and saved model parameters for reference and further exploration.
# Health-Claims-Forecasting-with-Decision-Trees

This project focuses on building a machine learning model in **R** to predict healthcare reimbursement amounts using structured insurance claim data. The goal is to estimate the log-transformed reimbursement for the year 2010 (`log10_reimb2010`) based on features provided in a historical dataset.

---

## 📁 Repository Contents

```text
├── 2025claimsforecast_v2.R      # Main R script: training and saving the prediction model
├── treeFinal.DataR              # Saved R model object to be used for evaluation
├── README.md                    # Project documentation (this file)


🎯 Project Objectives
	1.	Train a predictive model using historical claims data.
	2.	Save the trained model in R’s .DataR format for deployment and grading.
	3.	Evaluate performance using Out-of-Sample R-Squared (OSR²) on test data.
	4.	Ensure reproducibility for automated assessment.

⸻

🔍 Prediction Target
	•	log10_reimb2010: The base-10 logarithm of the total reimbursement amount in 2010.

⸻

🧠 Methodology Overview
	•	Data preprocessing and feature transformation (not shown here due to dataset privacy).
	•	Training a regression decision tree model using R.
	•	Saving the final model object as treeFinal.DataR using save().
	•	Preparing the code for compatibility with an automated evaluation script provided by the course.

⸻

📊 Evaluation Metric
	•	OSR² (Out-of-Sample R-Squared)
A measure of model accuracy on test data based on deviation from the training mean.

⚙️ Tools Used
	•	Language: R
	•	Modeling: Decision Tree Regression (rpart package assumed)
	•	Serialization: save() and .DataR format

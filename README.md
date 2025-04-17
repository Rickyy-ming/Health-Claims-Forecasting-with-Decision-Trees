# Health-Claims-Forecasting-with-Decision-Trees
This project aims to build and deploy a predictive model for healthcare reimbursement claims, specifically forecasting the log-transformed reimbursement amounts in 2010 (log10_reimb2010) using structured claims data.

This work was developed as part of the final course project in the 2025 Management Analytics curriculum. It demonstrates end-to-end model training, evaluation, and saving in R using decision tree-based techniques

📁 Project Structure
├── 2025claimsforecast_v2.R      # Main R script: training, evaluation, and model saving
├── treeFinal.DataR              # Serialized R model object
├── ClaimsTrain.csv              # (not included here) Training dataset (provided externally)
├── README.md                    # Project documentation


🚀 Project Goals
	1.	Model Training
	•	Build a regression model to predict log10_reimb2010 using features in the ClaimsTrain.csv dataset.
	•	Select and justify the modeling strategy based on performance and interpretability.
	2.	Model Evaluation
	•	Evaluate out-of-sample performance using OSR-squared as the main metric.
	3.	Model Persistence
	•	Save the trained model to a .DataR file (treeFinal.DataR) for downstream evaluation and reuse.

 📌 Key Features
	•	📊 Decision tree modeling for transparency and explainability.
	•	🧪 Test script compatibility – your code is fully compatible with automated testing frameworks (as per the course instructions).
	•	⚙️ Serialization with save() to allow quick loading and evaluation of the trained model.

⸻

🧠 Methodology

The approach includes:
	•	Preprocessing of features and appropriate transformations.
	•	Model selection using regression tree methods.
	•	OSR-squared calculation on test data for validation.
	•	Saving the model using save(model, file = "treeFinal.DataR").

⸻

📈 Evaluation Criteria

This project was graded on:
	•	✅ Executability of the saved model
	•	📉 Predictive performance (OSR²)
	•	🧠 Innovation and soundness of modeling strategy

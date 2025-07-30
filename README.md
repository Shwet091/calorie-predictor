 🏃‍♂️ Calorie Burn Prediction Using Machine Learning

This project predicts the number of calories burned during physical activity using various machine learning models such as XGBoost, Random Forest, Linear Regression, Lasso, and Ridge Regression.

---

📊 Project Overview

Goal: Predict calorie burn based on user attributes like age, gender, weight, height, and physical activity data.
Best Model: XGBoost showed the best performance with high prediction accuracy.
Tools Used:

    Python, Pandas, NumPy, Matplotlib, Seaborn
    Scikit-learn, XGBoost
    Streamlit for UI

---

📁 Folder Structure

```
calorie_proct/
├── app.py                       # Streamlit web app
├── Calorie_pred.ipynb           # Jupyter notebook with model building
├── requirements.txt             # Required Python libraries
├── .gitignore                   # Files excluded from GitHub
├── calories.csv                 # Dataset file
├── calories_model.pkl           # Trained model
├── scaling.pkl                  # Scaler object
├── Animation - 1741535941522.json  # Animation JSON for UI
└── Animation - 1741536478823.json  # Animation JSON for UI
```

---

🚀 How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/calorie-burn-prediction.git
   cd calorie-burn-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Note: The required `.pkl` model files and dataset (`calories.csv`) are already included in this repository.

---

🧠 Models Used

 Linear Regression
 Ridge and Lasso Regression
 Random Forest Regressor
 XGBoost Regressor *(Best performing)*

---

📈 Evaluation Metrics

 Mean Absolute Error (MAE)
 Mean Squared Error (MSE)
 R² Score

---

 ✅ Results

 XGBoost outperformed other models with the lowest MAE and highest R² Score.
 Real-time prediction app built using Streamlit.

---

🔒 Important Notes

 Sensitive files like `secrets.toml` are excluded via `.gitignore`.
 Dataset and model files (`.pkl`) are intentionally included to allow full functionality of the app.

---

 ✍️ Author

Shwet Bhagat
M.Sc. Data Science, Fergusson College
GitHub: Shwet091

---

🌟 Acknowledgments

 Inspired by health data projects and machine learning regression problems.
 Thanks to open datasets and scikit-learn/XGBoost libraries.

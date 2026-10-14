# SafeClaim — Insurance Claim Fraud Detection

A Flask web app for exploring vehicle insurance claim classification with a saved XGBoost model. This is a course project and its predictions are for educational demonstration.

## Run locally

1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Start the app: `python app.py`
5. Open `http://127.0.0.1:5000` in your browser.

The model and CSV dataset are kept in the project root because the app loads them at startup.

## Understanding the result

- **Model classification and probability** come from the saved XGBoost model. The displayed review bands are low below 30%, review from 30% to under 70%, and high at 70% or above.
- **Demo rule summary** is a separate teaching aid: two or more high-severity indicators produce “High review priority”; one high or at least two medium indicators produce “Review suggested”; otherwise it shows “Routine review.” This rule does not change the model prediction and is not proof of fraud.

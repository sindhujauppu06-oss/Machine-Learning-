from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
from datetime import datetime
import random

app = Flask(__name__)

# =========================================================
# LOAD MODEL
# =========================================================

model = xgb.XGBClassifier()
model.load_model("insurance_fraud_model.json")


# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv("fraud_oracle.csv")

# Remove target column to get training feature columns
train_columns = data.drop("FraudFound_P", axis=1).columns.tolist()


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# CLAIM FORM
# =========================================================

@app.route("/claim")
def claim():
    return render_template("claim_form.html")


# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

@app.route("/performance")
def performance():
    return render_template("performance.html")


# =========================================================
# ABOUT PAGE
# =========================================================

@app.route("/about")
def about():
    return render_template("about.html")


# =========================================================
# CREATE CLAIM ID
# =========================================================

def generate_claim_id():

    year = datetime.now().year

    number = random.randint(10000, 99999)

    return f"SC-{year}-{number}"


# =========================================================
# GET RISK LEVEL
# =========================================================

def get_risk_level(probability):

    if probability < 30:
        return "LOW RISK"

    elif probability < 70:
        return "REVIEW REQUIRED"

    else:
        return "HIGH RISK"


# =========================================================
# GENERATE INVESTIGATION FINDINGS
# =========================================================

def generate_findings(
    police,
    witness,
    past_claims,
    supplements,
    fault,
    age_vehicle,
    vehicle_price
):

    findings = []

    # -----------------------------------------------------
    # Police report
    # -----------------------------------------------------

    if police.lower() == "no":

        findings.append({
            "title": "Police Report",
            "description": "Police report has not been filed.",
            "level": "HIGH"
        })

    else:

        findings.append({
            "title": "Police Report",
            "description": "Police report is available.",
            "level": "LOW"
        })


    # -----------------------------------------------------
    # Witness
    # -----------------------------------------------------

    if witness.lower() == "no":

        findings.append({
            "title": "Witness Information",
            "description": "No witness has been reported for the accident.",
            "level": "MEDIUM"
        })

    else:

        findings.append({
            "title": "Witness Information",
            "description": "A witness has been reported.",
            "level": "LOW"
        })


    # -----------------------------------------------------
    # Previous claims
    # -----------------------------------------------------

    if past_claims.lower() in ["2 to 4", "more than 4"]:

        findings.append({
            "title": "Previous Claims",
            "description": "Multiple previous claims are associated with this policy.",
            "level": "HIGH"
        })

    elif past_claims.lower() == "1":

        findings.append({
            "title": "Previous Claims",
            "description": "One previous claim is associated with this policy.",
            "level": "MEDIUM"
        })

    else:

        findings.append({
            "title": "Previous Claims",
            "description": "No significant previous claim history detected.",
            "level": "LOW"
        })


    # -----------------------------------------------------
    # Supplements
    # -----------------------------------------------------

    if supplements.lower() in ["2 to 4", "more than 4"]:

        findings.append({
            "title": "Claim Supplements",
            "description": "Multiple supplements have been associated with the claim.",
            "level": "HIGH"
        })

    else:

        findings.append({
            "title": "Claim Supplements",
            "description": "No significant number of supplements detected.",
            "level": "LOW"
        })


    # -----------------------------------------------------
    # Fault
    # -----------------------------------------------------

    if fault.lower() == "policy holder":

        findings.append({
            "title": "Accident Fault",
            "description": "The policy holder is recorded as being at fault.",
            "level": "MEDIUM"
        })

    else:

        findings.append({
            "title": "Accident Fault",
            "description": "The policy holder is not recorded as being at fault.",
            "level": "LOW"
        })


    # -----------------------------------------------------
    # Vehicle age
    # -----------------------------------------------------

    if age_vehicle.lower() in ["7 years", "more than 7 years"]:

        findings.append({
            "title": "Vehicle Age",
            "description": "The vehicle has relatively high age.",
            "level": "MEDIUM"
        })

    else:

        findings.append({
            "title": "Vehicle Age",
            "description": "Vehicle age is within the normal range.",
            "level": "LOW"
        })


    return findings


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------------------------------
    # GET FORM DATA
    # -----------------------------------------------------

    age = int(request.form["Age"])

    gender = request.form["Gender"]

    marital = request.form["MaritalStatus"]

    make = request.form["Make"]

    vehicle_price = request.form["VehiclePrice"]

    age_vehicle = request.form["AgeOfVehicle"]

    month = request.form["Month"]

    area = request.form["AccidentArea"]

    fault = request.form["Fault"]

    police = request.form["PoliceReportFiled"]

    witness = request.form["WitnessPresent"]

    agent = request.form["AgentType"]

    past_claims = request.form["PastNumberOfClaims"]

    supplements = request.form["NumberOfSuppliments"]

    # Match the age bands used by the source dataset.
    if age <= 17:
        age_policy_holder = "16 to 17"
    elif age <= 20:
        age_policy_holder = "18 to 20"
    elif age <= 25:
        age_policy_holder = "21 to 25"
    elif age <= 30:
        age_policy_holder = "26 to 30"
    elif age <= 35:
        age_policy_holder = "31 to 35"
    elif age <= 40:
        age_policy_holder = "36 to 40"
    elif age <= 50:
        age_policy_holder = "41 to 50"
    elif age <= 65:
        age_policy_holder = "51 to 65"
    else:
        age_policy_holder = "over 65"


    # -----------------------------------------------------
    # CREATE CLAIM ID
    # -----------------------------------------------------

    claim_id = generate_claim_id()


    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    input_data = {

        "Month": month,

        "Make": make,

        "AccidentArea": area,

        "Sex": gender,

        "MaritalStatus": marital,

        "Age": age,

        "Fault": fault,

        "VehiclePrice": vehicle_price,

        "AgeOfVehicle": age_vehicle,

        "AgeOfPolicyHolder": age_policy_holder,

        "PoliceReportFiled": police,

        "WitnessPresent": witness,

        "AgentType": agent,

        "PastNumberOfClaims": past_claims,

        "NumberOfSuppliments": supplements

    }


    input_df = pd.DataFrame([input_data])


    # =====================================================
    # IMPORTANT:
    # USE DATASET CATEGORY MAPPINGS
    # =====================================================

    for column in train_columns:

        if column not in input_df.columns:

            input_df[column] = 0

            continue


        # -------------------------------------------------
        # CATEGORICAL COLUMNS
        # -------------------------------------------------

        if data[column].dtype == "object":

            categories = data[column].astype("category").cat.categories

            category_mapping = {
                category: index
                for index, category in enumerate(categories)
            }

            input_df[column] = input_df[column].map(
                category_mapping
            )


        # -------------------------------------------------
        # NUMERIC COLUMNS
        # -------------------------------------------------

        else:

            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="coerce"
            )


    # -----------------------------------------------------
    # HANDLE UNKNOWN VALUES
    # -----------------------------------------------------

    input_df = input_df.fillna(0)


    # -----------------------------------------------------
    # MATCH TRAINING COLUMN ORDER
    # -----------------------------------------------------

    input_df = input_df[train_columns]


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = int(model.predict(input_df)[0])


    # -----------------------------------------------------
    # FRAUD PROBABILITY
    # -----------------------------------------------------

    probabilities = model.predict_proba(input_df)[0]

    fraud_probability = float(probabilities[1]) * 100


    # =====================================================
    # RESULT
    # =====================================================

    if prediction == 1:

        result = "Fraudulent Claim"

    else:

        result = "Genuine Claim"


    # =====================================================
    # RISK LEVEL
    # =====================================================

    risk_level = get_risk_level(fraud_probability)


    # =====================================================
    # INVESTIGATION FINDINGS
    # =====================================================

    findings = generate_findings(
        police,
        witness,
        past_claims,
        supplements,
        fault,
        age_vehicle,
        vehicle_price
    )

    # Keep this educational rule summary separate from the model score.
    high_findings = sum(1 for finding in findings if finding["level"] == "HIGH")
    medium_findings = sum(1 for finding in findings if finding["level"] == "MEDIUM")
    if high_findings >= 2:
        rule_review_priority = "HIGH REVIEW PRIORITY"
        rule_review_reason = f"{high_findings} high-severity indicators matched the demo review rules."
    elif high_findings == 1 or medium_findings >= 2:
        rule_review_priority = "REVIEW SUGGESTED"
        rule_review_reason = "One high-severity or multiple medium-severity indicators matched the demo rules."
    else:
        rule_review_priority = "ROUTINE REVIEW"
        rule_review_reason = "Few high-severity indicators matched the demo rules."


    # =====================================================
    # INVESTIGATION DECISION
    # =====================================================

    if fraud_probability >= 70:

        investigation_decision = "Manual Review Recommended"

    elif fraud_probability >= 30:

        investigation_decision = "Additional Verification Required"

    else:

        investigation_decision = "No Immediate Fraud Indicator"


    # =====================================================
    # SEND DATA TO RESULT PAGE
    # =====================================================

    return render_template(

        "result.html",

        claim_id=claim_id,

        prediction=result,

        confidence=round(fraud_probability, 2),

        risk_level=risk_level,

        investigation_decision=investigation_decision,

        findings=findings,

        rule_review_priority=rule_review_priority,

        rule_review_reason=rule_review_reason,

        make=make,

        area=area,

        price=vehicle_price,

        police=police,

        witness=witness,

        claims=past_claims,

        age=age,

        gender=gender,

        marital=marital,

        vehicle_age=age_vehicle,

        month=month,

        fault=fault,

        agent=agent,

        supplements=supplements

    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)

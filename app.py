from flask import Flask, request,jsonify,render_template
import joblib

app=Flask(__name__)

model = joblib.load("loan_model.pkl")


@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict',methods=['POST'])
def predict():
   # Collect all four inputs from the form
    age = float(request.form['age'])
    income = float(request.form['income'])
    loan_amount = float(request.form['loanamount'])
    credit_score = float(request.form['creditscore'])

    # Pass them as a single row to the model
    input_data = [[age, income, loan_amount, credit_score]]
    prediction = model.predict(input_data)

    # Show result on the webpage
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    return render_template('index.html', prediction_text=f'Loan Status: {result}')
if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)


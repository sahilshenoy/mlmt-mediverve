from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np
app = Flask(__name__, template_folder='templates')

model_path = 'heart_model.pkl'  # Ensure this path is correct
loaded_model = joblib.load(model_path)

@app.route("/")
def home():
    return redirect(url_for('heart'))

@app.route("/heart")
def heart():
    return render_template("heart.html")

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        result = loaded_model.predict(to_predict)
        return result[0]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list)==7):
            result = ValuePredictor(to_predict_list,7)

    if(int(result)==1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 5000, debug=True)
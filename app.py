from flask import Flask, jsonify, request, render_template
import json

import pred as pred

app = Flask(__name__)

# This function is for website 
@app.route("/", methods = ["GET","POST"])
def index():

    output = ""
    if request.method == "POST":
 
        Fcatgeory = request.form["category"]
        Ftype = request.form["type"]
        Fyear = request.form["year"]
        Fmonth = request.form["month"]

        output = pred.prediction(Fcatgeory, Ftype, Fyear, Fmonth)     

    return render_template("index.html", value = output)


# This is for model predict result
@app.route("/predict", methods = ["POST"])
def predict():

    output = ""
    if request.method == "POST":
        
        Fcatgeory = request.json["category"]
        Ftype = request.json["type"]
        Fyear = request.json["year"]
        Fmonth = request.json["month"]

        output = pred.prediction(Fcatgeory, Ftype, Fyear, Fmonth)
        output = json.dumps(output.tolist()[0])

    return jsonify({"prediction": output})

if __name__ == "__main__":
    app.run(debug = True)
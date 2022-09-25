from flask import Flask, render_template, request
import numpy as np
import pickle
import datetime

app = Flask(__name__)
model_Linear_R = pickle.load(open("models/LinearRegression.pkl", "rb"))
model_Random_F = pickle.load(open("models/RandomForest.pkl", "rb"))

# / route
@app.route("/")
def index():
    return render_template("index.html")

# def date_validate(year, month, day):

#     dateString = (year+month+day).str()
#     date_format = '%Y-%m-%d'
#     try:
#         date_obj = datetime.datetime.strptime(dateString, date_format)
#         print(date_obj)
#     except ValueError:
#         print("Incorrect data format, should be YYYY-MM-DD")

# def date_validate(year, month, day):
#     try: 
#         datetime.datetime(
#             year = int(year),
#             month = int(month),
#             day = int(day)
#         )
#     except: 
#         return render_template("index.html", date = "incorrect date")


# /predict route
@app.route("/predict", methods=["GET", "POST"])
def predict():
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]


    # inputDate = input("Enter the date in format 'dd/mm/yy' : ")

    # day, month, year = inputDate.split('/')

    # isValidDate = True
    # try:
    #     datetime.datetime(int(year), int(month), int(day))
    # except ValueError:
    #     isValidDate = False

    # if(isValidDate):
    #     print("Input date is valid ..")
    # else:
    #     print("Input date is not valid..")
    

    # import csv
    # with open('X:\data.csv','rt')as f:
    #     data = csv.reader(f)
    #     for data in data:
    #         if(val1 = data.day and val2 = data.month and val3 = data.year):
    #             price = data.price
    #     return price

    arr = np.array([year, month, day])
    arr = arr.astype(np.float64)

    methods = request.form['method']
    if methods == "LR":
        prediction = model_Linear_R.predict([arr])
    else:
        prediction = model_Random_F.predict([arr])
    return render_template("index.html", data=int(prediction))    



    
        # y = arr.fit_transform(y.reshape(-1, 1))
    # output = 'Gold Price for the given date is: {} Dollars'
    # output = 'Gold Price for the given date is: {} Dollars'.format(prediction)
    #     # y = arr.fit_transform(y.reshape(-1, 1))
    #     output = 'Gold Price for the given date is: {} Dollars'.format(prediction)
    
    # return render_template("index.html", prediction_text=output, data=int(output))    


# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     val1 = request.form["price"]
#     type = request.form["type"]
   
#     arr = np.array([val1])
#     arr = arr.astype(np.float64)
#     if type == "Linear Regression":
#         prediction = model_Linear_R.predict(arr)
#     else:
#         prediction = model_Random_F.predict(arr)

#     return render_template("index.html", prediction_text="Predicted Price: ${}".format(prediction))

#     pred = model.predict([arr])
#     return render_template("index.html", data=int(pred))

    

# @app.route('/prediction', methods=['POST'])
# def prediction():
#     option = [[int(x) for x in request.form.values()]]
#     if option[0][0] == 2:
#         del option[0][0]
#         x_test = option
#         print(x_test)
#         payload_scoring = {"input_data": [{"field": [["year", "month", "day"]], "values": x_test}]}

#         # response_scoring = requests.post(
#         #     'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/efde5787-7fe6-44ed-b6ae-7602c8004740/predictions?version=2021-08-03',
#         #     json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
#         # print("Scoring response")
#         # prediction_ = response_scoring.json()
#         pred = prediction_['predictions'][0]['values'][0][0]
#         output = 'Gas Price for the given date is: {} Dollars'.format(pred)
#     else:
#         del option[0][0]
#         x_test = option
#         print(x_test)
#         payload_scoring = {"input_data": [{"field": [["year", "month", "day"]], "values": x_test}]}

#         # response_scoring = requests.post(
#         #     'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/3ff71c4f-5d42-4b2b-8089-de3380e43850/predictions?version=2021-08-03',
#         #     json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
#         # print("Scoring response")
#         # prediction_ = response_scoring.json()
        
#         pred = prediction_['predictions'][0]['values'][0][0]
#         output = 'Gas Price for the given date is: {} Dollars'.format(pred)
#     return render_template('index.html', predic_text=output)  





# 
# /// model evaluation ///


# from sklearn.datasets import load_breast_cancer
# X, y = data = load_breast_cancer(return_X_y=True)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8675309)
# 
# def run_exps(X_train: pd.DataFrame , y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
#     '''
#     Lightweight script to test many models and find winners
# :param X_train: training split
#     :param y_train: training target vector
#     :param X_test: test split
#     :param y_test: test target vector
#     :return: DataFrame of predictions
#     '''
    
#     dfs = []
# models = [
#           ('LogReg', LogisticRegression()), 
#           ('RF', RandomForestClassifier()),
#           ('KNN', KNeighborsClassifier()),
#           ('SVM', SVC()), 
#           ('GNB', GaussianNB()),
#           ('XGB', XGBClassifier())
#         ]
# results = []
#     names = []
#     scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
#     target_names = ['malignant', 'benign']
# for name, model in models:
#         kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
#         cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
#         clf = model.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         print(name)
#         print(classification_report(y_test, y_pred, target_names=target_names))
# results.append(cv_results)
#         names.append(name)
# this_df = pd.DataFrame(cv_results)
#         this_df['model'] = name
#         dfs.append(this_df)
# final = pd.concat(dfs, ignore_index=True)
# return final  

if __name__ == "__main__":
        app.run(debug=True)
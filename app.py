import flask
from flask import render_template
import pickle
import numpy as np


app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods =['POST', 'GET'])

@app.route('/index', methods =['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    elif flask.request.method =='POST':
        with open('model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        IW = float(flask.request.form['IW'])
        IF = float(flask.request.form['IF'])
        VW = float(flask.request.form['VW'])
        FP = float(flask.request.form['FP'])
        X = np.array([[IW, IF, VW, FP]])
        scaler = pickle.load(open('Scaler.pkl', 'rb'))
        X_scaled = scaler.transform(X)
        y_pred = loaded_model.predict(X_scaled)
        
        return render_template('main.html', Depth= (y_pred[0][0]).round(3), Width = y_pred[0][1].round(3))
    

if __name__ =='__main__':
    app.run()
    
#как правильно проводить препроцессинг данных в продакшене
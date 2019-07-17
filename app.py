import numpy as np
from flask import Flask, session,abort,request, jsonify, render_template,redirect,url_for,flash
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as mini
import os
import stripe
import datetime

app = Flask(__name__)
pub_key ='pk_live_2pO0yUvt9xKyjAo9rca8Vkc600FWtgJuqZ'

model = pickle.load(open('models/bit_model.pkl', 'rb'))
eth_model = pickle.load(open('models/eth_model.pkl', 'rb'))
AAPL_model = pickle.load(open('models/AAPL_model.pkl', 'rb'))
MSFT_model = pickle.load(open('models/MSFT_model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

def index():
    return render_template('index.html',pub_key=pub_key)

@app.route('/pay',methods=['POST'])
def pay():
    customer = stripe.Customer.create(email=request.form['stripeEmail'], source=request.form['stripeToken'])
    charge = stripe.Charge.create(
        customer=customer.id,
        amount=19900,
        currency='usd',
        description='The Product'
    )

@app.route('/predict_bitcoin',methods=['POST'])
def predict_bitcoin():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    data = pd.read_csv('data/BTC-USD.csv')
    data= data.drop(['Date'], axis =1)
    data = data.drop('Adj Close',axis=1)
    X= data.drop(['High'],axis=1)
    y= data['High']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[3270:3277]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/BTC-USD.csv')
    date = bata['Date']
    date = date[3276:3277]
    print(date)
    bata = pd.read_csv('data/BTC-USD.csv')
    date = bata['Date']
    print('PREDICTED HIGH')
    y = model.predict(future_x)
    print(y[3276:3277])
    y = model.predict(future_x)
    print(y[3276:3277])
    output =y[3276:3277]
    date = datetime.date.today()
    return render_template('index.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED HIGH FOR Bitcoin ON THE DAY OF {} IS $ {}'.format(date,output))

@app.route('/predict_ethereum',methods=['POST'])
def predict_ethereum():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    data = pd.read_csv('data/ETH-USD.csv')
    data= data.drop(['Date'], axis =1)
    data = data.drop('Adj Close',axis=1)
    X= data.drop(['High'],axis=1)
    y= data['High']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[1423:1430]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/ETH-USD.csv')
    date = bata['Date']
    date = date[1429:1430]
    print(date)
    bata = pd.read_csv('data/ETH-USD.csv')
    date = bata['Date']
    print('PREDICTED HIGH')
    y = eth_model.predict(future_x)
    print(y[1429:1430])
    y = eth_model.predict(future_x)
    print(y[1429:1430])
    output =y[1429:1430]
    date = datetime.date.today()
    return render_template('index.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED HIGH FOR Ethereum ON THE DAY OF {} IS $ {}'.format(date,output))

@app.route('/predict_APPLE',methods=['POST'])
def predict_APPLE():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    data = pd.read_csv('data/AAPL.csv')
    data = data.fillna(28.630752329973255)
    data= data.drop(['Date'], axis =1)
    data = data.drop('Adj Close',axis=1)
    X= data.drop(['High'],axis=1)
    y= data['High']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[9716:9723]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/AAPL.csv')
    date = bata['Date']
    date = date[9722:9723]
    print(date)
    bata = pd.read_csv('data/AAPL.csv')
    date = bata['Date']
    print('PREDICTED HIGH')
    y = AAPL_model.predict(future_x)
    print(y[9722:9723])
    y = AAPL_model.predict(future_x)
    print(y[9722:9723])
    output =y[9722:9723]
    date = datetime.date.today()
    return render_template('index.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED HIGH FOR APPLE ON THE DAY OF {} IS $ {}'.format(date,output))

@app.route('/predict_MSFT',methods=['POST'])
def predict_MSFT():
    '''
    For rendering results on HTML GUI
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
    data = pd.read_csv('data/MSFT.csv')
    data= data.drop(['Date'], axis =1)
    data = data.drop('Adj Close',axis=1)
    X= data.drop(['High'],axis=1)
    y= data['High']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[8390:8397]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/MSFT.csv')
    date = bata['Date']
    date = date[8396:8397]
    print(date)
    bata = pd.read_csv('data/MSFT.csv')
    date = bata['Date']
    print('PREDICTED HIGH')
    y = MSFT_model.predict(future_x)
    print(y[8396:8397])
    y = MSFT_model.predict(future_x)
    print(y[8396:8397])
    output =y[8396:8397]
    date = datetime.date.today()
    return render_template('index.html', prediction_text='THANK YOU FOR YOUR PURCHASE,\n PREDICTED HIGH FOR MICROSOFT ON THE DAY OF {} IS $ {}'.format(date,output))

@app.route('/predict_api',methods=['GET'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler as mini
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
    data = pd.read_csv('data/BTC-USD.csv')
    data= data.drop(['Date'], axis =1)
    data = data.drop('Adj Close',axis=1)
    X= data.drop(['High'],axis=1)
    y= data['High']
    mini = mini()
    X = mini.fit_transform(X)
    future_x = X
    X = X[3270:3277]
    # future_x = X[-1]
    # x = X[:-1]
    bata = pd.read_csv('data/BTC-USD.csv')
    date = bata['Date']
    date = date[3276:3277]
    print(date)
    bata = pd.read_csv('data/BTC-USD.csv')
    date = bata['Date']
    print('PREDICTED HIGH')
    y = model.predict(future_x)
    print(y[3276:3277])

    output =y[3276:3277]

    return jsonify(output)
@app.route('/thanks')
def thanks():
    return render_template('thanks.html')

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=8080) #debug=True,host="0.0.0.0",port=50000

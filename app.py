# Importing Libraries
from flask import Flask, render_template, request
import pickle
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Initializing the flask app
app = Flask(__name__,template_folder='templates')

# Opening the pickle files containing the models
model1 = pickle.load(open('models/model1.pkl','rb'))
model2 = pickle.load(open('models/model2.pkl','rb'))
model3 = pickle.load(open('models/model3.pkl','rb'))

# Rendering the index file
@app.route('/')
def index():
    return render_template('index.html')

# Rendering the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Predicting Stock Price based on the form input from index.html
@app.route('/predict', methods=['POST','GET'])
def predict():

    # Storing the input from the form
    company = request.form['company']

    # Initializing the start and end dates
    today = datetime.now()
    date_60_days_ago = today - timedelta(days=100)
    end = today.strftime('%Y-%m-%d')
    start = date_60_days_ago.strftime('%Y-%m-%d')

    # Initializing the MinMax Scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Downloading the stock price data
    if company == "google":
        quote = yf.download('GOOGL', start, end)
    elif company == "apple":
        quote = yf.download('AAPL', start, end)
    elif company == "tesla":
        quote = yf.download('TSLA', start, end)

    # Preparing the data for prediction
    df = quote.filter(['Close'])
    last_60_days = df[-60:].values
    last_60_days_scaled = scaler.fit_transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predicting the stock price based on the input from the form
    if company == "google":
        pred_price = (model1.predict(X_test))
        pred_price = scaler.inverse_transform(pred_price)
        return render_template('index.html', pred='Google\'s Stock Price will close today at {:.3f}'.format(pred_price[0][0]))
    elif company == "apple":
        pred_price = (model2.predict(X_test))
        pred_price = scaler.inverse_transform(pred_price)
        return render_template('index.html', pred='Apple\'s Stock Price will close today at {:.3f}'.format(pred_price[0][0]))

    elif company == "tesla":
        pred_price = (model3.predict(X_test))
        pred_price = scaler.inverse_transform(pred_price)
        return render_template('index.html', pred='Tesla\'s Stock Price will close today at {:.3f}'.format(pred_price[0][0]))

    else:
        return render_template('index.html', pred='Error')


# Running the flask app
if __name__ == '__main__':
    app.run(debug=True)
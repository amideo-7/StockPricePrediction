# Stock Price Prediction
This project uses machine learning to predict the future price of a stock. The project uses a Long Short-Term Memory (LSTM) neural network to predict the stock price. 

The LSTM neural network is a type of recurrent neural network that is well-suited for time series data, such as stock prices. LSTM models were designed to address the vanishing gradient problem. LSTM models have a special structure that allows them to learn long-term dependencies in data.

<div align="center">

<img src="https://github.com/amideo-7/StockPricePrediction/blob/main/Images/stock.png" width="40%" height="40%"/>

</div>

The project uses the following steps to predict the stock price:

- Collect historical stock price data.
- Clean and prepare the data.
- Split the data into a training set and a test set.
- Train the LSTM neural network on the training set.
- Test the LSTM neural network on the test set.
- Evaluate the performance of the LSTM neural network.

The project uses the following libraries:

- **Pandas**: Used to create dataframes and manipulate data.
- **NumPy**: Used to create easy to use and flexible array objects.
- **Datetime**: Used to work with dates and times. It provides a number of classes and functions for creating, manipulating, and formatting dates and times.
- **Matplotlib**:  Used to create plots and graphs of the data.
- **sklearn.preprocessing**: This library is used for its MinMaxScaler function in this project, to scale the values to make the calculations easier.
- **keras.models**: In this project it is used for its sequential model that provides linear stack of layers.
- **keras.layers**: In this project Dense, and LSTM model are used form this library.
- **pandas_datareader**: This library is used to read the opening and closing stock price os the given company from the given interval of time.
- **yfinance**: This libraray can be used as an alternate to read the stock prices from yahoo finance.

## Graph Plots:

***Opening Price Data***

![Opening_data](https://github.com/amideo-7/StockPricePrediction/blob/main/Images/opening_data.png)

***Closing Price Data***

![Closing_data](https://github.com/amideo-7/StockPricePrediction/blob/main/Images/closing_data.png)

***Predictions***

![Predictions](https://github.com/amideo-7/StockPricePrediction/blob/main/Images/predictions.png)

This concludes Stock Price Prediction project
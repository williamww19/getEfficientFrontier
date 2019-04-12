import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

quandl.ApiConfig.api_key = 'VryiW3X3hUb67WZtqmsj'

selected = ['AAPL', 'GOOGL', 'AMZN', 'FB', 'TSLA']

data = quandl.get_table('WIKI/PRICES', ticker=selected,
                        qopts={'columns':['date', 'ticker', 'adj_close']},
                        date = {'gte':'2014-1-1', 'lte':'2018-12-31'}, paginate=True)

table = data.set_index('date').pivot(columns='ticker')

# calculate daily and annual returns of the stocks
returns_daily = table.pct_change()
returns_annual = returns_daily.mean() * 250 # strong assumption of return calculation

# calculate daily and covariance of returns of the stocks
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 250

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(selected)

np.random.seed(1234)

def getPort(num_portfolios=50000):
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights) # sum = 1
        returns = np.dot(weights, returns_annual)
        # cal std. - square root of
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        sharpe = returns / volatility
        sharpe_ratio.append(sharpe)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)

    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility,
                 'Sharpe Ratio': sharpe_ratio}

    return portfolio

def getDF(port):
    # create dataframe
    for counter, symbol in enumerate(selected):
        port[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

    df = pd.DataFrame(port)

    column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in selected]
    df = df[column_order]

    return df

def plotEF(df):
    # plot the efficient frontier with a scatter plot
    min_volatility = df['Volatility'].min()
    max_sharpe = df['Sharpe Ratio'].max()

    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    min_variance_port = df.loc[df['Volatility'] == min_volatility]

    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='k', figsize=(10,8), grid=True)
    plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
    plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
    plt.xlabel('Std.')
    plt.ylabel('Exp. Return')
    plt.title('Portfolio EF')
    plt.savefig('./images/PortfolioEfficientFrontier.png')

if __name__ == '__main__':
    port = getPort()
    df = getDF(port)
    plotEF(df)
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

ticker = '^GSPC'
data = yf.download(ticker, period='5y', threads=False)
data = data['Close']
shift = data.shift(1).dropna()
returns = np.log(data/shift)

mu = returns.mean() # mu represents drift
sigma = returns.std() # sigma represents volatility

print(f'1d \t drift = {mu.iloc[0]:0.6f}, vol = {sigma.iloc[0]:0.6f}')

#montecarlo simulations to calculate VaR and CVaR
num_sims = 10**6
confidence_interval = 0.95

sim_returns = np.random.normal(mu, sigma, num_sims)

var_95 = np.percentile(sim_returns, 100-confidence_interval*100)
#95%VaR indicates 95% certainty loss won't exceed given value
cvar_95 = sim_returns[sim_returns < var_95].mean()
#95%ConditionalVaR indicates average value of losses exceeding 95%VaR

print(f'1d \t 95%VaR = {abs(round(var_95*100, 2))}%, 95%CVaR = {abs(round(cvar_95*100, 2))}%')
plt.hist(sim_returns, bins=100, color='skyblue', edgecolor='black')
plt.title('Distribution of Simulated Returns')
plt.xlabel('Log 1-day Return (%)')
plt.xlim(-0.045, 0.045)
plt.ylabel('Frequency')
plt.axvline(var_95, color='red', linestyle='dashed', label='95% VaR')
plt.axvline(cvar_95, color='green', linestyle='dashed', label='95% CVaR')
plt.legend()
plt.show()

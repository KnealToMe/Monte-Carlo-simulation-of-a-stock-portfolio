import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import warnings
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# Supprimer les warnings
warnings.simplefilter(action='ignore')

# Récupération des données
def get_data(stocks, start, end):
    data = yf.download(stocks, start=start, end=end, progress=False)['Close']
    returns = data.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

# Performance du portefeuille
def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns * weights) * Time
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(Time)
    return returns, std

# Liste de tickers
stockList = ['AAPL', 'MSFT', 'GOOGL']
stocks = stockList

# Période d'analyse
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

# Récupération des données
returns, meanReturns, covMatrix = get_data(stocks, startDate, endDate)
returns = returns.dropna()

# Poids aléatoires
weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)

# Ajout de la colonne portefeuille
returns['portfolio'] = returns.dot(weights)

# Calcul des performances
Time = 100
pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)

# Fonctions VaR / CVaR historiques
def historicalVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=alpha)
    else:
        raise TypeError("Expected returns to be dataframe or series")

def historicalCVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha=alpha)
    else:
        raise TypeError("Expected returns to be dataframe or series")

# Calcul VaR / CVaR historiques
InitialInvestment = 10000
hVaR = -historicalVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)
hCVaR = -historicalCVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)

# Affichage des résultats
print("Poids du portefeuille :", weights)
print("\nRendement attendu :", round(InitialInvestment * pRet, 2))
print("Volatilité attendue :", round(InitialInvestment * pStd, 2))
print("Value at Risk 95% :", round(InitialInvestment * hVaR, 2))
print("Conditional VaR 95% :", round(InitialInvestment * hCVaR, 2))
print("\nMatrice de covariance :\n", covMatrix)

# Fonctions VaR / CVaR paramétriques
def var_parametric(portofolioReturns, portfolioStd, distribution='normal', alpha=5, dof=6):
    if distribution == 'normal':
        VaR = norm.ppf(1 - alpha / 100) * portfolioStd - portofolioReturns
    elif distribution == 't-distribution':
        nu = dof
        VaR = np.sqrt((nu - 2) / nu) * t.ppf(1 - alpha / 100, nu) * portfolioStd - portofolioReturns
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return VaR

def cvar_parametric(portofolioReturns, portfolioStd, distribution='normal', alpha=5, dof=6):
    if distribution == 'normal':
        CVaR = (alpha / 100) ** -1 * norm.pdf(norm.ppf(alpha / 100)) * portfolioStd - portofolioReturns
    elif distribution == 't-distribution':
        nu = dof
        xanu = t.ppf(alpha / 100, nu)
        CVaR = -1 / (alpha / 100) * (1 - nu) ** -1 * (nu - 2 + xanu ** 2) * t.pdf(xanu, nu) * portfolioStd - portofolioReturns
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return CVaR

# Calcul VaR / CVaR paramétriques
normVaR = var_parametric(pRet, pStd)
normCVaR = cvar_parametric(pRet, pStd)
tVaR = var_parametric(pRet, pStd, distribution='t-distribution')
tCVaR = cvar_parametric(pRet, pStd, distribution='t-distribution')

print("Normal VaR 95% :", round(InitialInvestment * normVaR, 2))
print("Normal CVaR 95% :", round(InitialInvestment * normCVaR, 2))
print("t-distribution VaR 95% :", round(InitialInvestment * tVaR, 2))
print("t-distribution CVaR 95% :", round(InitialInvestment * tCVaR, 2))

# Simulation Monte Carlo
mc_sims = 400
T = 100
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns).T
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

for m in range(mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + L @ Z.T
    portfolio_sims[:, m] = np.cumprod(np.dot(weights, dailyReturns) + 1) * InitialInvestment

# Affichage graphique
plt.plot(portfolio_sims)
plt.ylabel('Equity ($)')
plt.xlabel('Days')
plt.title('Monte Carlo simulation of a stock portfolio')
plt.show()


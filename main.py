# 포트폴리오 Risk-Return 차트 그리는 함수

from pandas_datareader import data as web
import plotly.express as px
import ipywidgets as widgets
from ipywidgets import interact
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SPY, IED(7Y~10Y USD) 데이터를 가져옵니다.
df_SPY = web.DataReader('SPY',data_source='yahoo',start='20180101',end='20210601')
df_IEF = web.DataReader('IEF',data_source='yahoo',start='20180101',end='20210601')
Price = pd.DataFrame({'SPY':df_SPY['Close'], 'US10Y': df_IEF['Close']})

# Default 값 설정합니다.
start_date = '20180101' # 투자 시작일
end_date = '20210601' # 투자 종료일
wgt1 = 0.5
wgt2 = 0.5

# 일별 수익률을 구합니다.
log_Price = np.log(Price / Price.shift(1))
log_Price = log_Price.dropna()
log_Price.columns = ['SPY','US10Y']

# 연평균 수익률을 구합니다.
mean = log_Price.mean() * 252

# 주식 50%, 채권 50%를 가정합니다.
wgt = np.array([0.7, 0.3])

# 포트폴리오 기대 수익률을 구합니다.
port_return = wgt.dot(mean)

# 포트폴리오 공분산을 구합니다.
cov_mat = log_Price.cov() * 252
cov_mat = cov_mat.values # 행렬구조로 저장합니다

# 포트폴리오의 분산을 계산합니다.
port_var = np.dot(np.dot(wgt, cov_mat), wgt.T)
port_std = np.sqrt(port_var)

# SPY, US10Y, PF(0.5,0.5) risk_rtn 차트를 그립니다.
plt.figure()
plt.plot(np.sqrt(cov_mat[0][0]),mean[0], marker='s', color='#1F77B4', markeredgewidth=1, markersize=10, label='SPY');
plt.plot(np.sqrt(cov_mat[1][1]),mean[1], marker='s', color='#FF7F0E', markeredgewidth=1, markersize=10, label='US10Y');
plt.plot(port_std, port_return, marker='x', color='#2CA02C', markeredgewidth=2, markersize=10, label='Portfolio')
plt.plot(0,0)
plt.title('Risk-Return Chart')
plt.xlabel('Risk')
plt.ylabel('Return')

plt.legend();
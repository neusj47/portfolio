# 포트폴리오 Risk-Return 차트 그리는 함수

from pandas_datareader import data as web
import plotly.express as px
import ipywidgets as widgets
from ipywidgets import interact
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from datetime import date

# SPY, IEF(7Y~10Y USD) 데이터를 가져옵니다.
df_SPY = web.DataReader('SPY',data_source='yahoo',start='20180101',end='20210601')
df_IEF = web.DataReader('IEF',data_source='yahoo',start='20180101',end='20210601')
Price = pd.DataFrame({'SPY':df_SPY['Close'], 'US10Y': df_IEF['Close']})

# Default 값 설정합니다.
start_date = '20200101' # 투자 시작일
end_date = '20201231' # 투자 종료일
wgt1 = 0.6
wgt2 = 0.4
# 일별 수익률을 구합니다.
log_Price = np.log(Price / Price.shift(1))
log_Price = log_Price.dropna()
log_Price.columns = ['SPY','US10Y']
# 연평균 수익률을 구합니다.
mean = log_Price.mean() * 252
# 주식 60%, 채권 40%를 가정합니다.
wgt = np.array([wgt1,wgt2])
# 포트폴리오 기대 수익률을 구합니다.
port_return = wgt.dot(mean)
# 포트폴리오 공분산을 구합니다.
cov_mat = log_Price.cov() * 252
cov_mat = cov_mat.values # 행렬구조로 저장합니다
# 포트폴리오의 분산을 계산합니다.
port_var = np.dot(np.dot(wgt, cov_mat), wgt.T)
port_std = np.sqrt(port_var)

PF = pd.DataFrame({'PF':('SPY', 'US10Y', 'PF'), 'Risk' : (cov_mat[0][0],cov_mat[1][1],port_var)
                   , 'Return' : (mean[0], mean[1], port_return)})

fig = px.scatter(PF, x ='Risk', y='Return', color = 'PF')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
html.H1(children='Portfolio simulation'),

    html.Div(children='''
        백테스팅 기간을 입력하시오. 
    '''),
    html.Br(),
    dcc.DatePickerRange(
        id="my-date-picker-range",
        min_date_allowed=date(2015, 1, 1),
        start_date_placeholder_text='20200101',
        end_date_placeholder_text='20201231',
        display_format='YYYYMMDD'
    ),
    html.Br(),
    html.Br(),
    html.Div(children='''
       SPY와 US10Y에 대한 Weight를 입력하시오. (전체 = 1)
   '''),
    html.Br(),
    dcc.Input(
        id="wgt1", type="number", debounce=True, placeholder="0.6"
    ),
    dcc.Input(
        id="wgt2", type="number", debounce=True, placeholder="0.4"
    ),
    html.Br(),
    dcc.Graph(
        id="port_chart", figure = fig
    )
])

@app.callback(
    Output('port_chart', 'figure'),
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
    Input("wgt1", 'value'),
     Input('wgt2', 'value')])
def PF_chart (start_date, end_date, wgt1, wgt2):

    df_SPY = web.DataReader('SPY', data_source='yahoo', start=start_date, end=end_date)
    df_IEF = web.DataReader('IEF', data_source='yahoo', start=start_date, end=end_date)
    Price = pd.DataFrame({'SPY': df_SPY['Close'], 'US10Y': df_IEF['Close']})

    log_Price = np.log(Price / Price.shift(1))
    log_Price = log_Price.dropna()
    log_Price.columns = ['SPY', 'US10Y']
    # 연평균 수익률을 구합니다.
    mean = log_Price.mean() * 252
    # 주식 60%, 채권 40%를 가정합니다.
    wgt = np.array([wgt1, wgt2])
    # 포트폴리오 기대 수익률을 구합니다.
    port_return = wgt.dot(mean)
    # 포트폴리오 공분산을 구합니다.
    cov_mat = log_Price.cov() * 252
    cov_mat = cov_mat.values  # 행렬구조로 저장합니다
    # 포트폴리오의 분산을 계산합니다.
    port_var = np.dot(np.dot(wgt, cov_mat), wgt.T)
    # port_std = np.sqrt(port_var)

    PF_chart = pd.DataFrame({'PF': ('SPY', 'US10Y', 'PF'), 'Risk': (cov_mat[0][0], cov_mat[1][1], port_var)
                          , 'Return': (mean[0], mean[1], port_return)})
    fig = px.scatter(data_frame=PF_chart, x='Risk', y='Return', color = 'PF')
    return fig


def port_return_cum(start_date, end_date, wgt1, wgt2):
    wgt = np.array([wgt1, wgt2])
    df_SPY = web.DataReader('SPY', data_source='yahoo', start=start_date, end=end_date)
    df_IEF = web.DataReader('IEF', data_source='yahoo', start=start_date, end=end_date)
    Price = pd.DataFrame({'SPY': df_SPY['Close'], 'US10Y': df_IEF['Close']})

    log_Price = np.log(Price / Price.shift(1))
    log_Price = log_Price.dropna()
    log_Price.columns = ['SPY', 'US10Y']

    log_Rtn_cum = (1 + log_Price).cumprod() - 1

    return log_Rtn_cum



# SPY, US10Y, PF(0.5,0.5) risk_rtn 차트를 그립니다.
# plt.figure()
# plt.plot(np.sqrt(cov_mat[0][0]),mean[0], marker='s', color='#1F77B4', markeredgewidth=1, markersize=10, label='SPY');
# plt.plot(np.sqrt(cov_mat[1][1]),mean[1], marker='s', color='#FF7F0E', markeredgewidth=1, markersize=10, label='US10Y');
# plt.plot(port_std, port_return, marker='x', color='#2CA02C', markeredgewidth=2, markersize=10, label='Portfolio')
# plt.plot(0,0)
# plt.title('Risk-Return Chart')
# plt.xlabel('Risk')
# plt.ylabel('Return')#
# plt.legend();


if __name__ == "__main__":
    app.run_server(debug=True, port = 8060)
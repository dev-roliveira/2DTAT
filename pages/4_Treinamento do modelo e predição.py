import streamlit as st
import pickle
import streamlit as st
# Bibliotecas básicas de data science
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Para deep learning
import keras
from tensorflow.keras.preprocessing.sequence  import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

with open('modelo_brent.pkl', 'rb') as file_2:
    model = pickle.load(file_2)


# Carregar o DataFrame
df = pd.read_csv('ipea.csv', sep=';', decimal=',')
df.rename(columns={'Preço - petróleo bruto - Brent (FOB)': 'Fechamento'}, inplace=True)

# Converter a coluna de data para datetime e depois para timestamp Unix
df['Data'] = pd.to_datetime(df['Data'])
df = df.sort_values(by='Data',ascending=True)
alpha = 0.09   # Fator de suavização

df['Smoothed_Close'] = df['Fechamento'].ewm(alpha=alpha, adjust=False).mean()

close_data = df['Smoothed_Close'].values
close_data = close_data.reshape(-1,1) #transformar em array

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(close_data)
close_data = scaler.transform(close_data)

split_percent = 0.80
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Data'][:split]
date_test = df['Data'][split:]
look_back = 4

test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)
test_predictions = model.predict(test_generator)

# 1. Fazer previsões usando o conjunto de teste
test_predictions = model.predict(test_generator)

# 2. Inverter qualquer transformação aplicada aos dados
test_predictions_inv = scaler.inverse_transform(test_predictions.reshape(-1, 1))
test_actuals_inv = scaler.inverse_transform(np.array(close_test).reshape(-1, 1))

# Ajuste as dimensões
test_actuals_inv = test_actuals_inv[:len(test_predictions_inv)]

# Calcular o MAPE
mape = np.mean(np.abs((test_actuals_inv - test_predictions_inv) / test_actuals_inv)) * 100

# Avaliando o modelo nos dados de teste
mse = model.evaluate(test_generator, verbose=1)

# O RMSE é a raiz quadrada do MSE (Mean Squared Error), que é a média dos quadrados das diferenças entre as previsões do modelo e os valores reais.
rmse_value = np.sqrt(mse[0])

prediction = model.predict(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))

trace1 = go.Scatter(
    x = date_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = date_test,
    y = close_test,
    mode='lines',
    name = 'Ground Truth'
)
layout = go.Layout(
    title = "Treinamento do Modelo",
    xaxis = {'title' : "Data"},
    yaxis = {'title' : "Fechamento"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)


st.markdown('# Modelo Preditivo')
st.markdown('Foram realizados vários testes com modelos diferentes (Média Movel, Média Movel Aritmética, ARIMA, PROPHET e LSTM) até chegar em um modelo aceitável utilizando LSTM')
st.markdown('Esse modelo obteve um Erro Quadrático Médio de: ' + str(round(mse[0],4)))
st.markdown('E um MAPE de: ' + str(round(mape,2)))
st.markdown('O RSME desse modelo é de: ' + str(round(rmse_value, 3)))

st.plotly_chart(fig)
#---------------------------------------

# Forecastig

close_data = close_data.reshape((-1))

# Função para prever os próximos 'num_prediction' pontos da série temporal
# Utiliza o modelo treinado para prever cada ponto sequencialmente
# A cada iteração, adiciona a previsão à lista 'prediction_list'

def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]

    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]

    return prediction_list

# Função para gerar as datas dos próximos 'num_prediction' dias
# Assume que o DataFrame 'df' possui uma coluna 'Date' contendo as datas

def predict_dates(num_prediction):
    last_date = df['Data'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

num_prediction = 4 #definição dos próximos dias
forecast = predict(num_prediction, model) #resultado de novos dias
forecast_dates = predict_dates(num_prediction)

df = pd.DataFrame(df)
df_past = df[['Data','Smoothed_Close']]
df_past.rename(columns={'Smoothed_Close': 'Actual'}, inplace=True)         #criando nome das colunas
df_past['Data'] = pd.to_datetime(df_past['Data'])                          #configurando para datatime
df_past['Forecast'] = np.nan                                               #Preenchendo com NAs
df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

# Faz a transformação inversa das predições
forecast = forecast.reshape(-1, 1) #reshape para array
forecast = scaler.inverse_transform(forecast)

df_future = pd.DataFrame(columns=['Data', 'Actual', 'Forecast'])
df_future['Data'] = forecast_dates
df_future['Forecast'] = forecast.flatten()
df_future['Actual'] = np.nan

# Concatenando os DataFrames usando concat
frames = [df_past, df_future]
results = pd.concat(frames, ignore_index=True).set_index('Data')

results2024 =  results.loc['2024-01-01':]

plot_data = [
    go.Scatter(
        x=results2024.index,
        y=results2024['Actual'],
        name='actual'
    ),
    go.Scatter(
        x=results2024.index,
        y=results2024['Forecast'],
        name='prediction'
    )
]

plot_layout = go.Layout(
        title='Forecast do Modelo Treinado'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
st.plotly_chart(fig)
import streamlit as st

# Bibliotecas básicas de data science
import pandas as pd
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

# Carregar o DataFrame
df = pd.read_csv('ipea.csv', sep=';', decimal=',')

# Transdformar coluna em datetime
df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')

# Ordenar valores
df = df.sort_values(by='Data',ascending=True)

# Setar index
df.set_index('Data', drop=True, inplace=True)

# Escalar a coluna de preços, já que os modelos de DL geralmente funcionam melhor com dados normalizados
df.rename(columns={'Preço - petróleo bruto - Brent (FOB)': 'Fechamento'}, inplace=True)

st.markdown('# Petroleo Brent ao longo dos anos')
st.markdown('O petróleo Brent foi batizado assim porque era extraído de uma base da Shell chamada Brent. Atualmente, a palavra Brent designa todo o petróleo extraído no Mar do Norte e comercializado na Bolsa de Londres. ' + 
            'A cotação do petróleo Brent é referência para os mercados europeu e asiático.')
st.markdown('A cotação do mercado nos afeta diretamente, influenciando no preço de seus derivados como óleo diesel e a gasolina. Impactando o preço do transporte e afetando todas as mercadorias.')
st.markdown('Os preços dependem principalmente do custo de produção e transporte.') 

st.slider(
    "Ano",
    min_value=1987,
    max_value=2024,
    value=[1987, 2024],
    key='filtro_ano'
)

ano_inicial = st.session_state.filtro_ano[0]
ano_final = st.session_state.filtro_ano[1]

df.reset_index(inplace=True)

def plotar(ano_inicial, ano_final):
    date_data = df[(df['Data'].dt.year >= ano_inicial) & (df['Data'].dt.year <=ano_final)]

    trace1 = go.Scatter(
        x = date_data['Data'],
        y = date_data['Fechamento'],
        mode = 'lines',
        name = 'Data'
    )
    layout = go.Layout(
        title = "Evolução do valor do Petroleo Brent de 1987 a 2024",
        xaxis = {'title' : "Data"},
        yaxis = {'title' : "Fechamento"},
    )
    fig = go.Figure(data=trace1, layout=layout)
    st.plotly_chart(fig)

plotar(ano_inicial, ano_final)

st.markdown('## Épocas de destaque:')
with st.expander("1991"):
    st.markdown('Em 1991 tivemos a época conhecida como 4ª Crise do Petróleo.' + 
             '\n Nessa época, iniciou-se a guerra na Palestina. Bem como a guerra do golfo, ' + 
             'que com a invasão do Kuwait pelo Iraque, culminou em incêndios de poços de petróleo no país ' + 
             'ao serem expulsos pelos EUA.') 
    
with st.expander("Primeiro semestre de 2008"):
    st.markdown('Em Julho de 2008 o Petróleo atingiu seu máximo histórico, devido à tensões entre Irã, Nigéria e Paquistão. ' + 
                'Além da conscientização de que trata-se de um recurso limitado, ' + 
                'e grande procura em fundos de investimentos por matérias-primas.')     

with st.expander("Segundo semestre de 2008"):
    st.markdown('Observa-se uma queda no valor do Petróleo Puro a partir de Julho de 2008.' + 
                'Essa queda bruta e sem precedentes até então, foi resultado de crises financeira, ' + 
                'como a falência do Banco Lehman Brothers. ' + 
                'Que fez com que os investidores que antes adotaram à febre de investimento por matérias-primas, ' + 
                'abandonassem esse produto por precisarem de liquidez.' +
                'Além disso, o petróleo caro faz com que o consumo de combustível caia, e com isso a demanda diminua.')     

with st.expander("Crise de 2015"):
    st.markdown('No ano de 2015, tivemos mais uma crise de petróleo, com quedas significativas no preço do barril. ' + 
                'Os principais apontados como "culpados" pela queda dos preços são o aumento de produção, em especial nas áreas de xisto dos EUA, e uma demanda menor que a esperada na Europa e na Ásia.')    

with st.expander("Pandemia COVID-19"):
    st.markdown('No final de 2019 e início de 2020 vivemos o começo da pandemia COVID-19, ' + 
                'e toda a situação caótica mundial afetou os preços do petróleo.')      
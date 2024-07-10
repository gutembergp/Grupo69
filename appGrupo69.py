#Importação das bibliotecas
import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
#from utils import DropFeatures, OneHotEncodingNames, OrdinalFeature, MinMaxWithFeatNames
from sklearn.pipeline import Pipeline
import joblib
from joblib import load


############################# Streamlit ############################
st.markdown('<style>div[role="listbox"] ul{background-color: #6e42ad}; </style>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; '> Formulário para Análise do Ponto de Virada </h1>", unsafe_allow_html = True)

st.warning('Preencha o formulário com todos os Indicadores do aluno e clique **ENVIAR** no final da página.')

# IAN
st.write('Indicador de Adequação de Nível ')
ian = float(st.slider('IAN ', 0, 10))

# IDA
st.write('Indicador de Desenvolvimento Acadêmico')
ida = float(st.slider('IDA ', 0, 10))

# IDG
st.write('Indicador de Engajamento')
idg = float(st.slider('IEG ', 0, 10))

# IAA
st.write('Indicador de Auto Avaliação')
iaa = float(st.slider('IAA ', 0, 10))

# IPS
st.write('Indicador Psico Social')
ips = float(st.slider('IPS ', 0, 10))

# IPP
st.write('IPP')
st.write('Indicador Psico Pedagógico')
ipp = float(st.slider('Selecione o IPP ', 0, 10))

# IPV
st.write('Indicador Ponto de Virada')
ipv = float(st.slider('IPV ', 0, 10))

# Lista de todas as variáveis: 
novo_aluno = [[ ian, 
               ida, 
               idg,
               iaa, 
               ips, 
               ipp, 
               ipv]]
            


#Criando novo aluno
df_aluno_amostra = pd.DataFrame(novo_aluno)
df_target = pd.DataFrame(columns=['PONTO_VIRADA'])
df_target = pd.Series([0])

for i in range(10):
    df_aluno_amostra = pd.concat([df_aluno_amostra, df_aluno_amostra])
    df_target = pd.concat([df_target, df_target])

x = df_aluno_amostra
y = df_target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y,
                                                    random_state=7) #20% para teste e 80% de treino

#Predições 
if st.button('Enviar'):
    model = joblib.load('/mount/src/grupo69/xgb.joblib')
    final_pred = model.predict(df_aluno_amostra)
    if final_pred[-1] == 1:
        st.success('### Parabéns! Você Fez a Virada')
        st.balloons()
    else:
        st.error('### Continue se Esforçando!')
 

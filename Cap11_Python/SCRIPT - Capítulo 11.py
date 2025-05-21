##############################################################################
#                     MANUAL DE ANÁLISE DE DADOS                             #
#                Luiz Paulo Fávero e Patrícia Belfiore                       #
#                            Capítulo 11                                     #
##############################################################################
#!/usr/bin/env python
# coding: utf-8
# Nossos mais sinceros agradecimentos aos Professores Helder Prado Santos e
#Wilson Tarantin Junior pela contribuição com códigos e revisão do material.

##############################################################################
#                     IMPORTAÇÃO DOS PACOTES NECESSÁRIOS                     #
##############################################################################

import pandas as pd #manipulação de dados em formato de dataframe
import numpy as np #biblioteca para operações matemáticas multidimensionais
import matplotlib.pyplot as plt #biblioteca de visualização de dados
import seaborn as sns #biblioteca de visualização de informações estatísticas
from scipy.stats import chi2_contingency #estatística qui-quadrado e teste
import statsmodels.api as sm #cálculo de estatísticas da tabela de contingência
from scipy.stats.contingency import margins #cálculo manual dos resíduos padronizados
from scipy.linalg import svd #valores singulares e autovetores (eigenvectors)
import prince #funções 'CA' e 'MCA' para elaboração direta da Anacor e da MCA
import plotly.graph_objects as go #biblioteca para gráficos interativos
import plotly.io as pio #biblioteca para gráficos interativos
pio.renderers.default = 'browser' #biblioteca para gráficos interativos


#%%
##############################################################################
#       DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'perfil_investidor_aplicacao'      #
##############################################################################

# Carregamento da base de dados 'perfil_investidor_aplicacao'
df_perfil = pd.read_csv('perfil_investidor_aplicacao.csv', delimiter=',')

# Visualização da base de dados 'perfil_investidor_aplicacao'
df_perfil

# Características das variáveis do dataset
df_perfil.info()

# Tabelas de frequência das variáveis qualitativas 'perfil' e 'aplicacao'
df_perfil['perfil'].value_counts()
df_perfil['aplicacao'].value_counts()


#%%
################################################################################
#                     ANÁLISE DE CORRESPONDÊNCIA SIMPLES                       #
################################################################################

# Tabela de contingência com frequências absolutas observadas
tabela_contingencia = pd.crosstab(index = df_perfil['perfil'],
                                  columns = df_perfil['aplicacao'],
                                  margins = False)
tabela_contingencia.columns = ['Acoes','CDB','Poupanca'] #nomes das colunas
tabela_contingencia.index = ['Agressivo', 'Conservador', 'Moderado'] #nomes das linhas
tabela_contingencia #tabela de contingência com frequências absolutas esperadas
tabela_contingencia


#%%
# Estatísticas obtidas a partir da tabela de contingência por meio da
#função 'chi2_contingency' do pacote 'scipy.stats'

# Estatística qui-quadrado e teste
chi2, pvalue, df, freq_expected = chi2_contingency(tabela_contingencia)

f"estatística qui²: {chi2}" # estatística qui-quadrado
f"p-value da estatística: {pvalue}" # p-value da estatística qui-quadrado
f"graus de liberdade: {df}" # graus de liberdade

# Tabela de contingência com frequências absolutas esperadas
freq_expected
freq_expected = pd.DataFrame(data=freq_expected)
freq_expected.columns = ['Acoes','CDB','Poupanca'] #nomes das colunas
freq_expected.index = ['Agressivo', 'Conservador', 'Moderado'] #nomes das linhas
freq_expected #tabela de contingência com frequências absolutas esperadas

# Resíduos – diferenças entre frequências absolutas observadas e esperadas
tabela_contingencia - freq_expected

# Valores de qui-quadrado por célula
((tabela_contingencia - freq_expected)**2)/freq_expected

# Resíduos padronizados
(tabela_contingencia - freq_expected) / np.sqrt(freq_expected)

# Resíduos padronizados ajustados
tabela_array = np.array(tabela_contingencia)
n = tabela_array.sum()
rsum, csum = margins(tabela_array)
rsum = rsum.astype(np.float64)
csum = csum.astype(np.float64)
v = csum * rsum * (n - rsum) * (n - csum) / n**3
(tabela_array - freq_expected) / np.sqrt(v)


#%%
# Estatísticas obtidas diretamente a partir da tabela de contingência por meio
#da função 'Table' do pacote 'statsmodels'
tabela = sm.stats.Table(tabela_contingencia)    

# Estatística qui-quadrado e teste
print(tabela.test_nominal_association())

# Tabela de contingência com frequências absolutas esperadas
tabela.fittedvalues

# Resíduos – diferenças entre frequências absolutas observadas e esperadas
tabela.table_orig - tabela.fittedvalues

# Valores de qui-quadrado por célula
tabela.chi2_contribs

# Resíduos padronizados
tabela.resid_pearson

# Resíduos padronizados ajustados
tabela.standardized_resids


#%%
# Mapa de calor dos resíduos padronizados ajustados
plt.figure(figsize=(15,10))
sns.heatmap(tabela.standardized_resids, annot=True,
            cmap = plt.cm.viridis,
            annot_kws={'size':22})
plt.show()


#%%
# Massas das colunas (column profiles)
rsum, csum = margins(tabela_contingencia)
massa_colunas = rsum/rsum.sum()
massa_colunas = pd.DataFrame(data=massa_colunas)
massa_colunas.columns = ['Massas'] #nome da coluna
massa_colunas.index = ['Agressivo', 'Conservador', 'Moderado'] #nomes das linhas
massa_colunas #massas das colunas

# Massas das linhas (row profiles)
massa_linhas = csum/csum.sum()
massa_linhas = pd.DataFrame(data=massa_linhas)
massa_linhas = massa_linhas.T
massa_linhas.columns = ['Massas'] #nome da coluna
massa_linhas.index = ['Acoes','CDB','Poupanca'] #nomes das linhas
massa_linhas #massas das linhas

#%%
# Decomposição inercial para as dimensões:
# Cálculo da inércia principal total (a partir do qui-quadrado)
tabela_array = np.array(tabela_contingencia)
n = tabela_array.sum()
inercia_total = chi2/n
inercia_total

# Definição da matriz A
# Os valores das células da matriz A são iguais aos das respectivas células da
#matriz de resíduos padronizados (qui2$residuals) divididos pela raiz quadrada
#do tamanho da amostra (n)
matrizA = tabela.resid_pearson/np.sqrt(n)
matrizA

# Definição da matriz W
matrizW = np.matmul(matrizA.T, matrizA)
matrizW

# Definição da quantidade de dimensões
qtde_dimensoes = min(len(matrizW) - 1, len(matrizW[0]) - 1)
qtde_dimensoes


#%%
# Definição dos valores singulares e autovetores (função 'svd' do pacote
#'scipy.linalg')
autovetor_u, valores_singulares, autovetor_v = svd(matrizA)

# Valores singulares de cada dimensão
valores_singulares = valores_singulares[0:qtde_dimensoes]

# Autovalores (eigenvalues) de cada dimensão
eigenvalues = (valores_singulares)**2
eigenvalues

# Autovetores v das dimensões
autovetor_v.T[:,0:2]

# Autovetores u das dimensões
autovetor_u[:,0:2]

# Conforme discutido ao longo do capítulo, sabemos que:
# Os objetos 'autovetor_v' e 'autovetor_u' correspondem aos eigenvectors;
# Os valores contidos no objeto 'valores_singulares' elevados ao quadrado
#correspondem aos autovalores (eigenvalues) de cada dimensão, cuja soma
#corresponde ao valor presente no objeto 'inercia_total';
# Conforme realizamos algebricamente ao longo do capítulo, utilizaremos os
#valores presentes nesses objetos para o cálculo das coordenadas de cada
#categoria das variáveis qualitativas 'perfil' e 'aplicacao'.

# Cálculo da variância explicada em cada dimensão
variancia_explicada = eigenvalues / inercia_total
variancia_explicada

# Visualização de tabela com dimensões, valores singulares, eigenvalues,
#valores de qui-quadrado por dimensão, percentual da inércia principal total e
#percentual da inércia principal total acumulada
tabela_dim = np.array([valores_singulares,
                       eigenvalues,
                       eigenvalues*n,
                       variancia_explicada*100,
                       variancia_explicada.cumsum()*100])
tabela_dim = pd.DataFrame(data=tabela_dim)
tabela_dim = tabela_dim.T #transposição da matriz
tabela_dim.columns = ['Valor Singular', #nomes das colunas
                        'Eigenvalues',
                        'Qui²',
                        'Percentual da Inércia Principal Total',
                        'Percentual da Inércia Principal Total Acumulada']
tabela_dim.index = [f"Dimensão {i+1}" for i, #nomes das linhas
                    v in enumerate(tabela_dim.index)]
tabela_dim #tabela gerada


#%%
################################################################################
#      CÁLCULO DAS COORDENADAS DAS CATEGORIAS DAS VARIÁVEIS QUALITATIVAS       #
#                    PARA A CONSTRUÇÃO DO MAPA PERCEPTUAL                      #
################################################################################

# Variável em linha na tabela de contingência ('perfil'):
# Coordenadas das abcissas
coord_abcissas_perfil = np.sqrt(valores_singulares[0]) * (massa_colunas)**(-0.5) * autovetor_u[:,0:1]
coord_abcissas_perfil.columns = ['']
coord_abcissas_perfil

# Coordenadas das ordenadas
coord_ordenadas_perfil = np.sqrt(valores_singulares[1]) * (massa_colunas)**(-0.5) * autovetor_u[:,1:2]
coord_ordenadas_perfil.columns = ['']
coord_ordenadas_perfil

# Variável em coluna na tabela de contingência ('aplicacao'):
# Coordenadas das abcissas
coord_abcissas_aplicacao = np.sqrt(valores_singulares[0]) * (massa_linhas)**(-0.5) * autovetor_v[0:1,:].T
coord_abcissas_aplicacao.columns = ['']
coord_abcissas_aplicacao

# Coordenadas das ordenadas
coord_ordenadas_aplicacao = np.sqrt(valores_singulares[1]) * (massa_linhas)**(-0.5) * autovetor_v[1:2,:].T
coord_ordenadas_aplicacao.columns = ['']
coord_ordenadas_aplicacao


#%%
################################################################################
#                   CONSTRUÇÃO DO MAPA PERCEPTUAL DA ANACOR                    #
################################################################################

# Criação de um dataframe com as abcissas e ordenadas de cada categoria
#das variáveis qualitativas consideradas na análise

# Dataframe com coordenadas das abcissas
tabela_coord_abcissas = pd.concat([coord_abcissas_perfil,
                                   coord_abcissas_aplicacao])
tabela_coord_abcissas.columns = ['Abcissas']
tabela_coord_abcissas

# Dataframe com coordenadas das ordenadas
tabela_coord_ordenadas = pd.concat([coord_ordenadas_perfil,
                                   coord_ordenadas_aplicacao])
tabela_coord_ordenadas.columns = ['Ordenadas']
tabela_coord_ordenadas

# Dataframe com coordenadas conjuntas das abcissas e ordenadas
tabela_coord = pd.concat([tabela_coord_abcissas, tabela_coord_ordenadas],
                        axis=1)
tabela_coord

# Inserção no dataframe 'tabela_coord' de uma variável (oriunda de uma lista
#gerada no algoritmo a seguir) que indica a variável qualitativa original
#a que pertence determinada categoria
lista=[]
for i in tabela_coord.index:
    for item in df_perfil.columns:
        if i in df_perfil[item].unique():
            print(item, i)
            lista.append(item)

tabela_coord['variável'] = lista
tabela_coord #dataframe com coordenadas das abcissas e ordenadas, e variável
#a que pertencem as categorias

# Inserção, no dataframe 'tabela_coord', de um index referente às categorias
#das variáveis qualitativas originais para a elaboração do mapa perceptual
tabela_coord_chart = tabela_coord.reset_index()
tabela_coord_chart


#%%
# Construção do mapa perceptual propriamente dito
plt.figure(figsize=(12,8))
ax = sns.scatterplot(data = tabela_coord_chart,
                     x = 'Abcissas',
                     y = 'Ordenadas',
                     hue = 'variável',
                     s = 200,
                     style = 'variável')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.05, point['y'], point['val'], fontsize = 14)

label_point(x = tabela_coord_chart['Abcissas'],
            y = tabela_coord_chart['Ordenadas'],
            val = tabela_coord_chart['index'],
            ax = plt.gca())

plt.axhline(y=0, color='black', ls='--')
plt.axvline(x=0, color='black', ls='--')
plt.ylim([-1.5,1.5])
plt.xlim([-1.5,1.5])
plt.title('Mapa perceptual para perfil do investidor e tipo de aplicação financeira', fontsize=16)
plt.xlabel(f"Dimensão 1: {round(variancia_explicada[0]*100,2)}%", fontsize=16)
plt.ylabel(f"Dimensão 2: {round(variancia_explicada[1]*100,2)}%", fontsize=16)
ax.legend(fontsize=16)
plt.show()


#%%
# O que foi feito até o presente momento poderia ser obtido por meio do
#seguinte código (função 'CA' do pacote 'prince' versão 0.7.1)
anacor = prince.CA()
anacor = anacor.fit(tabela_contingencia)

# Massas das colunas (column profiles)
anacor.row_masses_ #sim, é isso mesmo ('row_masses')!

# Massas das linhas (row profiles)
anacor.col_masses_ #sim, é isso mesmo ('col_masses')!

# Inércia principal total
anacor.total_inertia_

# Quantidade de dimensões
anacor.n_components

# Autovalores (eigenvalues) de cada dimensão
anacor.eigenvalues_

# Variância explicada em cada dimensão
anacor.explained_inertia_

# Coordenadas das abcissas e ordenadas das categorias da variável 'perfil'
anacor.row_coordinates(tabela_contingencia)

# Coordenadas das abcissas e ordenadas das categorias da variável 'aplicacao'
anacor.column_coordinates(tabela_contingencia)

# Um pesquisador mais curioso irá notar que as coordenadas obtidas aqui por
#meio da função 'ca' do pacote 'prince' são proporcionais àquelas obtidas
#anteriormente de maneira algébrica. Portanto, não alteram em nada o padrão
#e o comportamento das categorias no mapa perceptual gerado a partir do
#código a seguir em relação àquele construído anteriormente pelo pacote
#'seaborn'.

# Mapa perceptual
anacor.plot_coordinates(tabela_contingencia)


#%%
# Mapa perceptual mais bem elaborado
anacor.plot_coordinates(X=tabela_contingencia,
                        ax=None,
                        figsize=(12,8),
                        x_component=0,
                        y_component=1,
                        show_row_labels=True,
                        show_col_labels=True)


#%%
# Mapa perceptual interativo

pio.renderers.default='browser'

chart_df = pd.DataFrame({'obs_x':anacor.row_coordinates(tabela_contingencia)[0].values,
                         'obs_y': anacor.row_coordinates(tabela_contingencia)[1].values})

fig = go.Figure(data=go.Scatter(x=chart_df['obs_x'],
                                y=chart_df['obs_y'],
                                name=tabela_contingencia.index.name,
                                textposition="top center",
                                text=tabela_contingencia.index,
                                mode="markers+text",))

fig.add_trace(go.Scatter(
    x=anacor.column_coordinates(tabela_contingencia)[0].values,
    mode="markers+text",
    name=tabela_contingencia.columns.name,
    textposition="top center",
    y=anacor.column_coordinates(tabela_contingencia)[1].values,
    text=anacor.column_coordinates(tabela_contingencia).index
))

fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    title_text='Coordenadas principais'

)

fig.show()


#%%
##############################################################################
#                    ANÁLISE DE CORRESPONDÊNCIA MÚLTIPLA                     #
# DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'perfil_investidor_aplicacao_estadocivil'#
##############################################################################

# Carregamento da base de dados 'perfil_investidor_aplicacao_estadocivil'
df_perfil_acm = pd.read_csv('perfil_investidor_aplicacao_estadocivil.csv',
                            delimiter=',')

# Visualização da base de dados 'perfil_investidor_aplicacao_estadocivil'
df_perfil_acm

# Características das variáveis do dataset
df_perfil_acm.info()

# Tabelas de frequência das variáveis qualitativas 'perfil', 'aplicacao' e
#'estado_civil'
df_perfil_acm['perfil'].value_counts()
df_perfil_acm['aplicacao'].value_counts()
df_perfil_acm['estado_civil'].value_counts()


#%% Criação de um dataframe 'df' que contém apenas as variáveis categóricas
#'perfil', 'aplicacao' e 'estado_civil'

df = df_perfil_acm.drop(columns=['estudante'])
df


#%%
# Tabelas de contingência com frequências absolutas observadas e esperadas,
#bem como os testes qui-quadrado
# Algoritmo para geração destas tabelas de contingência para os pares de
#variáveis categóricas presentes no dataframe 'df'

from itertools import combinations

for item in list(combinations(df.columns, 2)):
    print(item, "\n")
    tabela = pd.crosstab(df[item[0]], df[item[1]])
    
    print(tabela, "\n")
    
    chi2, pvalor, gl, freq_esp = chi2_contingency(tabela)

    print(f"estatística qui²: {chi2}") # estatística qui²
    print(f"p-valor da estatística: {pvalor}") # p-valor da estatística
    print(f"graus de liberdade: {gl} \n") # graus de liberdade


#%%
# Identificação das variáveis e de suas categorias únicas no dataframe 'df'

for col in df:
    print(col, df[col].unique())


#%%
# Ajustando as variáveis para type = 'category' no dataframe 'df'

df = df.astype('category')
df.info()


#%%
# Indicação das variáveis consideradas na MCA

mca_cols = df.select_dtypes(['category']).columns
mca_cols.tolist() #variáveis consideradas na MCA


#%%
# Elaboração da MCA propriamente dita (função 'MCA' do pacote 'prince')
mca = prince.MCA()
mca = mca.fit(df[mca_cols])


#%%
# Coordenadas das categorias das variáveis e das observações

# Coordenadas das abcissas e ordenadas das categorias das variáveis
mca.column_coordinates(df[mca_cols])

# Coordenadas das abcissas e ordenadas das observações (estudantes)
mca.row_coordinates(df[mca_cols])


#%%
##############################################################################
#                    CONSTRUÇÃO DO MAPA PERCEPTUAL DA ACM                    #
##############################################################################

# Mapa perceptual apenas com coordenadas das abcissas e ordenadas das
#categorias das variáveis
mca.plot_coordinates(X=df[mca_cols],
                     figsize=(10,8),
                     show_row_points = False,
                     show_column_points = True,
                     show_row_labels=False,
                     show_column_labels = True,
                     column_points_size = 100)


#%%
# Mapa perceptual com coordenadas das abcissas e ordenadas das categorias das
#variáveis e das observações (estudantes)
mca.plot_coordinates(X=df[mca_cols],
                     figsize=(10,8),
                     show_row_points = True,
                     show_column_points = True,
                     show_row_labels=False,
                     show_column_labels = True,
                     column_points_size = 100)


#%%
# Mapa perceptual com coordenadas das abcissas e ordenadas das categorias das
#variáveis e das observações (estudantes) com identificação
df_mca_cols = df[mca_cols]
df_mca_cols.index = df_perfil_acm['estudante']
mca.plot_coordinates(X=df_mca_cols,
                     figsize=(10,8),
                     show_row_points = True,
                     show_column_points = True,
                     show_row_labels=True,
                     show_column_labels = True,
                     column_points_size = 100)


#%%
##############################################################################
#                      MATRIZ BINÁRIA E MATRIZ DE BURT                       #
##############################################################################

#%%
# Matriz binária
df = df_perfil_acm.drop(columns=['estudante'])
df
matriz_binaria = (pd.get_dummies(df)*(-1)) + 1
matriz_binaria


#%%
# Matriz de Burt
matriz_burt = np.matmul(matriz_binaria.T, matriz_binaria)
matriz_burt


##############################################################################
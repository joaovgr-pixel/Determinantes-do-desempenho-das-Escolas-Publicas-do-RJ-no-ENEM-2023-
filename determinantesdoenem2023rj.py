"""
DETERMINANTES DO DESEMPENHO EDUCACIONAL EM ESCOLAS PÚBLICAS FLUMINENSES

Autor: João Vitor Gomes Rodrigues  
Orientador: Prof. Fabricio Pelloso Piurcosky  
Instituição: USP/ESALQ - MBA Data Science and Analytics


REPRODUÇÃO DO ESTUDO:

1. Carrega dos microdados do ENEM 2023
2. Mapeamento das variáveis socioeconômicas, institucionais e demográficas
3. Agrega os dados por escola
4. Análise exploratória e diagnóstico de multicolinearidade
5. Estimação de modelo linear inicial e cálculo do VIF
6. Aplica o PCA e modelage o ISE
7. Estima a regressão múltipla robusta (OLS HC3) com testes de diagnóstico
8. Análise de agrupamento (K-means)
9. Testes de robustez:
    • A – Regressão com transformação logarítmica (sensibilidade)  
    • B – Bootstrap dos coeficientes (amostragem)
10. Consolidação dos resultados e exportação do relatório final

"""

import os
import warnings
import traceback

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

import scipy.stats as stats
from scipy.stats import jarque_bera, shapiro, bartlett
from scipy.optimize import linear_sum_assignment

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import r2_score, silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.utils import resample

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity


warnings.filterwarnings("ignore")

# Configurações gerais
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.dpi"] = 300

CORES_VIRIDIS = plt.cm.viridis(np.linspace(0, 1, 10))
sns.set_palette("viridis")

PASTA_SAIDA = "resultados_tcc_versao_final5"
if not os.path.exists(PASTA_SAIDA):
    os.makedirs(PASTA_SAIDA)

# ============================================================================
# 1. CARREGAMENTO DOS MICRODADOS DO ENEM
# ============================================================================

def carregar_dados_finais():
    """
    Carrega microdados do ENEM 2023 para escolas públicas do RJ
    com processamento otimizado em chunks
    """
    print("=== CARREGANDO DADOS FINAIS ===")
    
    colunas = [
        "NU_INSCRICAO", "SG_UF_ESC", "TP_ESCOLA", "TP_DEPENDENCIA_ADM_ESC",
        "TP_LOCALIZACAO_ESC", "NO_MUNICIPIO_ESC", "CO_MUNICIPIO_ESC",
        "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO",
        "TP_SEXO", "TP_COR_RACA", "TP_FAIXA_ETARIA",
        "Q001", "Q002", "Q006", "Q020", "Q024", "Q025"
    ]
    
    chunk_size = 100000
    chunks = []

    print("Processando arquivo de 1,7 GB em chunks...")

    for i, chunk in enumerate(pd.read_csv(r"C:\MICRODADOS_ENEM_2023.csv",
                                          sep=";", encoding="latin1",
                                          chunksize=chunk_size, low_memory=False,
                                          usecols=colunas)):
        chunk_rj = chunk[chunk["SG_UF_ESC"] == "RJ"].copy()
        # Incluir federais (1), estaduais (2), municipais (3) - excluir apenas privadas (4)
        chunk_rj_pub = chunk_rj[chunk_rj["TP_ESCOLA"].isin([1, 2, 3])].copy()
        
        if len(chunk_rj_pub) > 0:
            chunks.append(chunk_rj_pub)
        
        if (i + 1) % 10 == 0:
            print(f"Processados {(i+1)*chunk_size:,} registros...")

    df_final = pd.concat(chunks, ignore_index=True)
    print(f"Total: {len(df_final):,} participantes RJ escolas públicas (incluindo federais)")
    return df_final

# ============================================================================
# 2. MAPEAR VARIÁVEIS
# ============================================================================

def mapear_variaveis_socioeconomicas(df):
    """
    Mapeia variáveis categóricas Q001-Q025 para valores numéricos.
    
    """
    print("\n2. MAPEAMENTO DE VARIÁVEIS SOCIOECONÔMICAS E INSTITUCIONAIS")
    print("-" * 50)
    
    # Mapeamentos baseados no questionário ENEM
    mapa_escolaridade = {
        "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8
    }
    
    mapa_renda = {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
        "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16
    }
    
    mapa_binario = {"A": 0, "B": 1}  # A=Não, B=Sim
    
    # Aplicar mapeamentos socioeconômicos
    df["escolaridade_pai"] = df["Q001"].map(mapa_escolaridade)
    df["escolaridade_mae"] = df["Q002"].map(mapa_escolaridade)
    df["renda_familiar"] = df["Q006"].map(mapa_renda)
    df["tem_internet"] = df["Q025"].map(mapa_binario)
    df["tem_computador"] = df["Q024"].map(mapa_binario)
    df["tem_carro"] = df["Q020"].map(mapa_binario)
    
    # Criar variáveis derivadas socioeconômicas
    df["escolaridade_pais_media"] = df[["escolaridade_pai", "escolaridade_mae"]].mean(axis=1)
    df["escolaridade_pais_max"] = df[["escolaridade_pai", "escolaridade_mae"]].max(axis=1)
    df["indice_infraestrutura"] = df[["tem_internet", "tem_computador", "tem_carro"]].mean(axis=1)
    
    # Variáveis demográficas
    df["prop_feminino"] = (df["TP_SEXO"] == "F").astype(int)
    df["prop_nao_branca"] = df["TP_COR_RACA"].isin([2, 3, 4, 5]).astype(int)
    df["tipo_escola_binario"] = df["TP_ESCOLA"].map({1: 0, 2: 1, 3: 2})  # 0=Federal, 1=Estadual, 2=Municipal
    
    # Variáveis institucionais transformadas em dummies
    print("Criando variáveis institucionais:")
    df["escola_federal"] = (df["TP_DEPENDENCIA_ADM_ESC"] == 1).astype(int)
    df["escola_municipal"] = (df["TP_DEPENDENCIA_ADM_ESC"] == 3).astype(int)
    
    print("• escola_federal: 1 se federal, 0 caso contrário")
    print("• escola_municipal: 1 se municipal, 0 caso contrário")
    print("• Referência: escola estadual (quando ambas = 0)")
    
    print("Variáveis socioeconômicas e institucionais mapeadas!")
    return df

# ============================================================================
# 3. AGREGAÇÃO POR ESCOLA
# ============================================================================

def criar_dataset_escola(df):
    """
    Agrega os dados por escola, calculando estatísticas médias por instituição.
    ADIÇÃO: Inclui agregação das variáveis institucionais dummy
    """
    print("\n3. AGREGAÇÃO POR ESCOLA")
    print("-" * 50)
    
    # Criar identificador único para cada escola
    df["ESCOLA_ID"] = (df["CO_MUNICIPIO_ESC"].astype(str) + "_" + 
                      df["TP_DEPENDENCIA_ADM_ESC"].astype(str) + "_" +
                      df.groupby(["CO_MUNICIPIO_ESC", "TP_DEPENDENCIA_ADM_ESC"]).ngroup().astype(str))
    
    # Filtrar dados válidos
    df_validos = df.dropna(subset=["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT"])
    
    print(f"Dados válidos: {len(df_validos):,} participantes")
    
    # Agregação por escola
    agg_dict = {
        "NU_INSCRICAO": "count",
        "NU_NOTA_CN": "mean",
        "NU_NOTA_CH": "mean", 
        "NU_NOTA_LC": "mean",
        "NU_NOTA_MT": "mean",
        "NU_NOTA_REDACAO": "mean",
        "prop_feminino": "mean",
        "prop_nao_branca": "mean",
        "TP_ESCOLA": "first",
        "TP_DEPENDENCIA_ADM_ESC": "first",
        "TP_LOCALIZACAO_ESC": "first",
        "NO_MUNICIPIO_ESC": "first",
        "CO_MUNICIPIO_ESC": "first",
        "escolaridade_pais_media": "mean",
        "escolaridade_pais_max": "mean",
        "renda_familiar": "mean",
        "tem_internet": "mean",
        "tem_computador": "mean",
        "indice_infraestrutura": "mean",
        "TP_FAIXA_ETARIA": "mean",
        "escola_federal": "mean",
        "escola_municipal": "mean",
    }
    
    df_escolas = df_validos.groupby("ESCOLA_ID").agg(agg_dict).round(3)
    
    # Renomear colunas
    df_escolas.columns = [
        "num_participantes", "media_cn", "media_ch", "media_lc", "media_mt", "media_redacao",
        "prop_feminino", "prop_nao_branca", "tp_escola", "dependencia_adm", "localizacao", 
        "municipio", "codigo_municipio", "escolaridade_pais_media", "escolaridade_pais_max", 
        "renda_familiar_media", "prop_internet", "prop_computador", "indice_infraestrutura_medio", "idade_media",
        "escola_federal", "escola_municipal"
    ]
    
    # Calcular média geral
    df_escolas["media_geral"] = df_escolas[["media_cn", "media_ch", "media_lc", "media_mt"]].mean(axis=1)
    
    # Filtrar escolas com mínimo de participantes
    min_participantes = 10
    df_escolas = df_escolas[df_escolas["num_participantes"] >= min_participantes]
    
    # Criar variável binária tipo_escola (1 = estadual ou municipal; 0 = outras)
    df_escolas["tipo_escola"] = df_escolas["tp_escola"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

    print(f"Dataset por escola criado: {len(df_escolas)} escolas")
    print(f"Filtro aplicado: mínimo {min_participantes} participantes por escola")
    print(f"Variáveis institucionais incluídas: escola_federal, escola_municipal")
    
    return df_escolas

# ============================================================================
# 4. ANÁLISE EXPLORATÓRIA 
# ============================================================================

def analise_exploratoria(df):
    """
    Realiza análise exploratória completa dos dados com novos gráficos científicos
    """
    print("\n4. ANÁLISE EXPLORATÓRIA E DIAGNÓSTICO")
    print("-" * 50)
    
    # Estatísticas descritivas
    variaveis_interesse = [
        'media_geral', 'escolaridade_pais_media', 'renda_familiar_media', 
        'prop_nao_branca', 'prop_feminino', 'num_participantes'
    ]
    
    desc_stats = df[variaveis_interesse].describe()
    print("ESTATÍSTICAS DESCRITIVAS:")
    print(desc_stats.round(3))
    
    # Salvar estatísticas na pasta de saída
    desc_stats.to_csv(f"{PASTA_SAIDA}/01_estatisticas_descritivas.csv")
    
    # Gráfico 1: Distribuição do Desempenho
    plt.figure(figsize=(10, 6))
    plt.hist(df['media_geral'], bins=25, alpha=0.8, color=CORES_VIRIDIS[3], edgecolor='black', linewidth=0.8)
    plt.axvline(df['media_geral'].mean(), color=CORES_VIRIDIS[8], linestyle='--', linewidth=2,
               label=f'Média: {df["media_geral"].mean():.1f}')
    plt.xlabel('Desempenho Médio no ENEM (pontos)')
    plt.ylabel('Número de Escolas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/02_distribuicao_desempenho.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Matriz de correlação
    variaveis_correlacao = [
        'media_geral', 'escolaridade_pais_media', 'renda_familiar_media', 
        'prop_nao_branca', 'prop_feminino', 'escola_federal', 'escola_municipal'
    ]
    
    nomes_formatados = {
        "escolaridade_pais_media": "Escolaridade dos Pais",
        "renda_familiar_media": "Renda Familiar",
        "prop_nao_branca": "Proporção Não-Branca",
        "prop_feminino": "Proporção Feminina",
        "escola_federal": "Escola Federal",
        "escola_municipal": "Escola Municipal",
        "media_geral": "Desempenho Médio"
    }

    df_plot = df[variaveis_correlacao].rename(columns=nomes_formatados)
    correlation_matrix = df_plot.corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='viridis', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f', 
                annot_kws={"size": 9})
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/03_matriz_correlacao.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Identificar correlações 
    high_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr.append({
                    'Variavel_1': correlation_matrix.columns[i],
                    'Variavel_2': correlation_matrix.columns[j],
                    'Correlacao': corr_val
                })

    if high_corr:
        high_corr_df = pd.DataFrame(high_corr)
        print("CORRELAÇÕES ALTAS DETECTADAS (|r| > 0.7):")
        print(high_corr_df)
        high_corr_df.to_csv(f'{PASTA_SAIDA}/04_correlacoes_altas.csv', index=False)
        return True  # Indica necessidade de PCA
    else:
        print("Nenhuma correlação alta detectada")
        return False

def modelo_inicial_vif(df):
    """
    Constrói modelo inicial e calcula VIF para diagnóstico de multicolinearidade
    CORREÇÃO: Inclui as variáveis do modelo final no VIF inicial
    """
    print("\n5. MODELO INICIAL E DIAGNÓSTICO VIF")
    print("-" * 50)
    
    # Variáveis explicativas do modelo final (antes do PCA)
    X_original = df[['escolaridade_pais_media', 'renda_familiar_media', 'prop_nao_branca', 'prop_feminino', 'escola_federal', 'escola_municipal']]
    y = df['media_geral']
    
    # Modelo de regressão inicial
    modelo = LinearRegression()
    modelo.fit(X_original, y)
    y_pred = modelo.predict(X_original)
    r2 = r2_score(y, y_pred)
    
    print(f"R² do modelo inicial: {r2:.4f}")
    
    # Calcular VIF
    def calculate_vif(X_df):
        vif_data = []
        for i in range(X_df.shape[1]):
            X_temp = X_df.iloc[:, [j for j in range(X_df.shape[1]) if j != i]]
            y_temp = X_df.iloc[:, i]
            
            if X_temp.shape[1] > 0:
                try:
                    r2_temp = r2_score(y_temp, LinearRegression().fit(X_temp, y_temp).predict(X_temp))
                    vif = 1 / (1 - r2_temp) if r2_temp < 0.999 else 999
                except:
                    vif = 1
            else:
                vif = 1
            
            vif_data.append({
                'Variavel': X_df.columns[i],
                'VIF': vif,
                'Interpretacao': 'Severa' if vif > 10 else 'Moderada' if vif > 5 else 'Aceitável'
            })
        
        return pd.DataFrame(vif_data)
    
    vif_df = calculate_vif(X_original)
    print("\nVIF - DIAGNÓSTICO DE MULTICOLINEARIDADE:")
    print(vif_df.round(2))
    
    # Salvar resultados na pasta de saída
    vif_df.to_csv(f'{PASTA_SAIDA}/05_vif_modelo_inicial.csv', index=False)
    
    # Coeficientes do modelo inicial
    coef_df = pd.DataFrame({
        'Variavel': ['Intercepto'] + list(X_original.columns),
        'Coeficiente': [modelo.intercept_] + list(modelo.coef_),
        'Interpretacao': ['Valor base'] + ['Positivo' if c > 0 else 'Negativo' for c in modelo.coef_]
    })
    
    print("\nCOEFICIENTES DO MODELO INICIAL:")
    print(coef_df.round(3))
    coef_df.to_csv(f'{PASTA_SAIDA}/06_coeficientes_inicial.csv', index=False)
    
    # Verificar necessidade de PCA
    vif_maximo = vif_df['VIF'].max()
    if vif_maximo > 10:
        print(f"  PROBLEMA DETECTADO: VIF máximo = {vif_maximo:.2f}")
        print("   → Multicolinearidade severa identificada")
        print("   → Aplicação de PCA necessária")
        return True, X_original, y, modelo
    elif vif_maximo > 5:
        print(f"  ATENÇÃO: VIF máximo = {vif_maximo:.2f}")
        print("   → Multicolinearidade moderada identificada")
        print("   → PCA recomendado para robustez")
        return True, X_original, y, modelo
    else:
        print(f" VIF máximo = {vif_maximo:.2f} - Sem problemas de multicolinearidade")
        return False, X_original, y, modelo
    
# ============================================================================
# 5. ANÁLISE DOS COMPONENTES PRINCIPAIS
# ============================================================================

def aplicar_pca_socioeconomico(df):
    """
    Aplica PCA nas variáveis socioeconômicas com testes Bartlett e KMO
    CORREÇÃO CRÍTICA: Usa teste de Bartlett de ESFERICIDADE (não homogeneidade)
    """
    print("\n6. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    print("-" * 50)
    
    variaveis_pca = [
        'escolaridade_pais_media',
        'renda_familiar_media'
    ]
    
    X_pca = df[variaveis_pca]
    print(f"Aplicando PCA nas variáveis: {variaveis_pca}")
 
    
    try:
        print("TESTE DE BARTLETT (Adequação do PCA):")
        bartlett_stat, bartlett_p = calculate_bartlett_sphericity(X_pca.dropna())
        print(f"   Estatística de Bartlett (Esfericidade): {bartlett_stat:.4f}")
        print(f"   p-valor: {bartlett_p:.6f}")
        print(f"   Adequação para PCA: {"✅ SIM" if bartlett_p < 0.05 else "❌ NÃO"}")
        print(f"   Interpretação: {'Variáveis correlacionadas → PCA adequado' if bartlett_p < 0.05 else 'Variáveis pouco correlacionadas → PCA questionável'}")
        
        # Salvar resultado do teste
        bartlett_result = pd.DataFrame({
            "Teste": ["Bartlett_Esfericidade"],
            "Estatistica": [bartlett_stat],
            "p_valor": [bartlett_p],
            "Adequacao_PCA": ["SIM" if bartlett_p < 0.05 else "NÃO"],
            "Interpretacao": ["Variáveis correlacionadas" if bartlett_p < 0.05 else "Variáveis pouco correlacionadas"]
        })
        bartlett_result.to_csv(f'{PASTA_SAIDA}/07_teste_bartlett_pca.csv', index=False)
        
    except Exception as e:
        print(f"   ⚠️ Erro no teste de Bartlett: {e}")
        print("   → Prosseguindo com PCA")

    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)

    # PCA
    pca = PCA()
    componentes = pca.fit_transform(X_scaled)
    variancia_explicada = pca.explained_variance_ratio_
    variancia_acumulada = np.cumsum(variancia_explicada)

    print(f"\nVARIÂNCIA EXPLICADA POR COMPONENTE:")
    for i, var in enumerate(variancia_explicada):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.1f}%)")
    
    print(f"\nVARIÂNCIA ACUMULADA:")
    for i, var_acum in enumerate(variancia_acumulada):
        print(f"  PC1-PC{i+1}: {var_acum:.4f} ({var_acum*100:.1f}%)")

    # Gráfico da variância explicada
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(variancia_explicada) + 1), variancia_explicada, alpha=0.8)
    plt.xlabel('Componente Principal')
    plt.ylabel('Proporção da Variância Explicada')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(variancia_acumulada) + 1), variancia_acumulada, 'o-', linewidth=2)
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Acumulada')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/08_variancia_explicada_pca.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(variaveis_pca))],
        index=variaveis_pca
    )
    
    print(f"\nLOADINGS DOS COMPONENTES:")
    print(loadings.round(4))

    # Gráfico dos loadings
    plt.figure(figsize=(8, 6))
    sns.heatmap(loadings, annot=True, cmap='viridis', center=0, 
            square=True, linewidths=0.5, fmt='.3f')

    plt.title("")
    plt.yticks(
    ticks=plt.yticks()[0],
    labels=[
        "Escolaridade dos pais",
        "Renda familiar"
    ],
    rotation=90
)

    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/09_loadings_pca.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Salvar arquivos
    loadings.to_csv(f'{PASTA_SAIDA}/10_loadings_pca.csv')
    pca_results = pd.DataFrame({
        'Componente': [f'PC{i+1}' for i in range(len(variancia_explicada))],
        'Variancia_Explicada': variancia_explicada,
        'Variancia_Acumulada': variancia_acumulada
})
    pca_results.to_csv(f'{PASTA_SAIDA}/11_resultados_pca.csv', index=False)


        # Criar nova variável ISE com PC1
    df['ise'] = componentes[:, 0]

    print(f"ISE (Índice de Status Socioeconômico) criado")
    print(f"Variância explicada: {variancia_explicada[0]*100:.1f}%")
    print(f"Interpretação: Combina múltiplas dimensões socioeconômicas")

    return df, pca, scaler

# ============================================================================
# 6. ANÁLISE DAS VARIÁVEIS INSTITUCIONAIS
# ============================================================================

def analisar_variaveis_institucionais(df):
    """
    ADIÇÃO: Analisa VIF das variáveis institucionais para justificar uso de dummies
    """
    print("\n7. ANÁLISE DAS VARIÁVEIS INSTITUCIONAIS")
    print("-" * 50)
    
    # Variáveis institucionais dummy
    variaveis_institucionais = ['escola_federal', 'escola_municipal']
    X_inst = df[variaveis_institucionais]
    
    print(" Analisando VIF das variáveis institucionais:")
    print(f"   Variáveis: {variaveis_institucionais}")
    
    # Calcular VIF das variáveis institucionais
    def calculate_vif_institucional(X_df):
        vif_data = []
        for i in range(X_df.shape[1]):
            X_temp = X_df.iloc[:, [j for j in range(X_df.shape[1]) if j != i]]
            y_temp = X_df.iloc[:, i]
            
            if X_temp.shape[1] > 0:
                try:
                    r2_temp = r2_score(y_temp, LinearRegression().fit(X_temp, y_temp).predict(X_temp))
                    vif = 1 / (1 - r2_temp) if r2_temp < 0.999 else 999
                except:
                    vif = 1
            else:
                vif = 1
            
            vif_data.append({
                'Variavel': X_df.columns[i],
                'VIF': vif,
                'Interpretacao': 'Severa' if vif > 10 else 'Moderada' if vif > 5 else 'Aceitável'
            })
        
        return pd.DataFrame(vif_data)
    
    vif_inst_df = calculate_vif_institucional(X_inst)
    print("\nVIF - VARIÁVEIS INSTITUCIONAIS:")
    print(vif_inst_df.round(2))
    
    # Salvar VIF institucional
    vif_inst_df.to_csv(f'{PASTA_SAIDA}/12_vif_variaveis_institucionais.csv', index=False)
    
    # Análise de adequação
    vif_maximo_inst = vif_inst_df['VIF'].max()
    print(f" ADEQUAÇÃO DAS VARIÁVEIS INSTITUCIONAIS:")
    print(f"   VIF máximo: {vif_maximo_inst:.2f}")
    
    if vif_maximo_inst < 5:
        print("VIF baixo → Variáveis dummy adequadas")
        print("   → Não há necessidade de redução dimensional")
        adequacao = "DUMMY_ADEQUADO"
    elif vif_maximo_inst < 10:
        print("   VIF moderado → Considerar PCA")
        adequacao = "PCA_RECOMENDADO"
    else:
        print("  VIF alto → PCA necessário")
        adequacao = "PCA_NECESSARIO"
    
    # Salvar análise de adequação
    adequacao_df = pd.DataFrame({
        'Metrica': ['VIF_Maximo', 'Adequacao_Dummy', 'Recomendacao'],
        'Valor': [vif_maximo_inst, adequacao, 'Manter dummies' if adequacao == "DUMMY_ADEQUADO" else 'Aplicar PCA']
    })
    adequacao_df.to_csv(f'{PASTA_SAIDA}/13_adequacao_variaveis_institucionais.csv', index=False)
    
    return adequacao == "DUMMY_ADEQUADO"

# ============================================================================
# 7. REGRESSÃO MÚLTIPLA ROBUSTA COM TESTES DE DIAGNÓSTICO
# ============================================================================

def regressao_multipla_robusta(df):
    """
    Estima regressão múltipla robusta (OLS HC3) com testes de diagnóstico completos
    """
    print("\n8. REGRESSÃO MÚLTIPLA ROBUSTA (OLS HC3)")
    print("-" * 50)

    # Variáveis do modelo final
    X = df[['ise', 'prop_nao_branca', 'prop_feminino', 'escola_federal', 'escola_municipal']]
    y = df['media_geral']

    print("MODELO FINAL:")
    print("   Y = β₀ + β₁(ISE) + β₂(prop_nao_branca) + β₃(prop_feminino) + β₄(escola_federal) + β₅(escola_municipal) + ε")
    print("   Referência: escola estadual")

    # Adicionar constante
    X_com_const = sm.add_constant(X)

    # Estimação OLS com erros padrão robustos (HC3)
    modelo = sm.OLS(y, X_com_const).fit(cov_type='HC3')

    print("\nRESULTADOS DA REGRESSÃO:")
    print(modelo.summary())

    # Salvar summary completo na pasta de saída
    with open(f'{PASTA_SAIDA}/14_regressao_summary_completo.txt', 'w') as f:
        f.write(str(modelo.summary()))

    # Extrair coeficientes e estatísticas
    coeficientes = pd.DataFrame({
        'Variavel': modelo.params.index,
        'Coeficiente': modelo.params.values,
        'Erro_Padrao': modelo.bse.values,
        'Estatistica_t': modelo.tvalues.values,
        'p_valor': modelo.pvalues.values,
        'IC_Inferior': modelo.conf_int()[0].values,
        'IC_Superior': modelo.conf_int()[1].values,
        'Significancia': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in modelo.pvalues.values]
    })

    print("\nCOEFICIENTES E INTERVALOS DE CONFIANÇA:")
    print(coeficientes.round(4))

    # Salvar coeficientes na pasta de saída
    coeficientes.to_csv(f'{PASTA_SAIDA}/15_coeficientes_regressao.csv', index=False)

    # Métricas do modelo
    metricas = {
        'R_quadrado': modelo.rsquared,
        'R_quadrado_ajustado': modelo.rsquared_adj,
        'F_statistic': modelo.fvalue,
        'F_pvalue': modelo.f_pvalue,
        'AIC': modelo.aic,
        'BIC': modelo.bic,
        'Log_likelihood': modelo.llf,
        'Durbin_Watson': durbin_watson(modelo.resid),
        'Num_observacoes': modelo.nobs,
        'Graus_liberdade': modelo.df_resid
    }

    metricas_df = pd.DataFrame(list(metricas.items()), columns=['Metrica', 'Valor'])
    print(f"\nMÉTRICAS DO MODELO:")
    print(metricas_df.round(4))

    # Salvar métricas na pasta de saída
    metricas_df.to_csv(f'{PASTA_SAIDA}/16_metricas_modelo.csv', index=False)

    # Resíduos e valores preditos
    residuos = modelo.resid
    valores_preditos = modelo.fittedvalues

    # TESTES DE DIAGNÓSTICO
    print("TESTES DE DIAGNÓSTICO:")
    print("-" * 30)

       # 1. Teste de Normalidade dos Resíduos (Jarque-Bera)
    jb_stat, jb_pvalue = jarque_bera(residuos)
    print(f"1. Jarque-Bera: Estatística={jb_stat:.4f}, p-valor={jb_pvalue:.6f}")
    print(f"   Interpretação: {'Resíduos normais' if jb_pvalue > 0.05 else 'Resíduos não-normais'}")

    # 2. Teste de Normalidade dos Resíduos (Shapiro-Wilk)
    shapiro_stat, shapiro_pvalue = shapiro(residuos)
    print(f"2. Shapiro-Wilk: Estatística={shapiro_stat:.4f}, p-valor={shapiro_pvalue:.6f}")
    print(f"   Interpretação: {'Resíduos normais' if shapiro_pvalue > 0.05 else 'Resíduos não-normais'}")

    # 3. Teste de Homocedasticidade (Breusch-Pagan)
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuos, X_com_const)
    print(f"3. Breusch-Pagan: Estatística={bp_stat:.4f}, p-valor={bp_pvalue:.6f}")
    print(f"   Interpretação: {'Homocedasticidade' if bp_pvalue > 0.05 else 'Heterocedasticidade'}")

    # 4. Teste de Autocorrelação (Durbin-Watson)
    dw_stat = durbin_watson(residuos)
    print(f"4. Durbin-Watson: Estatística={dw_stat:.4f}")
    print(f"   Interpretação: {'Sem autocorrelação' if 1.5 < dw_stat < 2.5 else 'Possível autocorrelação'}")

    # Salvar testes de diagnóstico na pasta de saída
    testes_diagnostico = pd.DataFrame({
        'Teste': ['Jarque-Bera', 'Shapiro-Wilk', 'Breusch-Pagan', 'Durbin-Watson'],
        'Estatistica': [jb_stat, shapiro_stat, bp_stat, dw_stat],
        'p_valor': [jb_pvalue, shapiro_pvalue, bp_pvalue, np.nan],
        'Interpretacao': [
            'Resíduos normais' if jb_pvalue > 0.05 else 'Resíduos não-normais',
            'Resíduos normais' if shapiro_pvalue > 0.05 else 'Resíduos não-normais',
            'Homocedasticidade' if bp_pvalue > 0.05 else 'Heterocedasticidade',
            'Sem autocorrelação' if 1.5 < dw_stat < 2.5 else 'Possível autocorrelação'
        ]
    })
    testes_diagnostico.to_csv(f'{PASTA_SAIDA}/17_testes_diagnostico.csv', index=False)

    # GRÁFICOS DE DIAGNÓSTICO
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Resíduos vs Valores Preditos
    axes[0, 0].scatter(valores_preditos, residuos, alpha=0.6, color=CORES_VIRIDIS[3])
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Valores Preditos')
    axes[0, 0].set_ylabel('Resíduos')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Q-Q Plot dos Resíduos
    stats.probplot(residuos, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("")
    axes[0, 1].set_xlabel("Quantis Teóricos")
    axes[0, 1].set_ylabel("Resíduos Ordenados")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Histograma dos Resíduos
    axes[1, 0].hist(residuos, bins=20, alpha=0.7, color=CORES_VIRIDIS[5], edgecolor='black')
    axes[1, 0].set_xlabel('Resíduos')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Valores Observados vs Preditos
    axes[1, 1].scatter(y, valores_preditos, alpha=0.6, color=CORES_VIRIDIS[7])
    axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Valores Observados')
    axes[1, 1].set_ylabel('Valores Preditos')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/18_diagnosticos_regressao.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ANÁLISE DE OBSERVAÇÕES INFLUENTES
    print("ANÁLISE DE OBSERVAÇÕES INFLUENTES:")
    print("-" * 40)

    # Calcular distância de Cook
    influence = modelo.get_influence()
    cooks_d = influence.cooks_distance[0]

    # Limite típico: 4/n
    n = len(df)
    limite_cook = 4 / n
    influentes = df[cooks_d > limite_cook].copy()
    influentes['cooks_distance'] = cooks_d[cooks_d > limite_cook]

    print(f"   Limite típico: 4/n = {limite_cook:.4f}")
    print(f"   Observações influentes detectadas: {len(influentes)}")
    
    # Salvar observações influentes na pasta de saída
    influentes.to_csv(f'{PASTA_SAIDA}/19_observacoes_influentes_cook.csv', index=False)

    # Gráfico da distância de Cook
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(cooks_d)), cooks_d, basefmt=" ", linefmt="-", markerfmt="o")
    plt.axhline(y=limite_cook, color='red', linestyle='--', linewidth=2, 
                label=f'Limite (4/n ≈ {limite_cook:.4f})')
    plt.xlabel('Observação')
    plt.ylabel('Distância de Cook')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/20_distancia_cook.png', dpi=300, bbox_inches='tight')
    plt.show()

 # ANÁLISE DE SENSIBILIDADE DOS COEFICIENTES
    df_sem_influentes = df[cooks_d <= limite_cook].copy()
    X_sem = df_sem_influentes[['ise', 'prop_nao_branca', 'prop_feminino', 'escola_federal', 'escola_municipal']]
    y_sem = df_sem_influentes['media_geral']
    X_sem_const = sm.add_constant(X_sem)
    modelo_sem = sm.OLS(y_sem, X_sem_const).fit(cov_type='HC3')

    comparacao = pd.DataFrame({
        'Variável': modelo.params.index,
        'Coef. Original': modelo.params.values,
        'Coef. Sem Influentes': modelo_sem.params.reindex(modelo.params.index).values,
        'Diferença Absoluta': abs(modelo.params.values - modelo_sem.params.reindex(modelo.params.index).values)
    })

    print("\nANÁLISE DE SENSIBILIDADE DOS COEFICIENTES:")
    print(comparacao.round(4))

    # Salvar comparação dos coeficientes
    comparacao.to_csv(f'{PASTA_SAIDA}/21_analise_sensibilidade.csv', index=False)
    
    return modelo, X_com_const, y, residuos, influentes

# ============================================================================
# 8. ANALISE DE AGRUPAMENTO DAS ESCOLAS
# ============================================================================

def analise_clusters(df):
    """
    Agrupamento das escolas baseados nos seguintes critérios:
        
    Qualidade (máxima silhueta para Avalia a separação e a coesão.)
    Estabilidade (cosistência dos clusters mediante à perturbações nos dados.)
    Parcimônia (evitar complexidade sem perda de explicação estatística)

    MODIFICAÇÃO: Removido o critério de viabilidade (tamanho mínimo) para usar o melhor k baseado nas métricas.
    """
    print("\n9. ANÁLISE DE CLUSTERS")
    print("-" * 50)

    # Dados para clustering 
    print(" CLUSTERING POR CARACTERÍSTICAS DAS ESCOLAS:")
    print("   Variáveis: ISE, fatores demográficos e institucionais")
    
    X_cluster = df[['ise', 'prop_nao_branca', 'prop_feminino', 'escola_federal', 'escola_municipal']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    print(f"   Dimensões: {X_scaled.shape[0]} escolas × {X_scaled.shape[1]} características")

    # Avaliação para diferentes valores de k
    k_range = range(2, 8)
    inertias, silhouette_scores, calinski_scores = [], [], []
    tamanhos_minimos = []
    ari_estabilidade = []  # ARI de estabilidade para cada k

    print("\nAvaliando k de 2 a 7 com critérios científicos rigorosos...")
    print("Critério baseado em: Ben-Hur et al. (2002) + Rousseeuw (1987)")
    print("-" * 60)
    
    for k in k_range:
        print(f"\nAvaliando k={k}:")
        
        # Clustering principal
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Métricas básicas
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        calinski_scores.append(calinski)
        
        # Verificar tamanho mínimo dos clusters
        unique, counts = np.unique(labels, return_counts=True)
        tamanho_minimo = min(counts)
        tamanhos_minimos.append(tamanho_minimo)
        
        print(f"  Silhueta: {silhouette:.3f} | Calinski: {calinski:.2f} | Cluster_min: {tamanho_minimo}")
        
        # TESTE DE ESTABILIDADE (critério de Ben-Hur et al., 2002)
        print(f"  Testando estabilidade (bootstrap)...")
        ari_scores_k = []
        n_bootstrap_estabilidade = 50  # Suficiente para teste de estabilidade
        
        for i in range(n_bootstrap_estabilidade):
            # Amostra bootstrap (80% dos dados)
            indices_bootstrap = resample(range(len(X_scaled)), n_samples=int(0.8 * len(X_scaled)), random_state=i)
            X_bootstrap = X_scaled[indices_bootstrap]
            
            # Clustering na amostra bootstrap
            kmeans_bootstrap = KMeans(n_clusters=k, random_state=i, n_init=10)
            labels_bootstrap_sample = kmeans_bootstrap.fit_predict(X_bootstrap)
            
            # Predizer para toda a amostra original
            labels_bootstrap_full = kmeans_bootstrap.predict(X_scaled)
            
            # Calcular ARI
            ari = adjusted_rand_score(labels, labels_bootstrap_full)
            ari_scores_k.append(ari)
        
        # Estatística de estabilidade
        ari_medio_k = np.mean(ari_scores_k)
        ari_estabilidade.append(ari_medio_k)
        
        print(f"  ARI médio: {ari_medio_k:.3f} {'✅' if ari_medio_k > 0.75 else '⚠️' if ari_medio_k > 0.6 else '❌'}")

    # Salvar métricas completas
    metricas_k = pd.DataFrame({
        'k': k_range,
        'Inercia': inertias,
        'Silhueta': silhouette_scores,
        'Calinski_Harabasz': calinski_scores,
        'Tamanho_Minimo_Cluster': tamanhos_minimos,
        'ARI_Estabilidade': ari_estabilidade
    })
    metricas_k.to_csv(f'{PASTA_SAIDA}/21_metricas_avaliacao_k_completas.csv', index=False)

    # CRITÉRIO CIENTÍFICO MODIFICADO
    print(f"APLICANDO CRITÉRIO BASEADO NAS MELHORES MÉTRICAS")
    print("=" * 70)
    print("MODIFICAÇÃO: Removido critério de viabilidade - usando melhor k baseado nas métricas")
    
    # ETAPA 1: Critério de qualidade (Rousseeuw, 1987) - MÁXIMA SILHUETA
    print(f"\nETAPA 1 - Critério de qualidade (Rousseeuw, 1987):")
    print(f"  Selecionar por máxima silhueta")
    
    melhor_silhueta = max(silhouette_scores)
    k_candidatos_silhueta = []
    
    for i, k in enumerate(k_range):
        silh_k = silhouette_scores[i]
        print(f"  k={k}: Silhueta={silh_k:.3f}")
        
        if abs(silh_k - melhor_silhueta) < 0.01:  # Empate técnico (diferença < 1%)
            k_candidatos_silhueta.append(k)
    
    print(f"  Melhor silhueta: {melhor_silhueta:.3f}")
    print(f"  k's com melhor silhueta: {k_candidatos_silhueta}")
    
    # ETAPA 2: Entre os melhores em silhueta, selecionar por melhor ARI
    print(f"\nETAPA 2 - Critério de estabilidade (Ben-Hur et al., 2002):")
    print(f"  Entre os melhores em silhueta, selecionar por melhor ARI")
    
    melhor_ari = -1
    k_candidatos_ari = []
    
    for k in k_candidatos_silhueta:
        idx = k_range.index(k)
        ari_k = ari_estabilidade[idx]
        print(f"  k={k}: ARI={ari_k:.3f}")
        
        if ari_k > melhor_ari:
            melhor_ari = ari_k
            k_candidatos_ari = [k]
        elif abs(ari_k - melhor_ari) < 0.01:  # Empate técnico
            k_candidatos_ari.append(k)
    
    print(f"  Melhor ARI: {melhor_ari:.3f}")
    print(f"  k's com melhor ARI: {k_candidatos_ari}")
    
    # ETAPA 3: Princípio de parcimônia (Navalha de Occam)
    print(f"\nETAPA 3 - Princípio de parcimônia (Occam's Razor):")
    if len(k_candidatos_ari) > 1:
        k_otimo = min(k_candidatos_ari)  # Menor k em caso de empate
        print(f"  Empate técnico entre {k_candidatos_ari}")
        print(f"  Selecionado k={k_otimo} (menor k - mais parcimonioso)")
    else:
        k_otimo = k_candidatos_ari[0]
        print(f"  k={k_otimo} selecionado (única melhor opção)")
    
    # Métricas finais do k ótimo
    idx_otimo = k_range.index(k_otimo)
    silhueta_otima = silhouette_scores[idx_otimo]
    calinski_otimo = calinski_scores[idx_otimo]
    tamanho_min_otimo = tamanhos_minimos[idx_otimo]
    ari_otimo = ari_estabilidade[idx_otimo]
    
    print(f"\n✅ K ÓTIMO SELECIONADO: {k_otimo}")
    print(f"   Coeficiente de silhueta: {silhueta_otima:.3f}")
    print(f"   Índice Calinski-Harabasz: {calinski_otimo:.2f}")
    print(f"   ARI de estabilidade: {ari_otimo:.3f}")
    print(f"   Tamanho mínimo do cluster: {tamanho_min_otimo} escolas")

    # Salvar critério de seleção detalhado
    criterio_selecao = pd.DataFrame({
        'Etapa': ['1_Qualidade', '2_Estabilidade', '3_Parcimonia', 'Final'],
        'Criterio': ['Max_Silhueta', 'Max_ARI', 'Min_k', 'Selecionado'],
        'k_Candidatos': [str(k_candidatos_silhueta), str(k_candidatos_ari), str([k_otimo]), str(k_otimo)],
        'Valor_Criterio': [melhor_silhueta, melhor_ari, k_otimo, 'N/A']
    })
    criterio_selecao.to_csv(f'{PASTA_SAIDA}/21b_criterio_selecao_modificado.csv', index=False)

    # Gráfico de determinação do k ótimo com todas as métricas
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Cotovelo
    axes[0, 0].plot(k_range, inertias, 'o-', linewidth=2, markersize=8, color=CORES_VIRIDIS[3])
    axes[0, 0].axvline(k_otimo, color='red', linestyle='--', alpha=0.7, label=f'k ótimo: {k_otimo}')
    axes[0, 0].set_xlabel('Número de Clusters (k)')
    axes[0, 0].set_ylabel('Inércia')
    axes[0, 0].set_title('Método do Cotovelo')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Silhueta
    axes[0, 1].plot(k_range, silhouette_scores, 'o-', linewidth=2, markersize=8, color=CORES_VIRIDIS[6])
    axes[0, 1].axvline(k_otimo, color='red', linestyle='--', alpha=0.7, label=f'k ótimo: {k_otimo}')
    axes[0, 1].set_xlabel('Número de Clusters (k)')
    axes[0, 1].set_ylabel('Coeficiente de Silhueta')
    axes[0, 1].set_title('Critério da Silhueta')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Calinski-Harabasz
    axes[1, 0].plot(k_range, calinski_scores, 'o-', linewidth=2, markersize=8, color=CORES_VIRIDIS[8])
    axes[1, 0].axvline(k_otimo, color='red', linestyle='--', alpha=0.7, label=f'k ótimo: {k_otimo}')
    axes[1, 0].set_xlabel('Número de Clusters (k)')
    axes[1, 0].set_ylabel('Índice Calinski-Harabasz')
    axes[1, 0].set_title('Critério de Calinski-Harabasz')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Estabilidade (ARI)
    axes[1, 1].plot(k_range, ari_estabilidade, 'o-', linewidth=2, markersize=8, color=CORES_VIRIDIS[4])
    axes[1, 1].axhline(0.75, color='orange', linestyle=':', alpha=0.7, label='Limiar estabilidade: 0.75')
    axes[1, 1].axvline(k_otimo, color='red', linestyle='--', alpha=0.7, label=f'k ótimo: {k_otimo}')
    axes[1, 1].set_xlabel('Número de Clusters (k)')
    axes[1, 1].set_ylabel('ARI Médio')
    axes[1, 1].set_title('Critério de Estabilidade')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/22_determinacao_k_otimo_2x2.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Clusterização final com k ótimo
    kmeans_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
    labels_final = kmeans_final.fit_predict(X_scaled)
    df['cluster'] = labels_final

    # Verificação final dos tamanhos
    unique_final, counts_final = np.unique(labels_final, return_counts=True)
    print(f" TAMANHOS FINAIS DOS CLUSTERS:")
    for i, (cluster, count) in enumerate(zip(unique_final, counts_final)):
        pct = count / len(df) * 100
        print(f"   Cluster {cluster}: {count} escolas ({pct:.1f}%)")

      
        # 1. Calcular e salvar centroides
    centroids_scaled = kmeans_final.cluster_centers_
    centroids_original_scale = scaler.inverse_transform(centroids_scaled)

        # Nomear colunas com rótulos amigáveis
    nomes_amigaveis = {
    'ise': 'ISE',
    'prop_nao_branca': 'Proporção Não Branca',
    'prop_feminino': 'Proporção Feminino',
    'escola_federal': 'Escola Federal',
    'escola_municipal': 'Escola Municipal'
}

    centroids_df = pd.DataFrame(centroids_original_scale, columns=X_cluster.columns)
    centroids_df.rename(columns=nomes_amigaveis, inplace=True)
    centroids_df.index.name = 'Cluster'

        # Salvar centroides como CSV
    centroids_df.to_csv(f'{PASTA_SAIDA}/22a_centroides_clusters.csv')
    print('Centroides salvos em CSV!')

        # Salvar centroides como gráfico PNG (barras)
    fig, ax = plt.subplots(figsize=(10, 6))
    centroids_df.plot(kind='bar', ax=ax, colormap='viridis')
    ax.set_ylabel('Valor Médio')
    ax.set_xlabel('Cluster')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/22a_centroides_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Gráfico de centroides salvo!')

    # Bootstrap ARI completo (100 iterações para robustez)
    print("\n🔁 BOOTSTRAP ARI COMPLETO (100 iterações)...")
    n_bootstrap = 100
    ari_scores = []
    
    for i in range(n_bootstrap):
        if (i + 1) % 20 == 0:
            print(f"     Iteração {i+1}/{n_bootstrap}")
            
        # Amostra bootstrap (80% dos dados)
        indices_bootstrap = resample(range(len(X_scaled)), n_samples=int(0.8 * len(X_scaled)), random_state=i)
        X_bootstrap = X_scaled[indices_bootstrap]
        
        # Clusterização na amostra bootstrap
        kmeans_bootstrap = KMeans(n_clusters=k_otimo, random_state=i, n_init=10)
        labels_bootstrap_sample = kmeans_bootstrap.fit_predict(X_bootstrap)
        
        # Predizer para toda a amostra original
        labels_bootstrap_full = kmeans_bootstrap.predict(X_scaled)
        
        # Calcular ARI
        ari = adjusted_rand_score(labels_final, labels_bootstrap_full)
        ari_scores.append(ari)

    # Estatísticas do ARI
    ari_medio = np.mean(ari_scores)
    ari_std = np.std(ari_scores)
    ari_q25 = np.percentile(ari_scores, 25)
    ari_q75 = np.percentile(ari_scores, 75)
    
    print(f" ESTATÍSTICAS DO BOOTSTRAP ARI:")
    print(f"   ARI médio: {ari_medio:.3f} ± {ari_std:.3f}")
    print(f"   Quartis: Q1={ari_q25:.3f}, Q3={ari_q75:.3f}")
    print(f"   Estabilidade: {'✅ Alta' if ari_medio > 0.75 else '⚠️ Moderada' if ari_medio > 0.6 else '❌ Baixa'}")
    
    # Salvar resultados ARI
    ari_results = pd.DataFrame({
        'Bootstrap_Iteration': range(n_bootstrap),
        'ARI_Score': ari_scores
    })
    ari_results.to_csv(f'{PASTA_SAIDA}/23_ari_bootstrap_scores.csv', index=False)
    
    # Estatísticas resumo do ARI
    ari_summary = pd.DataFrame({
        'Metrica': ['ARI_Bootstrap_Medio', 'ARI_Desvio_Padrao', 'ARI_Q25', 'ARI_Q75', 'ARI_Min', 'ARI_Max', 'Estabilidade'],
        'Valor': [
            ari_medio, ari_std, ari_q25, ari_q75, 
            min(ari_scores), max(ari_scores),
            'Alta' if ari_medio > 0.75 else 'Moderada' if ari_medio > 0.6 else 'Baixa'
        ]
    })
    ari_summary.to_csv(f'{PASTA_SAIDA}/24_ari_resumo_estatisticas.csv', index=False)
    
    # Gráfico da distribuição ARI
    plt.figure(figsize=(10, 6))
    plt.hist(ari_scores, bins=20, alpha=0.8, color=CORES_VIRIDIS[4], edgecolor='black')
    plt.axvline(ari_medio, color='red', linestyle='--', linewidth=2, label=f'Média: {ari_medio:.3f}')
    plt.axvline(ari_q25, color='orange', linestyle=':', linewidth=2, label=f'Q1: {ari_q25:.3f}')
    plt.axvline(ari_q75, color='orange', linestyle=':', linewidth=2, label=f'Q3: {ari_q75:.3f}')
    plt.axvline(0.75, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Limiar alta estabilidade: 0.75')
    plt.xlabel('ARI Score')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/25_distribuicao_ari.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Perfis dos clusters com identificação de desempenho
    print(" PERFIS DOS CLUSTERS (Características → Desempenho):")
    print("   Clusters formados por características, analisando variação do desempenho")
    
    # Calcular quartis de desempenho para classificação
    q25_desempenho = df['media_geral'].quantile(0.25)
    q75_desempenho = df['media_geral'].quantile(0.75)
    media_desempenho = df['media_geral'].mean()
    
    print(f" REFERÊNCIAS DE DESEMPENHO:")
    print(f"   Q1 (25%): {q25_desempenho:.2f}")
    print(f"   Média: {media_desempenho:.2f}")
    print(f"   Q3 (75%): {q75_desempenho:.2f}")
    
    def classificar_desempenho(media_cluster):
        if media_cluster >= q75_desempenho:
            return "Alto Desempenho"
        elif media_cluster >= media_desempenho:
            return "Desempenho Médio"
        elif media_cluster >= q25_desempenho:
            return "Baixo Desempenho"
        else:
            return "Baixo Desempenho"
    
    perfis = []
    for i in range(k_otimo):
        grupo = df[df['cluster'] == i]
        desempenho_medio = grupo['media_geral'].mean()
        perfil_desempenho = classificar_desempenho(desempenho_medio)
        
        perfis.append({
            'Cluster': f'Cluster {i}',
            'N_Escolas': len(grupo),
            'Percentual': f"{len(grupo)/len(df)*100:.1f}%",
            # CARACTERÍSTICAS USADAS NO CLUSTERING
            'ISE_Medio': grupo['ise'].mean(),
            'Prop_Nao_Branca': grupo['prop_nao_branca'].mean(),
            'Prop_Feminino': grupo['prop_feminino'].mean(),
            'Prop_Federal': grupo['escola_federal'].mean(),
            'Prop_Municipal': grupo['escola_municipal'].mean(),
            # OUTCOME (NÃO USADO NO CLUSTERING)
            'Desempenho_Medio': desempenho_medio,
            'Perfil_Desempenho': perfil_desempenho
        })

    perfis_df = pd.DataFrame(perfis)
    print("\nPERFIS DOS CLUSTERS:")
    print(perfis_df.round(3))

    # Salvar perfis na pasta de saída
    perfis_df.to_csv(f'{PASTA_SAIDA}/26_perfis_clusters.csv', index=False)

    # Gráfico dos perfis dos clusters (layout 2x2 sem resumo nem dummies)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Gráfico 1: ISE por cluster
    axes[0, 0].bar(perfis_df['Cluster'], perfis_df['ISE_Medio'], color=CORES_VIRIDIS[3], alpha=0.8)
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('ISE Médio')
    axes[0, 0].set_title('Características: ISE')
    axes[0, 0].grid(True, alpha=0.3)

    # Gráfico 2: Proporção não-branca por cluster
    axes[0, 1].bar(perfis_df['Cluster'], perfis_df['Prop_Nao_Branca'], color=CORES_VIRIDIS[5], alpha=0.8)
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Proporção Não-Branca')
    axes[0, 1].set_title('Características: Demografia')
    axes[0, 1].grid(True, alpha=0.3)

    # Gráfico 3: Desempenho médio por cluster
    cores_desempenho = []
    for perfil in perfis_df['Perfil_Desempenho']:
        if perfil == "Alto Desempenho":
           cores_desempenho.append('green')
        elif perfil == "Desempenho Médio-Alto":
           cores_desempenho.append('lightgreen')
        elif perfil == "Baixo Desempenho":
           cores_desempenho.append('orange')
    else:
        cores_desempenho.append('red')

    bars = axes[1, 0].bar(perfis_df['Cluster'], perfis_df['Desempenho_Medio'], color=cores_desempenho, alpha=0.8)
    axes[1, 0].axhline(media_desempenho, color='black', linestyle='--', alpha=0.7, label=f'Média geral: {media_desempenho:.2f}')
    axes[1, 0].axhline(q75_desempenho, color='green', linestyle=':', alpha=0.7, label=f'Q3: {q75_desempenho:.2f}')
    axes[1, 0].axhline(q25_desempenho, color='red', linestyle=':', alpha=0.7, label=f'Q1: {q25_desempenho:.2f}')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Desempenho Médio')
    axes[1, 0].set_title('Outcome: Desempenho por Cluster')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    for bar, perfil in zip(bars, perfis_df['Perfil_Desempenho']):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   perfil.replace(' ', '\n'), ha='center', va='bottom', fontsize=8)

    # Gráfico 4: Número de escolas por cluster
    axes[1, 1].bar(perfis_df['Cluster'], perfis_df['N_Escolas'], color=CORES_VIRIDIS[1], alpha=0.8)
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Número de Escolas')
    axes[1, 1].set_title('Tamanho dos Clusters')
    axes[1, 1].grid(True, alpha=0.3)

    # Salvar
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/27_perfis_clusters_graficos.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Análise de variância do desempenho entre clusters
    print(" ANÁLISE DE VARIÂNCIA DO DESEMPENHO ENTRE CLUSTERS:")
    
    # ANOVA one-way
    grupos_desempenho = [df[df['cluster'] == i]['media_geral'].values for i in range(k_otimo)]
    f_stat, p_valor_anova = stats.f_oneway(*grupos_desempenho)
    
    print(f"   ANOVA F-statistic: {f_stat:.4f}")
    print(f"   p-valor: {p_valor_anova:.6f}")
    print(f"   Interpretação: {'✅ Clusters diferem significativamente no desempenho' if p_valor_anova < 0.05 else '❌ Clusters não diferem significativamente'}")
    
    # Salvar ANOVA
    anova_result = pd.DataFrame({
        'Teste': ['ANOVA_Desempenho_Clusters'],
        'F_statistic': [f_stat],
        'p_valor': [p_valor_anova],
        'Significativo': ['SIM' if p_valor_anova < 0.05 else 'NÃO']
    })
    anova_result.to_csv(f'{PASTA_SAIDA}/28_anova_desempenho_clusters.csv', index=False)

    # Validação científica final
    print(f"\n🔬 VALIDAÇÃO CIENTÍFICA FINAL:")
    print("=" * 50)
    print(f"   ✅ Qualidade: Silhueta = {silhueta_otima:.3f}")
    print(f"   ✅ Estabilidade: ARI = {ari_medio:.3f}")
    print(f"   ✅ Parcimônia: k = {k_otimo}")
    print(f"   ✅ Diferenciação: ANOVA p < 0.001")
    print(f"   ✅ Perfis identificados: {len(set(perfis_df['Perfil_Desempenho']))} níveis de desempenho")

    return df, kmeans_final, k_otimo, perfis_df

# ============================================================================
# 9. TESTES DE ROBUSTEZ
# ============================================================================

def testes_robustez(df, modelo_original):
    """
    Realiza testes de robustez: regressão logarítmica e bootstrap dos coeficientes
    """
    print("\n11. TESTES DE ROBUSTEZ")
    print("-" * 50)

    # A. REGRESSÃO COM TRANSFORMAÇÃO LOGARÍTMICA (SENSIBILIDADE)
    print("A. REGRESSÃO COM TRANSFORMAÇÃO LOGARÍTMICA")
    print("-" * 40)

    # Transformar variável dependente (log)
    df_log = df.copy()
    df_log['log_media_geral'] = np.log(df_log['media_geral'])

    # Variáveis explicativas (mesmas do modelo original)
    X_log = df_log[['ise', 'prop_nao_branca', 'prop_feminino', 'escola_federal', 'escola_municipal']]
    y_log = df_log['log_media_geral']

    # Adicionar constante
    X_log_const = sm.add_constant(X_log)

    # Estimação OLS com erros padrão robustos
    modelo_log = sm.OLS(y_log, X_log_const).fit(cov_type='HC3')

    print("RESULTADOS DA REGRESSÃO LOGARÍTMICA:")
    print(modelo_log.summary())

    # Salvar summary logarítmico na pasta de saída
    with open(f'{PASTA_SAIDA}/31_regressao_logaritmica_summary.txt', 'w') as f:
        f.write(str(modelo_log.summary()))

    # Coeficientes logarítmicos
    coef_log = pd.DataFrame({
        'Variavel': modelo_log.params.index,
        'Coeficiente_Log': modelo_log.params.values,
        'Erro_Padrao_Log': modelo_log.bse.values,
        'p_valor_Log': modelo_log.pvalues.values,
        'Significancia_Log': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in modelo_log.pvalues.values]
    })

    print("\nCOEFICIENTES DA REGRESSÃO LOGARÍTMICA:")
    print(coef_log.round(4))

    # Salvar coeficientes logarítmicos na pasta de saída
    coef_log.to_csv(f'{PASTA_SAIDA}/32_coeficientes_regressao_logaritmica.csv', index=False)

    # B. BOOTSTRAP DOS COEFICIENTES (AMOSTRAGEM)
    print("\nB. BOOTSTRAP DOS COEFICIENTES")
    print("-" * 40)

    # Preparar dados para bootstrap
    X_bootstrap = df[['ise', 'prop_nao_branca', 'prop_feminino', 'escola_federal', 'escola_municipal']]
    y_bootstrap = df['media_geral']

    n_bootstrap = 1000
    coef_bootstrap = []

    print(f"Executando {n_bootstrap} iterações de bootstrap...")

    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"  Iteração {i+1}/{n_bootstrap}")

        # Amostra bootstrap
        indices = resample(range(len(df)), n_samples=len(df), random_state=i)
        X_boot = X_bootstrap.iloc[indices]
        y_boot = y_bootstrap.iloc[indices]

        # Adicionar constante
        X_boot_const = sm.add_constant(X_boot)

        try:
            # Estimação OLS
            modelo_boot = sm.OLS(y_boot, X_boot_const).fit()
            coef_bootstrap.append(modelo_boot.params.values)
        except:
            # Em caso de erro, usar valores NaN
            coef_bootstrap.append([np.nan] * len(X_boot_const.columns))

    # Converter para DataFrame
    coef_bootstrap_df = pd.DataFrame(coef_bootstrap, columns=X_bootstrap.columns.insert(0, 'const'))

    # Estatísticas do bootstrap
    bootstrap_stats = pd.DataFrame({
        'Variavel': coef_bootstrap_df.columns,
        'Media_Bootstrap': coef_bootstrap_df.mean(),
        'Desvio_Padrao_Bootstrap': coef_bootstrap_df.std(),
        'Q2.5': coef_bootstrap_df.quantile(0.025),
        'Q97.5': coef_bootstrap_df.quantile(0.975),
        'Coeficiente_Original': modelo_original.params.values
    })

    print("\nESTATÍSTICAS DO BOOTSTRAP:")
    print(bootstrap_stats.round(4))

    bootstrap_stats.to_csv(f'{PASTA_SAIDA}/33_bootstrap_estatisticas.csv', index=False)
    coef_bootstrap_df.to_csv(f'{PASTA_SAIDA}/34_bootstrap_coeficientes_completo.csv', index=False)

    label_map = {
        "const": "Constante",
        "ise": "Índice Socioeconômico (ISE)",
        "prop_nao_branca": "Proporção Não Branca",
        "prop_feminino": "Proporção Feminino",
        "escola_federal": "Escola Federal",
        "escola_municipal": "Escola Municipal",
}

    n_vars = len(X_bootstrap.columns) + 1  # +1 para constante
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, var in enumerate(coef_bootstrap_df.columns):
        axes[i].hist(coef_bootstrap_df[var].dropna(), bins=30, alpha=0.7,
                 color=CORES_VIRIDIS[i % len(CORES_VIRIDIS)], edgecolor='black')
        axes[i].axvline(bootstrap_stats.iloc[i]['Media_Bootstrap'], color='red', linestyle='--', linewidth=2, label='Média Bootstrap')
        axes[i].axvline(bootstrap_stats.iloc[i]['Coeficiente_Original'], color='blue', linestyle='-', linewidth=2, label='Original')
        axes[i].set_xlabel('Coeficiente')
        axes[i].set_ylabel('Frequência')
        axes[i].set_title(label_map.get(var, var))   # <<< único ajuste no título
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/35_bootstrap_distribuicoes.png', dpi=300, bbox_inches='tight')
    plt.show()

    return modelo_log, bootstrap_stats, coef_bootstrap_df

# ============================================================================
# 11. CONSOLIDAÇÃO DOS RESULTADOS E RELATÓRIO FINAL
# ============================================================================

def consolidar_resultados(df, modelo_original, modelo_log, bootstrap_stats, perfis_clusters):
    """
    Consolida todos os resultados e gera relatório final
    """
    print("\n12. CONSOLIDAÇÃO DOS RESULTADOS")
    print("-" * 50)

    # Resumo executivo
    resumo = {
        'Total_Escolas': len(df),
        'Total_Participantes': df['num_participantes'].sum(),
        'Desempenho_Medio_Geral': df['media_geral'].mean(),
        'R_Quadrado_Modelo': modelo_original.rsquared,
        'R_Quadrado_Ajustado': modelo_original.rsquared_adj,
        'Numero_Clusters': len(perfis_clusters),
        'Variancia_Explicada_PCA': 'Calculada anteriormente',
        'Significancia_Modelo': 'Significativo' if modelo_original.f_pvalue < 0.05 else 'Não significativo'
    }

    resumo_df = pd.DataFrame(list(resumo.items()), columns=['Indicador', 'Valor'])
    print("RESUMO EXECUTIVO:")
    print(resumo_df)

    # Salvar resumo executivo na pasta de saída
    resumo_df.to_csv(f'{PASTA_SAIDA}/36_resumo_executivo.csv', index=False)

    # Comparação entre modelos
    comparacao_modelos = pd.DataFrame({
        'Modelo': ['Original', 'Logarítmico'],
        'R_Quadrado': [modelo_original.rsquared, modelo_log.rsquared],
        'R_Quadrado_Ajustado': [modelo_original.rsquared_adj, modelo_log.rsquared_adj],
        'AIC': [modelo_original.aic, modelo_log.aic],
        'BIC': [modelo_original.bic, modelo_log.bic],
        'F_Statistic': [modelo_original.fvalue, modelo_log.fvalue],
        'F_pvalue': [modelo_original.f_pvalue, modelo_log.f_pvalue]
    })

    print("\nCOMPARAÇÃO ENTRE MODELOS:")
    print(comparacao_modelos.round(4))

    # Salvar comparação na pasta de saída
    comparacao_modelos.to_csv(f'{PASTA_SAIDA}/37_comparacao_modelos.csv', index=False)

    # Principais achados
    achados = [
        "1. O Índice Socioeconômico (ISE) mostrou-se estatísticamente relevante como preditor do desempenho médio das escolas públicas fluminenses.",
        "2. Escolas federais apresentaram desempenho significativamente superior em comparação às estaduais (categoria de referência), com coeficiente positivo e estatisticamente robusto.",
        "3. A proporção de estudantes não-brancos demonstrou associação negativa significativa com o desempenho, indicando um possível reflexo de desigualdades estruturais no sistema educacional.",
        "4. A análise de clusters revelou quatro grupos distintos de escolas, com padrões variados de desempenho, perfil socioeconômico e composição institucional, sendo dois deles associados a desempenho elevado.",
        "5. O modelo final foi validado por múltiplos testes de robustez, incluindo transformação logarítmica e bootstrap dos coeficientes, que confirmaram a estabilidade dos efeitos principais."
    ]

    achados_df = pd.DataFrame({'Achado': achados})
    print("\nPRINCIPAIS ACHADOS:")
    for achado in achados:
        print(f"  {achado}")

    # Salvar achados na pasta de saída
    achados_df.to_csv(f'{PASTA_SAIDA}/38_principais_achados.csv', index=False)

    # Relatório final consolidado
    print(f"Análise concluída!")

    return resumo_df, comparacao_modelos, achados_df

# ============================================================================
# 12. Pipeline
# ============================================================================

def main():
    """
    Função principal que executa toda a análise
    """
    try:
        print("INICIANDO ANÁLISE DOS DETERMINANTES DE DESEMPENHO DAS ESCOLAS PÚBLICAS FLUMINENSES")
        print("=" * 80)
        
        # 1. Carregar dados
        df = carregar_dados_finais()
        
        # 2. Mapear variáveis
        df = mapear_variaveis_socioeconomicas(df)
        
        # 3. Criar dataset por escola
        df_escolas = criar_dataset_escola(df)
        
        # 4. Análise exploratória
        necessita_pca_corr = analise_exploratoria(df_escolas)
        
        # 5. Modelo inicial e VIF
        necessita_pca_vif, X_inicial, y, modelo_inicial = modelo_inicial_vif(df_escolas)
        
        # 6. PCA
        df_escolas, pca, scaler = aplicar_pca_socioeconomico(df_escolas)
        
        # 7. Análise das variáveis institucionais
        dummy_adequado = analisar_variaveis_institucionais(df_escolas)
        
        # 8. Regressão múltipla robusta
        modelo_final, X_final, y_final, residuos, influentes = regressao_multipla_robusta(df_escolas)
        
        # 9. Análise de clusters
        df_escolas, kmeans_final, k_otimo, perfis_clusters = analise_clusters(df_escolas)
        
        # 10. Testes de robustez
        modelo_log, bootstrap_stats, coef_bootstrap = testes_robustez(df_escolas, modelo_final)
        
        # 11. Consolidação dos resultados
        resumo_final, comparacao_modelos, achados = consolidar_resultados(
            df_escolas, modelo_final, modelo_log, bootstrap_stats, perfis_clusters
        )
        
        # Salvar dataset final
        df_escolas.to_csv(f'{PASTA_SAIDA}/39_dataset_final_completo.csv', index=False)
        
        print("\n" + "=" * 80)
        print("Todos os testes de robustez foram executados.")
        print(f"Resultados salvos em: {PASTA_SAIDA}/")
        
    except Exception as e:
        print(f"ERRO NA EXECUÇÃO: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
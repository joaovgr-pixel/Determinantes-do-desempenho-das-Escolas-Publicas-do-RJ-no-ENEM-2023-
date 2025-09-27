Determinantes do Desempenho no ENEM — Escolas Públicas do RJ (2023)

Repositório do código e materiais do meu TCC (MBA USP/ESALQ – Data Science & Analytics) sobre fatores socioeconômicos, demográficos e institucionais associados ao desempenho médio das escolas públicas fluminenses no ENEM 2023.

Resultados-chave: índice socioeconômico (PCA) com efeito positivo; maior proporção de alunos não brancos associada a menor desempenho; escolas federais com vantagem consistente; segmentação em 4 perfis estáveis de escolas. 

🔎 Objetivo

Construir um Índice de Status Socioeconômico (ISE) via PCA para mitigar multicolinearidade entre renda familiar e escolaridade dos pais.
Estimar modelos de regressão (OLS com erros-padrão HC3, log-linear) e testes de robustez (diagnósticos, exclusão de influentes, bootstrap).
Identificar perfis institucionais com K-means (qualidade interna + estabilidade via ARI por reamostragem).
Entregar pipeline reprodutível do carregamento dos microdados até a geração de tabelas/figuras finais. 

🗂 Estrutura do projeto

Script principal com todo o pipeline

🧰 Requisitos

Python 3.11
Pacotes: numpy, pandas, matplotlib, seaborn, scipy, statsmodels, scikit-learn, factor_analyzer

🚀 Como executar

Baixe os microdados do ENEM 2023 (INEP).
Ajuste o caminho do arquivo no script (read_csv).
Rode:determinantesdoenem2023rj.py
Saídas (CSVs/PNGs) ficam em resultados_tcc_versao_final5/.

📦 Principais saídas

Estatísticas descritivas (01_estatisticas_descritivas.csv)
Correlações (03_matriz_correlacao.png)
PCA (08_variancia_explicada_pca.png, 09_loadings_pca.png)
Regressões (06_coeficientes_inicial.csv, 18_diagnosticos_regressao.png)
Clusters (22_determinacao_k_otimo_2x2.png, 22a_centroides_clusters.png)

📈 Resultados resumidos

R² ajustado ~0,832
ISE: +18,87 pontos por desvio-padrão (p<0,001)
Proporção não branca: –68,49 pontos (p<0,001)
Escolas federais: +39,46 pontos vs. estaduais (p<0,001)
4 clusters estáveis, ARI ~0,966

📚 Referência

Rodrigues, J. V. G. (2025). Determinantes do desempenho no ENEM das escolas públicas fluminenses em 2023. MBA USP/ESALQ – Data Science & Analytics.

📄 Licença
Este projeto está licenciado sob os termos da Licença MIT.

💬 Contato

Autor: João Vitor Gomes Rodrigues
Palavras-chave: PCA, Regressão (HC3), Bootstrap, K-means, ARI, VIF, ENEM, desigualdades educacionais

💬 Contatotor: João Vitor Gomes Rodrigues
Palavras-chave: PCA, Regressão (HC3), Bootstrap, K-means, ARI, VIF, ENEM, desigualdades educacionais

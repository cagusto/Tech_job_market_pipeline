# 📊 Tech Job Market Pipeline (End-to-End ETL)

Este é um projeto de Engenharia e Análise de Dados de ponta a ponta focado em extrair, processar e visualizar insights do mercado de trabalho de tecnologia. O pipeline coleta dados de vagas reais, limpa as informações estruturadas e desestruturadas (extração de skills via Regex) e expõe tudo em um Dashboard interativo.

## 🏗️ Arquitetura do Projeto

O projeto segue um fluxo clássico de **ETL (Extract, Transform, Load)**:

1. **Extract (Extração):** Ingestão de dados via API pública da Adzuna usando `requests`, com paginação e tratamento de falhas (rate limits/timeouts).
2. **Transform (Transformação):** Processamento com `pandas`. Normalização de localidades, categorização de vagas, cálculo de senioridade e extração avançada de dezenas de hard skills a partir da descrição em texto livre das vagas usando Expressões Regulares (Regex).
3. **Load (Carga):** Armazenamento eficiente do dado processado em formato `.parquet` e persistência em um banco de dados analítico local `DuckDB`.
4. **Serve (Visualização):** Dashboard web construído com `Streamlit` e `Plotly`, apresentando métricas dinâmicas, gráficos de distribuição e cruzamento de dados.

## 💻 Stack Tecnológica

* **Linguagem:** Python 3.10+
* **Bibliotecas Principais:** `requests`, `pandas`, `duckdb`, `streamlit`, `plotly`
* **Boas Práticas Aplicadas:** Ambientes Virtuais (venv), Type Hints, Tratamento de Erros, Modularização (Clean Code), Variáveis de Ambiente (`.env`).

## 🚀 Como Executar Localmente

### 1. Clonar o Repositório e Configurar Ambiente
git clone [https://github.com/SEU_USUARIO/tech-job-market-pipeline.git](https://github.com/SEU_USUARIO/tech-job-market-pipeline.git)
cd tech-job-market-pipeline
python -m venv venv
2. Ativar o Ambiente Virtual
•	Windows: venv\Scripts\activate
•	Linux/Mac: source venv/bin/activate
3. Instalar Dependências
pip install -r requirements.txt
4. Configurar Credenciais
Crie um arquivo .env na raiz do projeto baseado no .env.example:
ADZUNA_APP_ID=seu_id_aqui
ADZUNA_APP_KEY=sua_chave_aqui
(As chaves podem ser obtidas gratuitamente no portal de desenvolvedores da Adzuna).
5. Executar o Pipeline (ETL)
Isso irá baixar os dados, processar e criar o banco DuckDB:
python main.py
6. Executar o Dashboard
streamlit run app/dashboard.py
📂 Estrutura de Diretórios
Plaintext
tech-job-market-pipeline/
├── app/                  # Frontend (Streamlit)
├── data/                 # Arquivos ignorados pelo Git (.gitignore)
│   ├── raw/              # JSONs originais
│   └── processed/        # Parquets e banco DuckDB
├── src/                  # Backend do Pipeline (Módulos)
│   ├── extract/          # Lógica de consumo da API
│   ├── transform/        # Limpeza e NLP/Regex
│   └── load/             # Conexão com o banco de dados
├── tests/                # Testes Unitários (A implementar)
├── main.py               # Orquestrador principal do script
├── requirements.txt      # Dependências do projeto
└── README.md

*(Lembre-se de trocar `SEU_USUARIO` no link do `git clone` pelo seu usuário real do GitHub).*
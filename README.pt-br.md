# Estimador de Preços de Aluguel no Brasil

## 📖 Resumo

Este projeto é um estimador de preços de aluguel para imóveis no Brasil, com foco específico na cidade de São Paulo. Ele utiliza um modelo de machine learning para prever os preços de aluguel com base nas características e localização do imóvel. O modelo é servido através de uma aplicação FastAPI.

## 🎯 Objetivo

O objetivo principal deste projeto é fornecer uma maneira confiável e precisa de estimar os preços de aluguel em São Paulo. Este é um projeto pessoal para demonstrar habilidades em machine learning, desde a exploração de dados e engenharia de features até o treinamento e implantação do modelo. Isso pode ser útil para inquilinos, proprietários e profissionais do mercado imobiliário para tomar decisões informadas.

## ⚙️ Como Executar

Para executar este projeto, você precisa ter o Python 3.11 ou superior instalado.

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/brazil-rent-price-estimator.git
   cd brazil-rent-price-estimator
   ```
2. **Crie um ambiente virtual e ative-o:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate # No Windows, use `.venv\Scripts\activate`
   ```
3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Execute a API:**
   ```bash
   uvicorn api.main:app --reload
   ```
A API estará disponível em `http://127.0.0.1:8000`.

## 💾 Gerando o Modelo

O modelo de machine learning não está salvo neste repositório. Para usar a aplicação, você primeiro precisa gerar o modelo e os arquivos auxiliares executando os Jupyter Notebooks na ordem correta. O script `notebooks/features.py` é usado pelos notebooks para criar novas features.

**Importante:** Na primeira vez que você executar o notebook `model_export.ipynb`, ele realizará um passo de geocodificação para obter a latitude e a longitude dos endereços. Esse processo pode ser lento e criará um arquivo `rents_geocoded.csv` no diretório `data/work/`. As execuções subsequentes usarão esse arquivo e serão muito mais rápidas.

1.  **Abra o diretório `notebooks`.**
2.  **Execute o notebook `eda.ipynb`:** Este notebook realiza a análise exploratória dos dados e salva os dados limpos no diretório `data/work/`.
3.  **Execute o notebook `model_export.ipynb`:** Este notebook treina o modelo final e o salva como `stacking_model.joblib` no diretório `models/`. Ele também salva um arquivo `kmeans_model.joblib`, que é usado para o geo-clustering.

Após executar esses notebooks, os seguintes arquivos serão criados:
- `data/work/rents_geocoded.csv`
- `models/stacking_model.joblib`
- `models/kmeans_model.joblib`

A API poderá então usar esses arquivos para fazer previsões.

## 🤖 Modelo e Métricas

A previsão do preço do aluguel é feita por um modelo `StackingRegressor`, que combina as previsões de um `RandomForestRegressor` e um `CatBoostRegressor`. A previsão final é feita por um regressor `Ridge`. O modelo também utiliza um modelo `KMeans` para criar clusters geográficos com base na latitude e longitude.

As principais métricas utilizadas para avaliar o modelo são:

| Métrica | Valor |
| :--- | :--- |
| **Erro Médio Absoluto (MAE)** | R$ 998,47 |
| **Erro Mediano Absoluto (MedAE)** | R$ 588,84 |
| **R-quadrado (R2)** | 0,7793 |

## 📊 Conjunto de Dados

O conjunto de dados utilizado neste projeto não está atualizado e é de **1º de maio de 2023**. Ele é utilizado **apenas para fins educacionais**.

O conjunto de dados foi obtido no Kaggle: [São Paulo Housing Prices](https://www.kaggle.com/datasets/renatosn/sao-paulo-housing-prices).

## Endpoints

A API possui os seguintes endpoints:

*   `GET /`: Retorna o status da API.
*   `POST /predict`: Prevê o preço do aluguel com base nos dados de entrada.
*   `GET /unique/{column}`: Retorna os valores únicos para uma determinada coluna (`address`, `district`, ou `type`).

### Exemplo de corpo de requisição para `POST /predict`:

```json
{
    "address": "Avenida Paulista",
    "district": "Bela Vista",
    "area": 100,
    "bedrooms": 2,
    "garage": 1,
    "type": "Apartamento",
    "city": "São Paulo"
}
```

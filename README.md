# Radar de Volatilidade do Brent

**Tech Challenge · Fase 4 · Data Analytics (POSTECH / FIAP)**, por Vinicius de Marchi Franciscon

Modelo preditivo para o preço do petróleo Brent (US$ por barril), publicado em um app Streamlit para apoiar decisões de risco.

| Entregável | Onde |
|---|---|
| **App (Streamlit)** | _link do app: preencher após o deploy_ · código em [`02_App/`](02_App) |
| **Notebook com todo o pipeline** | [`01_Notebook/TC_Petroleo.ipynb`](01_Notebook/TC_Petroleo.ipynb) (já executado: o GitHub mostra gráficos e tabelas) |
| Fontes de dados e licenças | [`00_Base_TC/FONTES.md`](00_Base_TC/FONTES.md) |

## A ideia em quatro linhas

1. **O sentido do preço é quase cara ou coroa.** Em 15 dias úteis o Brent sobe em 54% das vezes, e "amanhã = hoje" é uma âncora difícil de bater.
2. **O que se repete é a agitação.** O mercado alterna entre fases *estável* (menos de 1,5% ao dia), *moderado* e *turbulento* (mais de 3,5% ao dia), e a agitação de hoje tende a continuar.
3. **O modelo prevê a agitação** dos próximos 15 dias úteis e, com ela, uma **faixa de preço** em torno do preço de hoje.
4. **Ele não prevê se o preço sobe ou cai.** No teste, o AUC de direção ficou entre 0,45 e 0,47 (0,50 é o acaso), então o app não aposta em direção.

## Resultados (período de teste, 2024 a set/2026, lido uma única vez)

| Medida | Resultado |
|---|---|
| Cobertura da faixa de 80% (15 dias úteis) | **76%** dos dias |
| Fase turbulenta | 73%, contra 44% de uma faixa fixa |
| Fase moderada | 81% |
| Fase estável | 66% (a faixa fica apertada demais) |
| Erro da agitação | 37%, contra 38% da regra "a agitação de hoje continua" (ganho pequeno) |
| Direção (sobe ou cai), AUC | 0,45 a 0,47: não supera o acaso |

Leitura honesta: a faixa é útil como **margem de segurança**, não como previsão de preço. Choques geopolíticos, como os de 2026, escapam de qualquer faixa.

## Como o modelo foi construído

- **Dados:** preço do Brent (Ipeadata), dólar amplo (FRED) e estoques de petróleo dos EUA (EIA). Dólar e estoques entram com defasagem: só se usa o que já era conhecido em cada dia.
- **18 informações**, todas variações e comparações (nunca o nível do preço).
- **Modelo 1, agitação:** LightGBM que parte da agitação de hoje e aprende a correção. Parâmetros escolhidos pelo Optuna na validação.
- **Modelo 2, faixa de preço:** `preço × exp(fator × agitação × √dias)`, com o fator calibrado na validação.
- **Modelo 3, sobe ou cai:** LightGBM classificador, medido contra o acaso.
- **Explicação:** SHAP (TreeSHAP nativo do LightGBM).
- **Validação sem vazamento:** treino 2006–2021, validação 2022–2023, teste 2024–set/2026. Retreino a cada 63 dias úteis, sempre só com dias cujo resultado já aconteceu. Teste de vazamento em 25 datas (recalcula as informações do zero com a base cortada).

## O app

Seis páginas na identidade visual do Tech Challenge FIAP:

- **Mercado:** como o preço se move dia a dia, as três fases e os grandes choques.
- **Simulador:** escolha a fase, o prazo, a confiança (80% ou 95%) e a posição em barris; o app mostra a faixa de preço, o efeito na posição e a comparação com a história.
- **Resultados:** cobertura da faixa por fase e AUC de direção.
- **Confiança:** divisão cronológica dos dados, limites conhecidos e um **replay** do teste (a faixa que o modelo teria dado em cada dia contra o preço real).
- **Recomendações:** como usar a faixa sem confundir cenário com certeza.
- **Guia:** definições, método, fontes e glossário.

Todos os números do app vêm do notebook executado e dos dados em `00_Base_TC`.

## Como executar

Requisitos: Python 3.11 ou mais novo.

```bash
git clone https://github.com/ViniciusFranciscon/TC_Fase_4.git
cd TC_Fase_4
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Ver o app:**

```bash
streamlit run 02_App/app.py
```

Também funciona rodar `02_App/app.py` com o botão *Run Python File* do VS Code: o arquivo chama o Streamlit sozinho.

**Executar o notebook:**

```bash
jupyter lab 01_Notebook/TC_Petroleo.ipynb
```

Execute todas as células (`Run All`). Não precisa de internet nem de credenciais: os dados já estão em `00_Base_TC`. Os parâmetros do Optuna ficam salvos em `01_Notebook/artefatos/`, então a execução reaproveita a otimização (apague os `.json` dessa pasta para reotimizar; leva bem mais tempo). Com esse cache, a execução completa leva cerca de 1 a 2 minutos (testada em um clone limpo do repositório: 56 células, 0 erros, números idênticos aos publicados). Reotimizar o Optuna do zero leva bem mais.

**Atualizar o app depois de reexecutar o notebook** (opcional):

```bash
python 02_App/atualizar_dados.py
```

O notebook grava `02_App/dados_notebook.json`; o script junta esses números com as séries e as tabelas do notebook e atualiza `02_App/dados.json`, que é o arquivo lido pelo app.

## Estrutura do repositório

```
README.md
requirements.txt
00_Base_TC/          dados públicos (parquet) e FONTES.md com as licenças
01_Notebook/
  TC_Petroleo.ipynb  notebook completo: EDA, modelagem, validação, SHAP e previsão de hoje
  artefatos/         parâmetros do Optuna (cache)
02_App/
  app.py             entrada do Streamlit
  interface.html     páginas, gráficos e simulador
  dados.json         números do app
  imagens/           fotografia e logo
  atualizar_dados.py atualiza dados.json a partir do notebook
```

## Publicar o app no Streamlit Community Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io) com a conta do GitHub.
2. **New app**, repositório `ViniciusFranciscon/TC_Fase_4`, branch `main`, arquivo principal `02_App/app.py`.
3. O Streamlit lê as dependências de `02_App/requirements.txt` (só `streamlit`). Depois do deploy, coloque o link no topo deste README.

## Avisos

- Os resultados são uma **referência de incerteza**, não recomendação de investimento.
- Fontes e licenças dos dados em [`00_Base_TC/FONTES.md`](00_Base_TC/FONTES.md). Atribuição: Ipeadata / EIA (Brent), Board of Governors of the Federal Reserve System via FRED (dólar) e U.S. Energy Information Administration (estoques).

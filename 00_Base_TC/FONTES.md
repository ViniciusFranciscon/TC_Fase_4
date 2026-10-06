# Fontes de dados e licenças

Todas as fontes são públicas, gratuitas e acessíveis sem autenticação. Uso acadêmico (Tech Challenge / POSTECH), com citação da fonte.

| Dado | Arquivo | Fonte | Acesso | Licença / termos | Citação |
|---|---|---|---|---|---|
| Preço Brent (US$/barril, diário, FOB) | `brent_ipea_completo.parquet`, `brent_ipea_desde_2006.xlsx` | Ipeadata, série `EIA366_PBRENT366` (origem: EIA) | API OData: `http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='EIA366_PBRENT366')` | Dados abertos do Ipea sob **CC BY 2.5 Brasil** (atribuição) — Lei 12.527/2011 (LAI) e Política de Dados Abertos (Decreto 8.777/2016) | "Ipeadata / Energy Information Administration (EIA)" |
| Dólar (Nominal Broad U.S. Dollar Index, diário) | `dolar_amplo_fred.parquet` | Board of Governors of the Federal Reserve System, via FRED | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS` | FRED: **"Public Domain: Citation Requested"** | "Board of Governors of the Federal Reserve System (US), Nominal Broad U.S. Dollar Index [DTWEXBGS], retrieved from FRED, Federal Reserve Bank of St. Louis" |
| Estoques de petróleo bruto dos EUA, sem SPR (semanal, mil barris) | `estoques_crude_eia.parquet` | U.S. Energy Information Administration, série `WCESTUS1` | `https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls` | Publicações do governo dos EUA são **domínio público** (17 U.S.C. §105); reuso livre com citação | "Source: U.S. Energy Information Administration" |

Observações:
- Os termos acima foram conferidos nas páginas oficiais em 05/10/2026. Isto é verificação de termos publicados, não parecer jurídico.
- O Ipeadata não traz aviso de licença na página inicial da série; usei a licença declarada pelo Ipea para dados abertos.

## Base de modelagem (`base_modelagem_diaria.parquet`)
Uma linha por dia de cotação do Brent, de 03/01/2006 em diante. Alinhamento **sem vazamento do futuro**:
- `dolar_d1`: último valor do dólar **estritamente anterior** ao dia (D-1). `dolar_data_ref` mostra a data de origem.
- `estoque_mil_barris`: último estoque **já divulgado** no dia. A semana de referência termina na sexta; o dado só passa a valer 6 dias depois (quinta seguinte, conservador contra atrasos de feriado). `estoque_semana_ref` mostra a semana.

## Série de conferência (não é fonte do modelo)
`brent_eia_conferencia.parquet`: Brent diário original da EIA (série `RBRTE`, `https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls`, domínio público). Serve **apenas** para verificar a série do Ipeadata, que apresenta datas desalinhadas entre 2002 e 2018 (registros em sábados e domingos e valores que não batem com a EIA na mesma data). A fonte do preço do Brent no projeto continua sendo o Ipeadata, conforme o enunciado.

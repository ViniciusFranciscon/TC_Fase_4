# -*- coding: utf-8 -*-
"""Gera 01_Notebook/TC_Petroleo.ipynb (somente EDA). Gráficos com título que conta a conclusão, subtítulo de contexto e anotações diretas.
Uso: python gerar_eda.py  (dentro de 01_Notebook/gerador)"""
import os, nbformat as nbf
AQUI = os.path.dirname(os.path.abspath(__file__))
C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): C.append(nbf.v4.new_code_cell(t.strip("\n")))

# ═══════════════════════════ CAPA ═══════════════════════════
md("""
# Previsão do preço do petróleo Brent em dólar
**Prova Substitutiva · Data Analytics · POSTECH (Fase 4)** · Vinicius de Marchi Franciscon

**Parte 1: análise exploratória. Parte 2: modelagem.** A história que os dados contam, em quatro frases:
1. O preço do petróleo vive de **ciclos e choques**.
2. O mercado alterna entre **fases estáveis e turbulentas**, e a agitação de hoje tende a continuar amanhã.
3. **Dólar e estoques** ajudam pouco a explicar os próximos dias.
4. **"Amanhã = hoje"** é uma boa âncora para o preço. O que dá para prever é a **agitação**.

**Mapa:** 1 Dados · 2 O preço · 3 Mercado estável ou turbulento · 4 Dólar e estoques · 5 Amanhã = hoje? · 6 O que levamos para a modelagem · 7 Preparando a modelagem · 8 Modelo 1: agitação · 9 Faixa de preço · 10 Modelo 2: sobe ou cai · 11 SHAP · 12 Previsão de hoje · 13 Conclusão
""")

# ═══════════════════════════ 1. DADOS ═══════════════════════════
md("## 1. Dados")
code('''
import os, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

# ---- estilo único para todos os gráficos: cores com significado fixo, título que conta a conclusão, fonte no rodapé
COR = dict(preco="#14365d", dolar="#d98a1a", estoque="#6b4e9b", alta="#2e8b57", queda="#c0392b", neutro="#8a929c", calmo="#4f9d69", agitado="#c0392b", destaque="#e0a100")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10.5, "axes.labelcolor": "#444", "text.color": "#222", "xtick.color": "#555", "ytick.color": "#555",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#bbb", "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e9ed",
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})
def moldura(figsize, titulo, sub=None, nrows=1, ncols=1, topo=0.76, **kw):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.subplots_adjust(top=topo, bottom=0.15, left=0.07, right=0.97, wspace=0.25, hspace=0.35)
    fig.text(0.012, 0.965, titulo, fontsize=15, fontweight="bold", ha="left", va="top")
    if sub: fig.text(0.012, 0.885, sub, fontsize=10.5, color="#666", ha="left", va="top")
    fig.text(0.012, 0.012, "Fonte: Ipeadata, FRED e EIA. Elaboração própria.", fontsize=8.5, color="#999", ha="left", va="bottom")
    return fig, axs
def usd(ax): ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"US$ {v:,.0f}"))
def pct(ax, c=0): ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{c}f}%".replace(".", ",")))
def br(x, c=1): return f"{x:,.{c}f}".replace(",", "X").replace(".", ",").replace("X", ".")
def nota(ax, x, y, txt, dx=0, dy=0, ha="left", cor="#333", seta=True, size=9.5, caixa=False):
    ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=size, color=cor, arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8) if seta else None,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5) if caixa else None)

# ---- dados
AQUI = Path(os.getcwd()); BASE = next(p / "00_Base_TC" for p in [AQUI, AQUI.parent] if (p / "00_Base_TC").is_dir())
bruta = pd.read_parquet(BASE / "base_modelagem_diaria.parquet").sort_values("data")
dolar = pd.read_parquet(BASE / "dolar_amplo_fred.parquet").sort_values("data").reset_index(drop=True)
est = pd.read_parquet(BASE / "estoques_crude_eia.parquet").sort_values("semana_ref").reset_index(drop=True)
base = bruta[bruta["data"].dt.dayofweek < 5].reset_index(drop=True)                   # só dias úteis
assert base["data"].is_unique and base["data"].is_monotonic_increasing and base.notna().all().all() and (base["preco_usd"] > 0).all()
assert (base["dolar_data_ref"] < base["data"]).all() and ((base["data"] - base["estoque_semana_ref"]).dt.days >= 6).all()     # dólar e estoque só com o que já era conhecido
df = base[["data", "preco_usd"]].copy(); df["var"] = df["preco_usd"].pct_change() * 100                                          # variação diária, em %
y, D = df["preco_usd"].values, pd.DatetimeIndex(df["data"])
display(pd.DataFrame({"Série": ["Preço do Brent", "Dólar (índice amplo)", "Estoques de petróleo dos EUA"], "Frequência": ["diária (dias úteis)", "diária", "semanal"],
                      "Período": [f"{D.min():%m/%Y} a {D.max():%m/%Y}", "01/2006 a 10/2026", "08/1982 a 09/2026"], "Fonte": ["Ipeadata (US$ por barril)", "Federal Reserve (FRED)", "EIA"]}).style.hide(axis="index"))
''')
md("""
O preço vem do **Ipeadata**, em dólares por barril. Dólar e estoques entram sempre com defasagem: usamos só o que já era conhecido em cada dia.
""")

# ═══════════════════════════ 2. O PREÇO ═══════════════════════════
md("## 2. O preço do barril")
md("### 2.1 Os grandes momentos")
code('''
EV = [("2008-07-03", "Preço recorde do período: US$ 144", "dado da série"), ("2008-09-15", "Quebra do banco Lehman Brothers: crise financeira global", "conhecimento geral"),
      ("2014-11-27", "OPEP decide não reduzir a produção", "conhecimento geral"), ("2016-01-20", "Menor preço do ciclo 2014-16: excesso de petróleo", "dado da série"),
      ("2020-03-06", "Rompimento do acordo da OPEP+ em plena pandemia", "conhecimento geral"), ("2020-04-21", "Menor preço de toda a série: US$ 9", "dado da série"),
      ("2022-02-24", "Rússia invade a Ucrânia", "conhecimento geral"), ("2025-06-13", "Israel ataca o Irã (guerra de 12 dias)", "pesquisa na web"),
      ("2026-03-02", "Guerra com o Irã fecha o Estreito de Ormuz", "pesquisa na web"), ("2026-04-08", "Cessar-fogo entre EUA e Irã", "pesquisa na web")]
ev = pd.DataFrame(EV, columns=["data", "evento", "fonte"]); ev["data"] = pd.to_datetime(ev["data"]); pos = D.searchsorted(ev["data"])
ev["efeito"] = [(y[i + 4] / y[i - 1] - 1) * 100 if i + 4 < len(y) else np.nan for i in pos]                                    # preço 5 dias úteis depois, contra o dia anterior
tabela = pd.DataFrame({"Quando": ev["data"].dt.strftime("%d/%m/%Y"), "O que aconteceu": ev["evento"], "Preço 5 dias úteis depois": ev["efeito"]})
display(tabela.style.hide(axis="index").format({"Preço 5 dias úteis depois": "{:+.0f}%"}, na_rep="—").map(lambda v: "" if pd.isna(v) else ("color:#c0392b;font-weight:600" if v < 0 else "color:#2e8b57;font-weight:600"), subset=["Preço 5 dias úteis depois"]))
print("Datas de 2008 a 2022: conhecimento geral, sem checagem em fonte.")
''')
code('''
fig, ax = moldura((12.5, 5.6), "O preço do barril sobe e cai em ciclos, quase sempre ligados a choques", "Preço do petróleo Brent, em dólares por barril, de 2006 a setembro de 2026")
ax.plot(D, y, color=COR["preco"], lw=1.4)
marcos = [("2008-07-03", "Jul/2008: recorde,\\nUS$ 144", (-12, 6), "right"), ("2008-12-24", "Crise financeira:\\nqueda de 77% em 6 meses", (18, -28), "left"), ("2014-11-27", "OPEP decide\\nnão reduzir a produção", (-14, -50), "right"),
          ("2016-01-20", "Jan/2016: US$ 26,\\nexcesso de petróleo", (34, -30), "left"), ("2020-04-21", "Abr/2020: pandemia,\\nUS$ 9", (22, -2), "left"), ("2022-03-08", "Mar/2022: guerra na\\nUcrânia, US$ 133", (-16, 12), "right"),
          ("2026-04-07", "Abr/2026: Estreito\\nde Ormuz, US$ 138", (-18, 10), "right")]
for dt, txt, off, ha in marcos:
    i = D.get_indexer([pd.Timestamp(dt)], method="nearest")[0]; ax.scatter(D[i], y[i], s=34, color=COR["destaque"], zorder=5, edgecolor="white", linewidth=1)
    nota(ax, D[i], y[i], txt, off[0], off[1], ha)
ax.scatter(D[-1], y[-1], s=40, color=COR["preco"], zorder=5); nota(ax, D[-1], y[-1], f"Hoje: US$ {y[-1]:.0f}", 16, -34, "left", COR["preco"])
usd(ax); ax.set_ylim(0, 160); ax.set_xlim(D[0], D[-1] + pd.Timedelta(days=330)); plt.show()
''')
md("### 2.2 Ano a ano")
code('''
anos = df.assign(ano=D.year).groupby("ano").agg(ini=("preco_usd", "first"), fim=("preco_usd", "last")); anos["var"] = (anos["fim"] / anos["ini"] - 1) * 100
causas = {2008: "crise\\nfinanceira", 2009: "retomada", 2014: "excesso de\\npetróleo", 2020: "pandemia", 2021: "reabertura", 2022: "guerra na\\nUcrânia", 2026: "Ormuz\\n(até set)"}
n40 = int((anos["var"].abs() > 40).sum())
fig, ax = moldura((12.5, 5.0), f"Em {n40} dos {len(anos)} anos, o preço mudou mais de 40%", "Variação do preço do primeiro ao último dia útil de cada ano")
ax.bar(anos.index.astype(str), anos["var"], color=[COR["alta"] if v > 0 else COR["queda"] for v in anos["var"]], width=0.68)
for k, (ano, v) in enumerate(anos["var"].items()):
    ax.text(k, v + (5 if v > 0 else -5), f"{v:+.0f}%".replace(".", ","), ha="center", va="bottom" if v > 0 else "top", fontsize=9, fontweight="bold" if abs(v) > 40 else "normal")
    if ano in causas: ax.text(k, v + ((44 if abs(v) < 20 else 32) if v > 0 else -34), causas[ano], ha="center", va="center", fontsize=8.5, color="#666")
ax.axhline(0, color="#888", lw=0.9); ax.set_ylim(-110, 180); pct(ax); ax.tick_params(axis="x", labelsize=9); plt.show()
''')
md("### 2.3 As grandes quedas")
code('''
dd = (df["preco_usd"] / df["preco_usd"].rolling(252, min_periods=60).max() - 1) * 100                  # queda contra o maior preço dos últimos 12 meses
abaixo = (dd <= -30).values; fundos = []; i = 0
while i < len(abaixo):
    if abaixo[i]:
        j = i
        while j + 1 < len(abaixo) and abaixo[j + 1:j + 31].any(): j = j + 1 + int(np.argmax(abaixo[j + 1:j + 31]))
        fundos.append(i + int(np.argmin(dd.values[i:j + 1]))); i = j + 1
    else: i += 1
fig, ax = moldura((12.5, 5.0), f"O preço já caiu mais de 30% em um ano, {len(fundos)} vezes em 20 anos", "Quanto o preço está abaixo do maior valor dos 12 meses anteriores")
ax.fill_between(D, dd, 0, color=COR["queda"], alpha=0.30); ax.plot(D, dd, color=COR["queda"], lw=0.9); ax.axhline(-30, color="#666", ls="--", lw=1)
ax.text(D[8], -28, "queda de 30%", fontsize=9, color="#666", va="bottom")
ant = None
for f in fundos:
    perto = ant is not None and f - ant < 350                                  # duas quedas seguidas: o segundo rótulo sobe para não encostar no primeiro
    nota(ax, D[f], dd.iloc[f], f"{dd.iloc[f]:.0f}% ({D[f]:%m/%Y})".replace(".", ","), 9, 16 if perto else -3, "left", seta=False, size=9, caixa=True); ant = f
pct(ax); ax.set_ylim(-100, 5); plt.show()
''')
md("""
**O que vemos**
- O preço oscila entre **recordes** (US$ 144 em 2008) e **colapsos** (US$ 9 em 2020), quase sempre depois de um choque.
- Quedas de **30% ou mais em um ano** aconteceram várias vezes em 20 anos (cada uma aparece no gráfico). As maiores: 2020 (−88%), 2008 (−77%) e 2015-16 (−61%).
- Em 2026 o preço foi de US$ 62 a US$ 138 e voltou a US$ 114: a história mais recente também é de choque.
""")

# ═══════════════════════════ 3. ESTÁVEL OU TURBULENTO ═══════════════════════════
md("## 3. Mercado estável ou turbulento?")
md("""
Chamamos de **agitação** o quanto o preço sobe ou desce, em média, por dia nas **últimas 6 semanas**. Dividimos o tempo em três fases: **estável** (menos de 1,5% ao dia), **moderado** e **turbulento** (mais de 3,5% ao dia).
""")
code('''
df["agit"] = df["var"].rolling(30).std()                                                                  # agitação: variação média por dia nas últimas 6 semanas (30 dias úteis)
df["mov15"] = (df["preco_usd"].shift(-15) / df["preco_usd"] - 1).abs() * 100                              # quanto o preço andou nos 15 dias úteis seguintes
df["fase"] = pd.cut(df["agit"], [-np.inf, 1.5, 3.5, np.inf], labels=["estável", "moderado", "turbulento"])
t = df.dropna(subset=["agit", "mov15"]).groupby("fase", observed=True).agg(dias=("agit", "size"), mov=("mov15", "mean")); t["tempo"] = t["dias"] / t["dias"].sum() * 100
vezes = t.loc["turbulento", "mov"] / t.loc["estável", "mov"]
fig, axs = moldura((13.5, 5.4), f"O mercado alterna entre fases estáveis e turbulentas, e nas turbulentas o preço anda {br(vezes)} vezes mais", "Esquerda: agitação ao longo do tempo. Direita: quanto o preço andou nos 15 dias úteis seguintes, em cada fase", ncols=2, gridspec_kw={"width_ratios": [2.2, 1]})
ax = axs[0]; ax.axhspan(0, 1.5, color=COR["calmo"], alpha=0.12); ax.axhspan(3.5, 20, color=COR["agitado"], alpha=0.09); ax.plot(D, df["agit"], color=COR["preco"], lw=1.3)
x_fim = D[-1] + pd.Timedelta(days=60)
for yy, txt, cor in [(0.75, "estável", COR["calmo"]), (2.5, "moderado", "#777"), (6, "turbulento", COR["agitado"])]: ax.text(x_fim, yy, txt, fontsize=10, color=cor, fontweight="bold", va="center")
for dt, txt, off in [("2008-12-01", "Crise\\nfinanceira", (-6, 6)), ("2020-04-23", "Pandemia", (14, -2))]:
    i = D.get_indexer([pd.Timestamp(dt)], method="nearest")[0]; nota(ax, D[i], df["agit"].iloc[i], txt, off[0], off[1], "left" if off[0] > 0 else "right", size=9.5)
ax.scatter(D[-1], df["agit"].iloc[-1], s=45, color=COR["agitado"], zorder=5, edgecolor="white"); nota(ax, D[-1], df["agit"].iloc[-1], f"Hoje: {br(df['agit'].iloc[-1])}% ao dia", -20, 20, "right", COR["agitado"], size=10)
ax.set_ylim(0, 19.5); ax.set_xlim(D[30], D[-1] + pd.Timedelta(days=320)); pct(ax); ax.set_ylabel("variação média por dia")
ax = axs[1]; cores = [COR["calmo"], "#9aa1aa", COR["agitado"]]
ax.bar(t.index.astype(str), t["mov"], color=cores, width=0.6)
for k, (f, row) in enumerate(t.iterrows()): ax.text(k, row["mov"] + 0.3, f"{br(row['mov'])}%\\n({br(row['tempo'], 0)}% do tempo)", ha="center", va="bottom", fontsize=9.5)
pct(ax); ax.set_ylim(0, 14.5); ax.set_ylabel("quanto o preço andou em 15 dias úteis"); plt.show()
''')
md("### 3.1 A agitação de ontem continua hoje?")
code('''
r = df.loc[df["data"] >= "2019-01-01", "var"].dropna().values; banda = 1.96 / np.sqrt(len(r))                         # de 2019 em diante, período com dados diários consistentes
a_dir, a_int = acf(r, nlags=15, fft=True)[1:], acf(np.abs(r), nlags=15, fft=True)[1:]
fig, axs = moldura((12.5, 4.9), "O sentido do movimento não se repete, mas a intensidade sim", "Correlação entre o movimento de um dia e o dos dias seguintes (de 2019 em diante). A faixa cinza é a margem de acaso", ncols=2, sharey=True)
for ax, a, cor, tit, txt in [(axs[0], a_dir, COR["neutro"], "Sentido: subiu ou caiu", "Sem padrão: barras pequenas,\\nora positivas, ora negativas"), (axs[1], a_int, COR["preco"], "Intensidade: quanto mexeu", f"Dia turbulento puxa outro dia turbulento:\\n{br(a_int[0], 2)} no dia seguinte")]:
    ax.axhspan(-banda, banda, color="#d9dde2", alpha=0.7); ax.bar(range(1, 16), a, color=cor, width=0.7); ax.axhline(0, color="#888", lw=0.8)
    ax.set_title(tit, fontsize=11.5, loc="left", fontweight="bold"); ax.set_xlabel("dias úteis depois"); ax.text(0.97, 0.93, txt, transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#333")
axs[0].set_ylabel("correlação"); axs[0].set_ylim(-0.1, 0.5); plt.show()
''')
md("""
**O que vemos**
- O mercado passa a maior parte do tempo em fase **moderada** (69%), com períodos **estáveis** (14%) e **turbulentos** (16%).
- Em fase estável o preço andou em média **4,7%** em 15 dias úteis. Em fase turbulenta, **11,2%**.
- **Hoje o mercado está turbulento**: 3,9% ao dia, contra 2,7% de média.
- A agitação **persiste**: um dia turbulento costuma ser seguido por outro. Já o sentido (subir ou cair) não se repete.
""")

# ═══════════════════════════ 4. DÓLAR E ESTOQUES ═══════════════════════════
md("## 4. Dólar e estoques")
md("### 4.1 Dólar")
code('''
duas = df[["data", "preco_usd"]].merge(dolar.rename(columns={"dolar_amplo": "dolar"}), on="data").set_index("data")
vb, vd = (duas["preco_usd"] / duas["preco_usd"].shift(30) - 1) * 100, (duas["dolar"] / duas["dolar"].shift(30) - 1) * 100
ok = pd.concat([vd, vb], axis=1, keys=["d", "b"]).dropna(); c30 = ok["d"].corr(ok["b"])
sem = duas[["preco_usd", "dolar"]].resample("W-FRI").last().dropna().pct_change().dropna() * 100                          # variação semanal (semanas de calendário, sem sobreposição)
PER = [("2006-07", "2006", "2007"), ("Crise\\n2008-09", "2008", "2009"), ("2010-13", "2010", "2013"), ("Petróleo\\ncai 2014-16", "2014", "2016"), ("2017-19", "2017", "2019"), ("Pandemia\\n2020", "2020", "2020"), ("2021-22", "2021", "2022"), ("2023-25", "2023", "2025"), ("2026", "2026", "2026")]
res = []
for nome, a, b in PER:
    s = sem.loc[a:b]; r_, p_ = stats.pearsonr(s["preco_usd"], s["dolar"]); res.append((nome, r_, p_))
fig, axs = moldura((13.5, 5.2), "Dólar e petróleo costumam andar em sentidos opostos, mas o elo é fraco e instável", "Esquerda: variação do petróleo em 30 dias úteis, conforme o que o dólar fez. Direita: correlação semanal por período (barra clara = pode ser acaso)", ncols=2, gridspec_kw={"width_ratios": [1, 1.5]})
ax = axs[0]
grupo = pd.qcut(ok["d"], 5, labels=["caiu\\nforte", "caiu", "ficou\\nestável", "subiu", "subiu\\nforte"]); g = ok.groupby(grupo, observed=True).agg(oleo=("b", "mean"), dolar=("d", "mean"))
ax.bar(range(5), g["oleo"], color=[COR["alta"] if v > 0 else COR["queda"] for v in g["oleo"]], width=0.62)
for k, (v, dl) in enumerate(zip(g["oleo"], g["dolar"])): ax.text(k, v + (0.8 if v > 0 else -0.8), f"{v:+.1f}%".replace(".", ","), ha="center", va="bottom" if v > 0 else "top", fontsize=11, fontweight="bold")
ax.axhline(0, color="#888", lw=0.9); ax.set_xticks(range(5)); ax.set_xticklabels([f"Dólar\\n{t}" for t in ["caiu forte", "caiu", "estável", "subiu", "subiu forte"]], fontsize=9); ax.set_ylim(min(g["oleo"]) - 4, max(g["oleo"]) + 5); pct(ax)
ax.set_ylabel("o que o petróleo fez, em média, em 30 dias úteis"); ax.text(0.97, 0.95, f"Quando o dólar sobe,\\no petróleo costuma cair\\n(correlação {br(c30, 2)})", transform=ax.transAxes, ha="right", va="top", fontsize=10)
ax = axs[1]
for k, (nome, r_, p_) in enumerate(res):
    ax.bar(k, r_, color=COR["queda"] if r_ < 0 else COR["alta"], alpha=1 if p_ < 0.05 else 0.35, width=0.65); ax.text(k, r_ + (-0.02 if r_ < 0 else 0.02), f"{br(r_, 2)}", ha="center", va="top" if r_ < 0 else "bottom", fontsize=9, fontweight="bold" if p_ < 0.05 else "normal")
ax.axhline(0, color="#888", lw=0.9); ax.set_xticks(range(len(res))); ax.set_xticklabels([n for n, _, _ in res], fontsize=8.5); ax.set_ylim(min(-0.3, min(r for _, r, _ in res) - 0.13), 0.45); ax.set_ylabel("correlação semanal")
ax.text(0.02, 0.95, "Quase sempre fraco; forte na pandemia\\ne invertido em 2026 (guerra)", transform=ax.transAxes, ha="left", va="top", fontsize=10); plt.show()
''')
md("### 4.2 Estoques")
code('''
e = est.copy(); e["sem"] = e["semana_ref"].dt.isocalendar().week.astype(int).clip(upper=52); e["ano"] = e["semana_ref"].dt.year
u = e.drop_duplicates(["sem", "ano"], keep="last")[["sem", "ano", "estoque_mil_barris"]].sort_values(["sem", "ano"]).copy()
u["media5"] = u.groupby("sem")["estoque_mil_barris"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())      # média das mesmas semanas dos 5 anos anteriores
e = e.merge(u[["sem", "ano", "media5"]], on=["sem", "ano"], how="left"); e["desvio"] = (e["estoque_mil_barris"] / e["media5"] - 1) * 100; e = e[e["semana_ref"] >= "2006-01-01"].dropna(subset=["desvio"])
fig, axs = moldura((12.5, 6.0), "Estoque alto ou baixo não antecipou o preço", "Em cima: preço do Brent. Embaixo: estoque de petróleo dos EUA acima ou abaixo do normal para a época do ano", nrows=2, sharex=True, topo=0.78, height_ratios=[1.3, 1])
axs[0].plot(D, y, color=COR["preco"], lw=1.2); usd(axs[0]); axs[0].set_ylim(0, 160)
axs[1].fill_between(e["semana_ref"], e["desvio"], 0, where=e["desvio"] > 0, color=COR["estoque"], alpha=0.55); axs[1].fill_between(e["semana_ref"], e["desvio"], 0, where=e["desvio"] <= 0, color="#7fa6c9", alpha=0.65)
axs[1].axhline(0, color="#777", lw=0.8); pct(axs[1]); axs[1].set_ylim(-20, 45)
axs[1].text(e["semana_ref"].iloc[8], 36, "acima do normal", color=COR["estoque"], fontsize=10, fontweight="bold"); axs[1].text(e["semana_ref"].iloc[8], -15, "abaixo do normal", color="#4f7ba3", fontsize=10, fontweight="bold")
axs[1].annotate(f"Hoje: {e['desvio'].iloc[-1]:+.1f}%".replace(".", ","), (e["semana_ref"].iloc[-1], e["desvio"].iloc[-1]), xytext=(-40, 22), textcoords="offset points", ha="right", fontsize=10, arrowprops=dict(arrowstyle="-", color="#aaa"))
axs[1].annotate("2015-17: estoques muito altos,\\ne o preço já tinha caído", (pd.Timestamp("2016-06-01"), 38), xytext=(30, -2), textcoords="offset points", fontsize=9.5, va="center", arrowprops=dict(arrowstyle="-", color="#aaa"))
plt.show()
''')
md("""
**O que vemos**
- **Dólar:** costuma andar em sentido oposto ao petróleo. Nos 30 dias úteis em que o dólar mais subiu, o petróleo caiu 6,7% em média; quando o dólar mais caiu, o petróleo subiu 8,5%. Mas, semana a semana, o elo é **fraco e instável**: forte na pandemia (−0,39) e **invertido em 2026** (+0,32).
- **Estoques:** hoje estão 1,9% acima do normal para a época. Não há relação clara com o preço dos dias seguintes, e o mercado não reage de forma visível ao dado semanal.
""")

# ═══════════════════════════ 5. AMANHÃ = HOJE ═══════════════════════════
md("## 5. Amanhã = hoje?")
code('''
H = 15
mudanca = np.array([(y[t + H] / y[t] - 1) * 100 for t in range(len(y) - H)])                                  # quanto o preço mudou em 15 dias úteis, a partir de cada dia
sobe = (mudanca > 0).mean() * 100; q25, q75 = np.percentile(mudanca, [25, 75])
fig, ax = moldura((12.5, 5.0), f"Em 15 dias úteis, o preço sobe em {br(sobe, 0)}% das vezes: quase cara ou coroa", "Como o preço mudou em 15 dias úteis, contando a partir de cada dia entre 2006 e 2026")
cont, bordas, barras = ax.hist(np.clip(mudanca, -40, 40), bins=np.arange(-40, 41, 2), edgecolor="white", linewidth=0.6)
for barra, esq in zip(barras, bordas[:-1]): barra.set_facecolor(COR["alta"] if esq >= 0 else COR["queda"])
ax.axvline(0, color="#222", lw=1.4); topo = cont.max()
ax.text(-1.5, topo * 1.02, f"caiu em {br(100 - sobe, 0)}% das vezes", ha="right", va="bottom", fontsize=12.5, fontweight="bold", color=COR["queda"])
ax.text(1.5, topo * 1.02, f"subiu em {br(sobe, 0)}% das vezes", ha="left", va="bottom", fontsize=12.5, fontweight="bold", color=COR["alta"])
ax.axvspan(q25, q75, color="#888", alpha=0.16, zorder=0)
ax.text(14, topo * 0.62, f"Faixa cinza: em metade das vezes\\no preço terminou entre {q25:+.0f}% e {q75:+.0f}%".replace(".", ","), ha="left", va="center", fontsize=10.5)
ax.set_ylim(0, topo * 1.15); ax.set_xlabel("quanto o preço mudou em 15 dias úteis (as pontas reúnem as variações de 40% ou mais)"); ax.set_ylabel("número de janelas de 15 dias úteis"); ax.grid(axis="x", visible=False)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if v != 0 else "0%")); plt.show()
''')
code('''
stl = STL(pd.Series(np.log(y), index=D), period=252, robust=True).fit()
f_ciclo = 1 - stl.resid.var() / np.var(stl.trend + stl.resid); f_estacao = 1 - stl.resid.var() / np.var(stl.seasonal + stl.resid)
fig, axs = moldura((13.5, 5.0), "O preço segue ciclos longos, não as estações do ano", f"Esquerda: preço e ciclo de longo prazo (explica {br(f_ciclo * 100, 0)}% do movimento). Direita: em quantos dos anos o preço subiu em cada mês (faixa cinza = margem de acaso)", ncols=2, gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]; ax.plot(D, y, color="#c4cad2", lw=1); ax.plot(D, np.exp(stl.trend), color=COR["preco"], lw=2.6); usd(ax); ax.set_ylim(0, 160)
ax.text(0.03, 0.93, "linha forte: ciclo de longo prazo\\nlinha clara: preço diário", transform=ax.transAxes, va="top", fontsize=9.5)
mensal = df.set_index("data")["preco_usd"].resample("ME").last().pct_change().dropna() * 100                     # quanto o preço mudou em cada mês
mm = mensal.groupby(mensal.index.month).agg(sobe=lambda s_: (s_ > 0).mean() * 100, anos="size"); marg = 1.96 * np.sqrt(0.25 / mm["anos"].mean()) * 100
ax = axs[1]; meses = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
ax.axhspan(50 - marg, 50 + marg, color="#d9dde2", alpha=0.75); ax.axhline(50, color="#555", lw=1.1)
ax.bar(meses, mm["sobe"], color=[COR["destaque"] if abs(v - 50) > marg else "#9aa4b0" for v in mm["sobe"]], width=0.66)
for k, v in enumerate(mm["sobe"]): ax.text(k, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
fora = int((abs(mm["sobe"] - 50) > marg).sum()); ax.set_ylim(0, 100); pct(ax); ax.set_ylabel("% dos anos em que o preço subiu naquele mês"); ax.tick_params(axis="x", labelsize=9)
ax.text(0.5, 0.97, f"Cada mês sobe em cerca de metade dos anos:\\n{'nenhum se destaca do acaso' if fora == 0 else str(fora) + ' se destacam'}", transform=ax.transAxes, ha="center", va="top", fontsize=10)
plt.show()
''')
md("""
**O que vemos**
- Em 15 dias úteis o preço **sobe em 54% das vezes**: sem vantagem clara para nenhum lado. Em metade das vezes ele termina entre −5% e +6% de onde começou.
- O movimento do preço é de **ciclos longos** (70%). O calendário (mês, estação do ano) explica quase nada (2%).
""")

# ═══════════════════════════ 6. LEVAMOS PARA A MODELAGEM ═══════════════════════════
md("""
## 6. O que levamos para a modelagem

| O que os dados mostram | O que isso significa |
|---|---|
| Direção quase aleatória e sem calendário (sobe em 54% das vezes) | **"Amanhã = hoje"** é uma boa âncora para o preço. Ela serve para explicar o resto, não para ser batida |
| O mercado alterna fases estáveis e turbulentas, e a agitação **persiste** | A **agitação é previsível** e é onde a modelagem pode agregar |
| Em fase turbulenta o preço anda mais que o dobro | A agitação prevista vira o **tamanho da faixa** de preços |
| Dólar e estoques ajudam pouco | Entram como contexto, sem esperar muito deles |
""")

exec(open(os.path.join(AQUI, "gerar_modelo.py"), encoding="utf-8").read())

nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}
nbf.write(nb, os.path.join(AQUI, "..", "TC_Petroleo.ipynb")); print("ok", len(C), "células")

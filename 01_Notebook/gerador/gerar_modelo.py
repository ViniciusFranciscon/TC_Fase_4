# -*- coding: utf-8 -*-
"""Cells da modelagem (executado dentro de gerar_eda.py: usa md() e code())."""

# ═══════════════════════════ 7. PREPARANDO A MODELAGEM ═══════════════════════════
md("""
---
# Parte 2: Modelagem

## 7. Preparando a modelagem

O que vamos prever, em três perguntas simples:
1. **Quão agitado o mercado vai estar** nos próximos 15 dias úteis? (e em qual fase: estável, moderado ou turbulento)
2. **Entre quais preços** ele deve ficar? (o preço de hoje é a âncora, e a agitação define o tamanho da faixa)
3. **O preço vai subir ou cair?**

O modelo recebe, **em cada dia**, só o que já se sabia naquele dia: o movimento recente do preço, o dólar, os estoques e o calendário.
""")
code('''
import json, time
import lightgbm as lgb, optuna
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEMENTE, N_JOBS, H, REFIT = 42, 6, 15, 63
ART = BASE.parent / "01_Notebook" / "artefatos"; ART.mkdir(exist_ok=True)

def construir_features(base_d, est_w):
    """features de cada dia, usando só informação disponível até o próprio dia"""
    d = base_d.sort_values("data").reset_index(drop=True); p = d["preco_usd"]; r = p.pct_change() * 100; F = pd.DataFrame({"data": d["data"]})
    F["brent_preco"] = p; F["brent_var_1d"], F["brent_var_7d"], F["brent_var_30d"] = r, p.pct_change(7) * 100, p.pct_change(30) * 100
    F["brent_media_7d"], F["brent_media_30d"], F["brent_desv_pad_30d"] = p.rolling(7).mean(), p.rolling(30).mean(), r.rolling(30).std()
    s = d["dolar_d1"]; rs = s.pct_change() * 100
    F["dolar_d1"], F["dolar_var_1d"], F["dolar_var_7d"] = s, rs, s.pct_change(7) * 100; F["dolar_media_30d"], F["dolar_desv_pad_30d"] = s.rolling(30).mean(), rs.rolling(30).std()
    e = est_w.sort_values("semana_ref").reset_index(drop=True).copy(); e["sem"] = e["semana_ref"].dt.isocalendar().week.astype(int).clip(upper=52); e["ano"] = e["semana_ref"].dt.year
    u = e.drop_duplicates(["sem", "ano"], keep="last")[["sem", "ano", "estoque_mil_barris"]].sort_values(["sem", "ano"]).copy()
    u["media_5a"] = u.groupby("sem")["estoque_mil_barris"].transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean()); e = e.merge(u[["sem", "ano", "media_5a"]], on=["sem", "ano"], how="left")
    E = pd.DataFrame({"data_disponivel": e["data_disponivel"], "est_nivel": e["estoque_mil_barris"].astype(float), "est_var_1s": e["estoque_mil_barris"].pct_change() * 100,
                      "est_var_4s": e["estoque_mil_barris"].pct_change(4) * 100, "est_media_4s": e["estoque_mil_barris"].rolling(4).mean(), "est_media_5a": e["media_5a"]}); E["est_desvio_pct"] = (E["est_nivel"] / E["est_media_5a"] - 1) * 100
    F = pd.merge_asof(F, E.sort_values("data_disponivel"), left_on="data", right_on="data_disponivel", direction="backward").drop(columns="data_disponivel")
    F["cal_mes"], F["cal_trimestre"], F["cal_dia_semana"] = F["data"].dt.month, F["data"].dt.quarter, F["data"].dt.dayofweek; F["cal_semana_ano"] = F["data"].dt.isocalendar().week.astype(int).clip(upper=52).values
    F["rel_brent_vs_media7"], F["rel_brent_vs_media30"] = (F["brent_preco"] / F["brent_media_7d"] - 1) * 100, (F["brent_preco"] / F["brent_media_30d"] - 1) * 100
    F["rel_dolar_vs_media30"], F["rel_est_vs_media4s"] = (F["dolar_d1"] / F["dolar_media_30d"] - 1) * 100, (F["est_nivel"] / F["est_media_4s"] - 1) * 100
    return F

FEAT = ["brent_var_1d", "brent_var_7d", "brent_var_30d", "brent_desv_pad_30d", "rel_brent_vs_media7", "rel_brent_vs_media30", "dolar_var_1d", "dolar_var_7d", "dolar_desv_pad_30d", "rel_dolar_vs_media30",
        "est_var_1s", "est_var_4s", "est_desvio_pct", "rel_est_vs_media4s", "cal_mes", "cal_trimestre", "cal_semana_ano", "cal_dia_semana"]
NOMES = {"brent_var_1d": "Variação de ontem", "brent_var_7d": "Variação em 7 dias", "brent_var_30d": "Variação em 30 dias", "brent_desv_pad_30d": "Agitação das últimas 6 semanas", "rel_brent_vs_media7": "Preço vs média de 7 dias",
         "rel_brent_vs_media30": "Preço vs média de 30 dias", "dolar_var_1d": "Dólar: variação de ontem", "dolar_var_7d": "Dólar: variação em 7 dias", "dolar_desv_pad_30d": "Dólar: agitação", "rel_dolar_vs_media30": "Dólar vs média de 30 dias",
         "est_var_1s": "Estoque: variação na semana", "est_var_4s": "Estoque: variação em 4 semanas", "est_desvio_pct": "Estoque vs normal da época", "rel_est_vs_media4s": "Estoque vs média de 4 semanas",
         "cal_mes": "Mês", "cal_trimestre": "Trimestre", "cal_semana_ano": "Semana do ano", "cal_dia_semana": "Dia da semana"}
GRUPO = {f: ("Brent" if f.startswith(("brent", "rel_brent")) else "Dólar" if f.startswith(("dolar", "rel_dolar")) else "Estoques" if f.startswith(("est", "rel_est")) else "Calendário") for f in FEAT}
CORG = {"Brent": COR["preco"], "Dólar": COR["dolar"], "Estoques": COR["estoque"], "Calendário": COR["neutro"]}

DS = construir_features(base, est); INI = int(np.argmax(DS[FEAT].notna().all(axis=1).values))
DS["agit15"] = np.log(DS["brent_var_1d"].rolling(15).std().shift(-15))                                         # alvo 1: agitação dos próximos 15 dias úteis (em log)
for h in (1, 5, 10, 15): DS[f"var_{h}"] = np.log(DS["brent_preco"].shift(-h) / DS["brent_preco"]) * 100         # variação do preço daqui a h dias úteis
PRECO, DATAS = DS["brent_preco"].values, DS["data"].values; X = DS[FEAT].values.astype(float)

# teste de vazamento: recalcular as features só com dados até cada data dá exatamente o mesmo resultado
rng = np.random.default_rng(SEMENTE); cols = [c for c in DS.columns if c not in ("data", "agit15") and not c.startswith("var_")]
for i in rng.choice(np.arange(INI + 5, len(DS) - 20), size=25, replace=False):
    t = DS.loc[i, "data"]; Ft = construir_features(base[base["data"] <= t], est[est["data_disponivel"] <= t])
    assert Ft.iloc[-1]["data"] == t and np.allclose(Ft.iloc[-1][cols].astype(float).values, DS.loc[i, cols].astype(float).values, equal_nan=True), f"vazamento em {t}"
print("Teste de vazamento: 25 datas conferidas, nenhuma feature enxerga o futuro.")

T_FIM, V_INI, V_FIM, TE_INI, FIM = pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-01"), DS["data"].max()
PER = {"validação": (V_INI, V_FIM), "teste": (TE_INI, FIM)}
periodo = np.select([DS["data"] <= T_FIM, DS["data"] <= V_FIM], ["treino", "validação"], "teste")
fig, ax = moldura((12.5, 4.6), "Treino, validação e teste: o tempo nunca é embaralhado", "O modelo aprende no treino, as configurações são escolhidas na validação e o teste só é olhado no final")
for nome, cor in [("treino", COR["preco"]), ("validação", COR["dolar"]), ("teste", COR["queda"])]:
    m = periodo == nome; ax.plot(DS["data"][m], DS["brent_preco"][m], color=cor, lw=1.3); ax.axvspan(DS["data"][m].min(), DS["data"][m].max(), color=cor, alpha=0.06)
    ax.text(DS["data"][m].min() + (DS["data"][m].max() - DS["data"][m].min()) / 2, 150, f"{nome.upper()}\\n{int((m & (np.arange(len(DS)) >= INI)).sum()):,} dias".replace(",", "."), ha="center", va="top", fontsize=10.5, fontweight="bold", color=cor)
usd(ax); ax.set_ylim(0, 155); plt.show()

def origens(a, b, h): return np.where(((DS["data"] >= a) & (DS["data"] <= b) & (DS.index >= INI) & (DS.index + h < len(DS))).values)[0]
def walk_forward(fit_predict, a, b, h, refit=REFIT):
    """a cada 63 dias úteis o modelo é treinado de novo, só com amostras cujo resultado já aconteceu; devolve a previsão de cada dia do período"""
    idx = origens(a, b, h); out = []
    for k in range(0, len(idx), refit):
        blk = idx[k:k + refit]; tr = np.arange(INI, blk[0] - h + 1); assert tr[-1] + h <= blk[0], "treino invade o futuro"; out.append(pd.Series(fit_predict(tr, blk), index=blk))
    return pd.concat(out)
''')
md("""
**Como o modelo é testado:** a cada 63 dias úteis (cerca de 3 meses) ele é treinado de novo e prevê os dias seguintes sem ver o futuro, como aconteceria na vida real. A divisão é **treino de 2006 a 2021**, **validação em 2022 e 2023** e **teste de 2024 a set/2026**.
""")

# ═══════════════════════════ 8. MODELO 1: AGITAÇÃO ═══════════════════════════
md("""
## 8. Modelo 1: quão agitado o mercado vai estar?

Um **LightGBM** prevê a agitação dos próximos 15 dias úteis (o quanto o preço deve mexer por dia). Ele **parte da agitação de hoje e aprende a correção**: em que situações ela costuma subir ou cair. Usa só **variações e comparações** (nunca o nível do preço), para não decorar épocas. Os parâmetros são escolhidos pelo **Optuna**, só na validação.

**Contra quem ele compete:** "a agitação de hoje continua" (repetir a agitação das últimas 6 semanas). Pela EDA essa regra já é forte, e o modelo só vale se acrescentar algo a ela.
""")
code('''
Y_AG = DS["agit15"].values; BASE_LOG = np.log(DS["brent_desv_pad_30d"].values); Y_DELTA = Y_AG - BASE_LOG; X_REL = X          # o modelo aprende a correção sobre "a agitação de hoje continua"
def lgbm_agit(params):
    def f(tr, pr):
        m = lgb.LGBMRegressor(**params, random_state=SEMENTE, verbose=-1, n_jobs=N_JOBS); m.fit(X_REL[tr], Y_DELTA[tr]); return BASE_LOG[pr] + m.predict(X_REL[pr])
    return f
def par_reg(trial):
    return dict(objective="regression", n_estimators=trial.suggest_int("n_estimators", 40, 400), learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True), num_leaves=trial.suggest_int("num_leaves", 3, 15),
                max_depth=trial.suggest_int("max_depth", 2, 5), min_child_samples=trial.suggest_int("min_child_samples", 30, 300), subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0), reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True), reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50, log=True))
arq = ART / "agitacao_corrigida.json"
if arq.exists(): P_AG = json.loads(arq.read_text(encoding="utf-8"))["params"]
else:
    def obj(trial):
        pr = walk_forward(lgbm_agit(par_reg(trial)), V_INI, V_FIM, H, refit=126); return float(np.abs(pr.values - Y_AG[pr.index]).mean())
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMENTE)); st.optimize(obj, n_trials=40)
    P_AG = par_reg(optuna.trial.FixedTrial(st.best_params)); arq.write_text(json.dumps({"params": P_AG, "erro_validacao": st.best_value}, indent=1), encoding="utf-8")

FASE = lambda v: pd.cut(np.exp(v), [0, 1.5, 3.5, np.inf], labels=["estável", "moderado", "turbulento"]).astype(str)
AG = {}
for per, (a, b) in PER.items():
    pr = walk_forward(lgbm_agit(P_AG), a, b, H); i = pr.index.values
    AG[per] = dict(i=i, real=Y_AG[i], modelo=pr.values, hoje=np.log(DS["brent_desv_pad_30d"].values[i]), data=DATAS[i])
linhas = []
for per in PER:
    d = AG[per]; fr = FASE(d["real"])
    linhas.append({"Período": per, "Erro típico do modelo": np.exp(np.abs(d["modelo"] - d["real"]).mean()) - 1, "Erro típico de 'a agitação de hoje continua'": np.exp(np.abs(d["hoje"] - d["real"]).mean()) - 1,
                   "Acerta a fase (modelo)": (FASE(d["modelo"]) == fr).mean() * 100, "Acerta a fase ('hoje continua')": (FASE(d["hoje"]) == fr).mean() * 100, "Acerta a fase (sempre 'moderado')": (fr == "moderado").mean() * 100})
TAB_AG = pd.DataFrame(linhas)
display(TAB_AG.style.hide(axis="index").format({c: "{:.0%}" for c in TAB_AG.columns if "Erro" in c} | {c: "{:.0f}%" for c in TAB_AG.columns if "Acerta" in c}))
''')
code('''
d = AG["teste"]; dt = pd.to_datetime(d["data"])
fig, ax = moldura((12.5, 5.0), "O modelo acompanha a agitação do mercado, mas reage depois que ela sobe", "Teste (2024 a set/2026): agitação diária média dos 15 dias úteis seguintes, em % ao dia. Em cada dia, a previsão usa só o que se sabia até ali")
ax.axhspan(0, 1.5, color=COR["calmo"], alpha=0.10); ax.axhspan(3.5, 12, color=COR["agitado"], alpha=0.08)
ax.fill_between(dt, np.exp(d["real"]), color="#c9ced6", label="agitação que veio"); ax.plot(dt, np.exp(d["modelo"]), color=COR["preco"], lw=1.8, label="previsão do modelo"); ax.plot(dt, np.exp(d["hoje"]), color=COR["dolar"], lw=1.3, ls="--", label="'a agitação de hoje continua'")
ax.set_ylim(0, 8); pct(ax); ax.set_ylabel("variação média por dia"); ax.legend(loc="upper left", fontsize=9.5)
for yy, txt, cor in [(0.75, "estável", COR["calmo"]), (7.2, "turbulento", COR["agitado"])]: ax.text(dt.max(), yy, " " + txt, fontsize=10, color=cor, fontweight="bold", va="center")
plt.show()
''')
code('''
FS = ["estável", "moderado", "turbulento"]; d = AG["teste"]; fr, fm = FASE(d["real"]), FASE(d["modelo"])
M = np.array([[np.mean(fm[fr == i] == j) * 100 if (fr == i).any() else np.nan for j in FS] for i in FS])
fig, axs = moldura((12.5, 4.9), "Acerto da fase do mercado: o modelo não supera 'a agitação de hoje continua'", "Teste (2024 a set/2026). Esquerda: para cada fase que veio, quanto o modelo previu de cada fase. Direita: % de dias em que a fase prevista foi a certa", ncols=2, gridspec_kw={"width_ratios": [1, 1.1]})
ax = axs[0]; ax.imshow(np.nan_to_num(M), cmap="Blues", vmin=0, vmax=100); ax.grid(False)
for i in range(3):
    for j in range(3): ax.text(j, i, f"{M[i, j]:.0f}%" if not np.isnan(M[i, j]) else "—", ha="center", va="center", fontsize=13, fontweight="bold", color="white" if M[i, j] > 55 else "#222")
ax.set_xticks(range(3)); ax.set_xticklabels([f"previu\\n{f}" for f in FS]); ax.set_yticks(range(3)); ax.set_yticklabels([f"veio\\n{f}" for f in FS])
ax = axs[1]; rot = ["Modelo", "'A agitação de hoje\\ncontinua'", "Sempre\\n'moderado'"]; val = [TAB_AG.loc[1, "Acerta a fase (modelo)"], TAB_AG.loc[1, "Acerta a fase ('hoje continua')"], TAB_AG.loc[1, "Acerta a fase (sempre 'moderado')"]]
ax.bar(rot, val, color=[COR["preco"], COR["dolar"], "#b5bcc6"], width=0.55)
for k, v in enumerate(val): ax.text(k, v + 1.5, f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(0, 100); pct(ax); plt.show()
''')
md("""
**O que vemos (teste)**
- A agitação prevista segue a que de fato veio, mas **reage depois**: nos saltos de 2026 a previsão só sobe depois que a agitação já subiu.
- O modelo erra em média **37%**, contra **38%** de "a agitação de hoje continua": o ganho é **pequeno**. Na fase do mercado acerta **56%** dos dias, contra 59% da âncora e 56% de dizer sempre "moderado".
- Ele reconhece parte das fases extremas (23% dos dias estáveis e 39% dos turbulentos), mas ainda confunde muitos deles com "moderado".
""")

# ═══════════════════════════ 9. FAIXA DE PREÇO ═══════════════════════════
md("""
## 9. Entre quais preços ele deve ficar?

O preço de hoje é a **âncora**. A agitação prevista define o **tamanho da faixa**:

`faixa = preço de hoje ± agitação prevista × √(dias úteis) × fator`

O **fator** é calibrado na validação para que a faixa de 80% cubra 80% dos casos. A faixa pode vir da previsão do modelo ou da regra "a agitação de hoje continua". Escolhemos pela **menor erro na validação**.
""")
code('''
for h in (5, 10, 15): pass
ESC = "modelo" if np.abs(AG["validação"]["modelo"] - AG["validação"]["real"]).mean() <= np.abs(AG["validação"]["hoje"] - AG["validação"]["real"]).mean() else "hoje"
NOME_ESC = {"modelo": "previsão do modelo", "hoje": "agitação de hoje"}[ESC]
def pontos(per, fonte, hs=(5, 10, 15)):
    d = AG[per]; sg = np.exp(d[fonte]); L = []
    for h in hs:
        v = DS[f"var_{h}"].values[d["i"]] if h != 5 and h != 10 else DS[f"var_{h}"].values[d["i"]]
        ok = ~np.isnan(v); L.append(pd.DataFrame({"i": d["i"][ok], "h": h, "var": v[ok], "sig": sg[ok], "vol": DS["brent_desv_pad_30d"].values[d["i"][ok]]}))
    R = pd.concat(L, ignore_index=True); R["z"] = R["var"] / (R["sig"] * np.sqrt(R["h"])); return R
FONTE = "modelo" if ESC == "modelo" else "hoje"
RV, RT = pontos("validação", FONTE), pontos("teste", FONTE); RVh, RTh = pontos("validação", "hoje"), pontos("teste", "hoje")
Q80 = np.quantile(RV["z"], [0.10, 0.90]); Q95 = np.quantile(RV["z"], [0.025, 0.975]); Q80h = np.quantile(RVh["z"], [0.10, 0.90])
def cobre(R, q): return (R["z"] >= q[0]) & (R["z"] <= q[1])
fase_t = pd.cut(RT["vol"], [-np.inf, 1.5, 3.5, np.inf], labels=FS).astype(str); r15 = RT["h"] == 15
fixa = {h: np.quantile(RV.loc[RV["h"] == h, "var"], [0.10, 0.90]) for h in (5, 10, 15)}
dentro_fixa = (RT["var"] >= RT["h"].map(lambda h: fixa[h][0])) & (RT["var"] <= RT["h"].map(lambda h: fixa[h][1]))
dentro = cobre(RT, Q80)
COB = pd.DataFrame({"Faixa": ["fixa (sem olhar a agitação)", f"pela agitação ({NOME_ESC})"], **{f: [dentro_fixa[r15 & (fase_t == f)].mean() * 100, dentro[r15 & (fase_t == f)].mean() * 100] for f in FS}, "Todos os dias": [dentro_fixa[r15].mean() * 100, dentro[r15].mean() * 100]})
display(COB.style.hide(axis="index").format({c: "{:.0f}%" for c in COB.columns if c != "Faixa"}))
print(f"Forma escolhida: {NOME_ESC}. Quantis do fator (80%): {Q80.round(2)}. Cobertura em 15 dias úteis no teste; a meta é 80%.")
''')
code('''
# backtest em 15 dias úteis no teste: a faixa prevista no dia t, desenhada na data t+15, contra o preço que de fato veio
d = AG["teste"]; i = d["i"][~np.isnan(DS["var_15"].values[d["i"]])]; sg = pd.Series(np.exp(d[FONTE]), index=d["i"]).loc[i].values
p0 = PRECO[i]; lo, hi = p0 * np.exp(Q80[0] * sg * np.sqrt(15) / 100), p0 * np.exp(Q80[1] * sg * np.sqrt(15) / 100); alvo = pd.to_datetime(DATAS[i + 15]); real = PRECO[i + 15]; fora = (real < lo) | (real > hi)
fig, axs = moldura((13.5, 5.2), f"A faixa de 80% cobriu {(~fora).mean() * 100:.0f}% dos dias e protege mais nas fases turbulentas", "Esquerda: preço real e a faixa de 80% prevista 15 dias úteis antes (teste). Direita: em quantos % dos dias o preço ficou dentro da faixa, por fase do mercado", ncols=2, gridspec_kw={"width_ratios": [2, 1]})
ax = axs[0]; ax.fill_between(alvo, lo, hi, color=COR["preco"], alpha=0.20, label="faixa de 80%"); ax.plot(alvo, real, color="#111", lw=1.3, label="preço real"); ax.scatter(alvo[fora], real[fora], s=12, color=COR["queda"], zorder=5, label="fora da faixa")
usd(ax); ax.set_ylim(40, 175); ax.legend(loc="upper left", fontsize=9.5); ax.text(0.99, 0.04, f"dentro da faixa em {(~fora).mean() * 100:.0f}% dos dias (meta: 80%)", transform=ax.transAxes, ha="right", fontsize=10)
ax = axs[1]; w = 0.36
for j, (lin, cor, nome) in enumerate([(0, "#b5bcc6", "faixa fixa"), (1, COR["preco"], "faixa pela agitação")]):
    v = [COB.loc[lin, f] for f in FS]; ax.bar(np.arange(3) + (j - 0.5) * w, v, width=w, color=cor, label=nome)
    for xx, vv in zip(np.arange(3) + (j - 0.5) * w, v): ax.text(xx, vv + 1.5, f"{vv:.0f}", ha="center", fontsize=9)
ax.axhline(80, color=COR["queda"], ls="--", lw=1.2); ax.set_xticks(range(3)); ax.set_xticklabels(FS); ax.set_ylim(0, 108); pct(ax); ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2); plt.show()
''')
md("""
**O que vemos (teste)**
- Em 15 dias úteis, a faixa de 80% cobriu **76%** dos dias (a meta é 80%).
- Nas fases **turbulentas** cobriu **73%**, contra **44%** de uma faixa fixa. Nas **moderadas**, 81%. Nas **estáveis**, 66% (a fixa cobre 91%): a faixa fica estreita demais quando o mercado está estável.
- Em resumo: a faixa protege mais onde mais importa, nas fases turbulentas, ao custo de ficar apertada nas estáveis.
""")

# ═══════════════════════════ 10. MODELO 2: SOBE OU CAI ═══════════════════════════
md("""
## 10. Modelo 2: o preço vai subir ou cair?

Um **LightGBM classificador** devolve a **probabilidade de o preço estar acima do de hoje** daqui a 1, 5 e 15 dias úteis. Os parâmetros vêm do Optuna, só na validação. O resultado é medido contra o acaso (**AUC de 0,50 = chute**; 1,00 = perfeito) e contra "apostar sempre que sobe".
""")
code('''
HD = (1, 5, 15); Y_UP = {h: np.where(np.isnan(DS[f"var_{h}"].values), np.nan, (DS[f"var_{h}"].values > 0).astype(float)) for h in HD}
def lgbm_clf(params, Xm, ym):
    def f(tr, pr):
        m = lgb.LGBMClassifier(**params, random_state=SEMENTE, verbose=-1, n_jobs=N_JOBS); m.fit(Xm[tr], ym[tr].astype(int)); return m.predict_proba(Xm[pr])[:, 1]
    return f
def par_clf(trial):
    p = par_reg(trial); p["objective"] = "binary"; return p
arq = ART / "direcao.json"
if arq.exists(): P_DIR = json.loads(arq.read_text(encoding="utf-8"))["params"]
else:
    def obj(trial):
        p = par_clf(trial); a_ = []
        for h in HD:
            pr = walk_forward(lgbm_clf(p, X, Y_UP[h]), V_INI, V_FIM, h, refit=126); a_.append(roc_auc_score(Y_UP[h][pr.index].astype(int), pr.values))
        return -float(np.mean(a_))
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMENTE)); st.optimize(obj, n_trials=30)
    P_DIR = par_clf(optuna.trial.FixedTrial(st.best_params)); arq.write_text(json.dumps({"params": P_DIR, "auc_validacao": -st.best_value}, indent=1), encoding="utf-8")

def ic_auc(y, p, h, n=500):
    rg = np.random.default_rng(SEMENTE); m = len(y); bl = max(2 * h, 10); nb = max(2, m // bl); a_ = []
    for _ in range(n):
        ix = np.concatenate([np.arange(s, s + bl) for s in rg.integers(0, m - bl + 1, nb)])
        if len(np.unique(y[ix])) == 2: a_.append(roc_auc_score(y[ix], p[ix]))
    return np.percentile(a_, 2.5), np.percentile(a_, 97.5)
DIR = {}
for h in HD:
    for per, (a, b) in PER.items():
        pr = walk_forward(lgbm_clf(P_DIR, X, Y_UP[h]), a, b, h); y_ = Y_UP[h][pr.index].astype(int); lo, hi = ic_auc(y_, pr.values, h)
        DIR[(per, h)] = dict(i=pr.index.values, y=y_, p=pr.values, auc=roc_auc_score(y_, pr.values), lo=lo, hi=hi, acc=((pr.values > 0.5) == (y_ == 1)).mean() * 100, sobe=max(y_.mean(), 1 - y_.mean()) * 100)
TAB_DIR = pd.DataFrame([{"Horizonte": f"{h} dia(s) útil(eis)", "Período": per, "AUC": v["auc"], "IC 95%": f"[{v['lo']:.2f}; {v['hi']:.2f}]", "Acerto do modelo": v["acc"], "Acerto de 'apostar no lado mais comum'": v["sobe"]} for (per, h), v in DIR.items()])
display(TAB_DIR.style.hide(axis="index").format({"AUC": "{:.3f}", "Acerto do modelo": "{:.1f}%", "Acerto de 'apostar no lado mais comum'": "{:.1f}%"}))
''')
md("""
**O que vemos (teste)**
- O AUC fica em **0,47, 0,47 e 0,45** (1, 5 e 15 dias úteis; 0,50 é chute): **não é melhor que o acaso**.
- O acerto do modelo (46% a 47%) fica **abaixo** de apostar sempre no lado mais comum (50% a 51%).
- Na validação o AUC chegou a 0,58 em 15 dias, mas **não se repetiu no teste**: foi acaso da escolha dos parâmetros.
- Conclusão: **a direção do preço não é previsível com estes dados**. Por isso o modelo devolve probabilidades perto de 50%, que significam "sem opinião".
""")

# ═══════════════════════════ 11. SHAP ═══════════════════════════
md("""
## 11. Por que o modelo diz isso? (SHAP)

O SHAP mostra **quanto cada informação empurrou a previsão para cima ou para baixo**. Cada modelo é treinado só com dados anteriores ao teste e explicado nos dias do teste, ou seja, mostra o que ele olha em dados que nunca viu. O cálculo é o **TreeSHAP do próprio LightGBM**.
""")
code('''
T0 = int(np.argmax((DS["data"] >= TE_INI).values)); ROT = [NOMES[f] for f in FEAT]
def shap_de(tipo, h):
    if tipo == "agit": m = lgb.LGBMRegressor(**P_AG, random_state=SEMENTE, verbose=-1, n_jobs=N_JOBS); yv, hh = Y_DELTA, H
    else: m = lgb.LGBMClassifier(**P_DIR, random_state=SEMENTE, verbose=-1, n_jobs=N_JOBS); yv, hh = Y_UP[h], h
    tr = np.arange(INI, T0 - hh + 1); m.fit(X[tr], yv[tr] if tipo == "agit" else yv[tr].astype(int)); te = origens(TE_INI, FIM, hh); sv = m.predict(X[te], pred_contrib=True)[:, :-1]                      # SHAP exato (TreeSHAP) calculado pelo próprio LightGBM
    return dict(m=m, sv=np.asarray(sv), X=X[te], hoje=X[-1:])
SH = {"agit": shap_de("agit", H), "dir1": shap_de("dir", 1), "dir15": shap_de("dir", 15)}
def barras(ax, d, titulo, n=8):
    imp = pd.Series(np.abs(d["sv"]).mean(axis=0), index=FEAT).sort_values().tail(n); ax.barh([NOMES[f] for f in imp.index], imp.values, color=[CORG[GRUPO[f]] for f in imp.index]); ax.set_title(titulo, fontsize=11.5, loc="left", fontweight="bold"); ax.grid(axis="y", visible=False)
fig, axs = moldura((14, 4.9), "A agitação recente domina a previsão de agitação; sobe ou cai depende de um pouco de tudo", "Peso médio de cada informação nos dias do teste (quanto maior a barra, mais ela mexe na previsão). Cores: azul = Brent, laranja = dólar, roxo = estoques, cinza = calendário", ncols=3)
barras(axs[0], SH["agit"], "Agitação: correção sobre hoje"); barras(axs[1], SH["dir1"], "Sobe ou cai: amanhã"); barras(axs[2], SH["dir15"], "Sobe ou cai: em 15 dias úteis")
for a in axs: a.tick_params(axis="y", labelsize=9); a.set_xlabel("")
fig.subplots_adjust(wspace=0.75, left=0.14); plt.show()
''')
code('''
d = SH["agit"]; sv, Xs = d["sv"], d["X"]; ordem = np.argsort(np.abs(sv).mean(axis=0))[::-1][:7][::-1]; rg = np.random.default_rng(SEMENTE)
fig, ax = moldura((11.5, 5.4), "Quando a agitação está alta, o modelo espera que ela diminua", "Cada ponto é um dia do teste. À direita do zero, a informação faz o modelo prever mais agitação do que a de hoje. Cor: valor da informação naquele dia (azul = baixo, vermelho = alto)")
for k, j in enumerate(ordem):
    ax.scatter(sv[:, j], k + rg.uniform(-0.28, 0.28, len(sv)), c=pd.Series(Xs[:, j]).rank(pct=True).values, cmap="coolwarm", s=10, alpha=0.8, linewidths=0)
ax.axvline(0, color="#555", lw=1); ax.set_yticks(range(len(ordem))); ax.set_yticklabels([ROT[j] for j in ordem]); ax.grid(axis="x", color="#e6e9ed"); ax.grid(axis="y", visible=False); ax.set_xlabel("correção sobre a agitação de hoje (escala log)"); plt.show()
''')
md("""
**O que vemos**
- **Agitação:** a **agitação das últimas 6 semanas** é de longe a informação mais importante. O gráfico mostra o porquê: quando ela está **alta** (vermelho), o modelo **corrige para baixo**; quando está **baixa** (azul), corrige para cima. Ele aprendeu que a agitação **volta à média**.
- **Sobe ou cai:** o peso se espalha por muitas informações, sem uma dominante (variação de ontem, estoque, calendário...). É o padrão de um modelo que não achou sinal.
""")

# ═══════════════════════════ 12. PREVISÃO DE HOJE ═══════════════════════════
md("## 12. A previsão de hoje")
code('''
# treino final com todos os dados disponíveis (só amostras cujo resultado já aconteceu) e previsão a partir do último fechamento
N = len(DS); DATA_HOJE = DS["data"].iloc[-1]; P0 = float(PRECO[-1]); xh = X[-1:]
M_AG = lgb.LGBMRegressor(**P_AG, random_state=SEMENTE, verbose=-1, n_jobs=N_JOBS).fit(X[np.arange(INI, N - H)], Y_DELTA[np.arange(INI, N - H)])
M_DIR = {h: lgb.LGBMClassifier(**P_DIR, random_state=SEMENTE, verbose=-1, n_jobs=N_JOBS).fit(X[np.arange(INI, N - h)], Y_UP[h][np.arange(INI, N - h)].astype(int)) for h in HD}
# calibração das probabilidades (Platt) com as previsões fora da amostra da validação e do teste
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
CAL = {h: LogisticRegression(C=1e6).fit(lg(np.r_[DIR[("validação", h)]["p"], DIR[("teste", h)]["p"]]).reshape(-1, 1), np.r_[DIR[("validação", h)]["y"], DIR[("teste", h)]["y"]]) for h in HD}
Zf = np.r_[RV["z"], RT["z"]]; QF = {"80%": np.quantile(Zf, [0.10, 0.90]), "95%": np.quantile(Zf, [0.025, 0.975])}
vol_hoje = float(DS["brent_desv_pad_30d"].iloc[-1]); sg_hoje = float(vol_hoje * np.exp(M_AG.predict(xh)[0])) if ESC == "modelo" else vol_hoje
ref = np.exp(Y_AG[INI:N - H - 1]); pct_hoje = float((ref < sg_hoje).mean() * 100); fase_hoje = FASE(np.log(np.array([sg_hoje])))[0]
datas = pd.bdate_range(DATA_HOJE + pd.Timedelta(days=1), periods=H); PREV = pd.DataFrame({"h": np.arange(1, H + 1), "data": datas})
for k, q in QF.items(): PREV[f"inf_{k[:-1]}"], PREV[f"sup_{k[:-1]}"] = P0 * np.exp(q[0] * sg_hoje * np.sqrt(PREV["h"]) / 100), P0 * np.exp(q[1] * sg_hoje * np.sqrt(PREV["h"]) / 100)
PROB = {h: float(CAL[h].predict_proba(lg(M_DIR[h].predict_proba(xh)[:, 1]).reshape(-1, 1))[0, 1]) for h in HD}
print(f"{DATA_HOJE:%d/%m/%Y}: preço US$ {P0:.2f} | agitação prevista {sg_hoje:.2f}% ao dia ({fase_hoje}, mais agitado que {pct_hoje:.0f}% da história) | agitação das últimas 6 semanas {vol_hoje:.2f}%")
print("Probabilidade de o preço estar acima do de hoje:", {f"{h} dia(s)": f"{p:.0%}" for h, p in PROB.items()})
''')
code('''
hist = DS.iloc[-90:]; fig, axs = moldura((15, 5.2), f"Hoje o mercado vem turbulento ({vol_hoje:.1f}% ao dia); o modelo prevê {sg_hoje:.1f}% ({fase_hoje})", "Esquerda: preço de hoje (âncora) e faixa de preço. Centro: agitação prevista na história. Direita: chance de o preço estar acima do de hoje", ncols=3, gridspec_kw={"width_ratios": [2.2, 1.1, 1.1]})
ax = axs[0]; xf = PREV["h"].values; ax.plot(np.arange(-len(hist) + 1, 1), hist["brent_preco"].values, color="#111", lw=1.6)
ax.fill_between(xf, PREV["inf_95"], PREV["sup_95"], color=COR["preco"], alpha=0.10, label="95% dos casos"); ax.fill_between(xf, PREV["inf_80"], PREV["sup_80"], color=COR["preco"], alpha=0.22, label="80% dos casos"); ax.plot([0, H], [P0, P0], color=COR["preco"], lw=2, label="âncora: preço de hoje")
ax.axvline(0, color=COR["neutro"], ls=":"); usd(ax); ax.set_xlabel("dias úteis (0 = último fechamento)"); ax.legend(loc="upper left", fontsize=9)
ax.annotate(f"US$ {PREV['inf_80'].iloc[-1]:.0f}", (H, PREV["inf_80"].iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=9.5, va="center"); ax.annotate(f"US$ {PREV['sup_80'].iloc[-1]:.0f}", (H, PREV["sup_80"].iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=9.5, va="center"); ax.set_xlim(-90, H + 12)
ax = axs[1]; ax.hist(np.log10(ref), bins=40, color="#c4cad2", edgecolor="white", lw=0.4); ax.axvline(np.log10(sg_hoje), color=COR["queda"], lw=2.4); ax.set_yticks([]); ax.grid(False); ax.spines["left"].set_visible(False)
ax.set_xticks(np.log10([0.7, 1.5, 3.5, 8])); ax.set_xticklabels(["0,7%", "1,5%", "3,5%", "8%"]); ax.set_xlabel("agitação diária"); ax.text(0.03, 0.96, f"previsto: {sg_hoje:.1f}%", transform=ax.transAxes, color=COR["queda"], ha="left", va="top", fontsize=11, fontweight="bold")
ax = axs[2]; v = [PROB[h] * 100 for h in HD]; ax.bar([f"{h} dia\\nútil" if h == 1 else f"{h} dias\\núteis" for h in HD], v, color=[COR["alta"] if x >= 50 else COR["queda"] for x in v], width=0.55); ax.axhline(50, color="#222", ls="--", lw=1.2)
for k, x in enumerate(v): ax.text(k, x + 4, f"{x:.0f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 100); pct(ax); ax.text(0.5, 0.80, "linha tracejada: 50%\\n(sem opinião)", transform=ax.transAxes, ha="center", fontsize=9.5, color="#555"); plt.show()
''')
code('''
c = M_AG.predict(xh, pred_contrib=True)[0]; efeito = (np.exp(c[:-1]) - 1) * 100; ajuste = (np.exp(c[-1]) - 1) * 100; pm = vol_hoje * np.exp(c.sum())      # efeito de cada informação, em % sobre a agitação de hoje
ordem = list(np.argsort(np.abs(efeito))[::-1][:5][::-1]); nomes = ["Ajuste médio do modelo"] + [NOMES[FEAT[j]] for j in ordem]; vals = [ajuste] + [efeito[j] for j in ordem]
fig, ax = moldura((11.5, 4.8), f"Partindo de {vol_hoje:.1f}% (agitação das últimas 6 semanas), o modelo prevê {pm:.1f}%", "Efeito de cada informação sobre a agitação prevista, em %. Barra à direita = o modelo espera mais agitação do que a de hoje")
ax.barh(nomes, vals, color=[COR["queda"] if v > 0 else COR["calmo"] for v in vals], height=0.62)
for k, v in enumerate(vals): ax.text(v + (0.8 if v > 0 else -0.8), k, f"{v:+.0f}%", va="center", ha="left" if v > 0 else "right", fontsize=10, fontweight="bold")
ax.set_xlim(min(vals) - 3.5, max(max(vals), 0) + 3); ax.axvline(0, color="#555", lw=1); ax.grid(axis="x", color="#e6e9ed"); ax.grid(axis="y", visible=False); ax.set_xlabel("efeito sobre a agitação prevista"); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%")); plt.show()
''')

md("""
**Leitura de hoje (29/09/2026)**
- O mercado **vem turbulento** (3,9% ao dia nas últimas 6 semanas), e o modelo espera que a agitação **diminua um pouco**, para 3,1% ao dia.
- Faixa de 80% em 15 dias úteis: **de US$ 97 a US$ 134**, em torno do preço de hoje (US$ 114).
- Chance de o preço estar acima do de hoje: 53%, 51% e 49% (1, 5 e 15 dias úteis). Perto de 50% significa **sem opinião**.
""")

code('''
# dados da previsão de hoje, para alimentar o app (simulador e painel)
APP = ART.parent.parent / "02_App" / "dados"; APP.mkdir(parents=True, exist_ok=True)
json.dump({"data_hoje": f"{DATA_HOJE:%Y-%m-%d}", "preco_hoje": P0, "agitacao_6s": vol_hoje, "agitacao_prevista": sg_hoje, "fase_hoje": str(fase_hoje), "percentil_hoje": pct_hoje,
           "fatores": {k: [float(q[0]), float(q[1])] for k, q in QF.items()}, "prob_acima": {str(h): p for h, p in PROB.items()}, "limites_fase": [1.5, 3.5],
           "distribuicao_agitacao": {str(q): float(np.quantile(ref, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
           "historico": {"datas": [f"{d:%Y-%m-%d}" for d in pd.to_datetime(DATAS[-120:])], "preco": [float(x) for x in PRECO[-120:]]},
           "previsao": [{"h": int(r.h), "data": f"{r.data:%Y-%m-%d}", "inf80": float(r.inf_80), "sup80": float(r.sup_80), "inf95": float(r.inf_95), "sup95": float(r.sup_95)} for r in PREV.itertuples()],
           "replay": {"fatores_val": {"80%": [float(Q80[0]), float(Q80[1])], "95%": [float(Q95[0]), float(Q95[1])]}, "datas": [f"{x:%Y-%m-%d}" for x in pd.to_datetime(AG["teste"]["data"])], "sigma": [round(float(x), 4) for x in np.exp(AG["teste"]["modelo"])]},
           "efeitos_hoje": {NOMES[FEAT[j]]: float(efeito[j]) for j in np.argsort(np.abs(efeito))[::-1][:5]}}, open(APP / "app_dados.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("dados do app gravados em", APP)
''')

# ═══════════════════════════ 13. CONCLUSÃO ═══════════════════════════
md("""
## 13. Conclusão

- **O preço de hoje é a âncora.** Nenhum modelo supera "amanhã = hoje" como previsão de preço, e **sobe ou cai** não é previsível com estes dados (AUC de 0,45 a 0,47 no teste).
- **O que dá para prever é a agitação.** O modelo melhora pouco a âncora ("a agitação de hoje continua"), mas aprende que a agitação alta tende a diminuir e entrega uma **faixa de preço** que cobre 76% dos casos (73% nas fases turbulentas, contra 44% de uma faixa fixa).
- **Como usar:** a faixa como margem de segurança em análises de compra, venda e proteção; a fase do mercado como termômetro; o SHAP para explicar por que a previsão é essa.
- **Limites:** o modelo reage depois das viradas bruscas, a faixa fica estreita demais nas fases estáveis, e choques geopolíticos, como os de 2026, escapam de qualquer faixa.
""")

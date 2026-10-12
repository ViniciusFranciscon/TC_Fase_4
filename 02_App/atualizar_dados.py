# -*- coding: utf-8 -*-
"""Atualiza os números do app (02_App/dados.json) a partir do notebook executado e dos parquets.
Uso: executar o notebook (ele grava 02_App/dados_notebook.json) e depois  python 02_App/atualizar_dados.py"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
APP = AQUI
NB = json.load(open(RAIZ / "01_Notebook" / "TC_Petroleo.ipynb", encoding="utf-8"))
D = json.load(open(APP / "dados_notebook.json", encoding="utf-8"))


# ───────── tabelas do notebook (números reais) ─────────
def linhas(cel):
    for o in NB["cells"][cel]["outputs"]:
        h = o.get("data", {}).get("text/html")
        if h:
            h = re.sub(r"<style.*?</style>", "", "".join(h), flags=re.S)
            return [[re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, flags=re.S)]
                    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", h, flags=re.S)]
    raise SystemExit(f"sem tabela na célula {cel}")


num = lambda s: float(s.replace("%", "").replace(",", ".").strip())

# célula 34: erro e acerto da fase | 39: cobertura da faixa | 43: AUC de direção
t34 = {r[0]: r[1:] for r in linhas(34) if len(r) > 2 and r[0] in ("validação", "teste")}
t39 = {r[0]: r[1:] for r in linhas(39) if len(r) > 2 and r[0].split()[0] in ("fixa", "pela")}
t43 = [r for r in linhas(43) if len(r) > 2 and "dia" in r[0]]
fixa = [num(x) for x in next(v for k, v in t39.items() if k.startswith("fixa"))]
modelo = [num(x) for x in next(v for k, v in t39.items() if k.startswith("pela"))]
D["metricas"] = {
    "agitacao": {p: {"erro_modelo": num(v[0]), "erro_ancora": num(v[1]), "fase_modelo": num(v[2]), "fase_ancora": num(v[3]), "fase_moderado": num(v[4])} for p, v in t34.items()},
    "cobertura": {"fases": ["estável", "moderado", "turbulento"], "faixa_fixa": fixa[:3], "faixa_modelo": modelo[:3], "geral_fixa": fixa[3], "geral_modelo": modelo[3]},
    "auc": [{"horizonte": int(r[0].split()[0]), "periodo": r[1], "auc": num(r[2]), "ic": [num(x) for x in re.findall(r"[\d.]+", r[3])],
             "acerto": num(r[4]), "acerto_lado_comum": num(r[5])} for r in t43],
    # tamanhos exibidos no gráfico de divisão do notebook (célula 31); teste com alvo de 15 dias = 693 - 15
    "split": {"treino": {"ini": "2006", "fim": "2021", "dias": 4096}, "validacao": {"ini": "2022", "fim": "2023", "dias": 505},
              "teste": {"ini": "2024", "fim": "set/2026", "dias": 693, "dias_alvo15": 678}},
    "vazamento_datas": 25,
}

# ───────── série semanal para os gráficos ─────────
b = pd.read_parquet(RAIZ / "00_Base_TC" / "base_modelagem_diaria.parquet").sort_values("data")
b = b[b["data"].dt.dayofweek < 5].reset_index(drop=True)          # só dias úteis, como no notebook
ag = (b["preco_usd"].pct_change() * 100).rolling(30).std()
assert abs(ag.iloc[-1] - D["agitacao_6s"]) < 1e-6, "agitação do app diferente da do notebook"
s = b.iloc[30::5].reset_index(drop=True); a = ag.iloc[30::5].reset_index(drop=True)
D["serie"] = {"datas": [f"{d:%Y-%m-%d}" for d in s["data"]], "preco": [round(float(x), 2) for x in s["preco_usd"]], "agit": [round(float(x), 2) for x in a]}
D["historico"]["datas"] = [f"{d:%Y-%m-%d}" for d in b["data"].iloc[-120:]]


# ───────── página Mercado: como o preço se move dia a dia ─────────
pr = b["preco_usd"]; rv = pr.pct_change() * 100; rr = rv.dropna()
fs = pd.cut(ag, [-np.inf, 1.5, 3.5, np.inf], labels=[0, 1, 2]).astype(float)
cont, _ = np.histogram(rr.clip(-9.999, 9.999), np.arange(-10, 10.5, 0.5))
up = b.assign(var=rv).nlargest(5, "var"); dn = b.assign(var=rv).nsmallest(5, "var")
mov = lambda d: [{"data": f"{x.data:%Y-%m-%d}", "preco": round(float(x.preco_usd), 2), "var": round(float(x.var), 1)} for x in d.itertuples()]
mov15 = (pr.shift(-15) / pr - 1).abs() * 100
ok = fs.notna() & mov15.notna()
anos = {}
for y, g in fs[fs.notna()].groupby(b["data"].dt.year):
    anos[int(y)] = [round(float((g == k).mean() * 100), 1) for k in (0, 1, 2)]
turb = (ag >= 3.5).astype(int); grp = (turb.diff() != 0).cumsum(); ep = turb[ag.notna()].groupby(grp[ag.notna()]).agg(["first", "size"]); ep = ep[ep["first"] == 1]["size"]
grande = rr.abs() > 3
hf = {}
for h in range(1, 16):
    ch = ((pr.shift(-h) / pr - 1) * 100)
    hf[h] = [[round(float(ch[(fs == k) & ch.notna()].quantile(q)), 2) for q in (0.1, 0.5, 0.9)] for k in (0, 1, 2)]
D["mercado"] = {
    "hist": {"bins": [round(float(x), 1) for x in np.arange(-10, 10, 0.5)], "pct": [round(float(c / len(rr) * 100), 2) for c in cont]},
    "dia": {"mediana_abs": round(float(rr.abs().median()), 2), "acima": {str(t): round(float((rr.abs() > t).mean() * 100), 1) for t in (1, 2, 3, 5, 8)}, "sobe": round(float((rr > 0).mean() * 100), 1), "dias": int(len(rr))},
    "maiores_altas": mov(up), "maiores_quedas": mov(dn),
    "fases": {"tempo": [round(float((fs[ok] == k).mean() * 100), 1) for k in (0, 1, 2)], "mov15_medio": [round(float(mov15[ok & (fs == k)].mean()), 1) for k in (0, 1, 2)]},
    "por_ano": anos,
    "memoria": {"p_grande_apos_grande": round(float(grande.shift(-1)[grande].mean() * 100), 1), "p_grande": round(float(grande.mean() * 100), 1),
                "episodios": int(len(ep)), "mediana_dias": int(ep.median()), "maior_dias": int(ep.max())},
    "faixa_historica": hf,
    "eventos": [{"data": "-".join(r_[0].split("/")[::-1]), "texto": r_[1], "apos5": r_[2]} for r_ in linhas(6)[1:]],
}

# ───────── replay: a faixa que o modelo daria em cada dia do teste ─────────
if "replay" in D:
    R = D["replay"]; datas = pd.to_datetime(R["datas"]); pos = {d: i for i, d in enumerate(b["data"])}
    ini = pos[datas[0]] - 45
    R["serie"] = {"datas": [f"{d:%Y-%m-%d}" for d in b["data"].iloc[ini:]], "preco": [round(float(x), 2) for x in b["preco_usd"].iloc[ini:]], "offset": 45}
    q = R["fatores_val"]["80%"]; dentro = []
    for d, sg in zip(datas, R["sigma"]):
        i = pos[d]; p0, p1 = float(pr.iloc[i]), float(pr.iloc[i + 15])
        dentro.append(p0 * np.exp(q[0] * sg * np.sqrt(15) / 100) <= p1 <= p0 * np.exp(q[1] * sg * np.sqrt(15) / 100))
    R["cobertura80"] = round(float(np.mean(dentro) * 100), 1)
    assert abs(R["cobertura80"] - D["metricas"]["cobertura"]["geral_modelo"]) < 1.5, ("replay não reproduz a cobertura do notebook", R["cobertura80"])
    print("replay: cobertura de 80% em 15 dias =", R["cobertura80"], "| dias:", len(datas))

# ───────── grava o arquivo lido pelo app ─────────
(APP / "dados.json").write_text(json.dumps(D, ensure_ascii=False), encoding="utf-8")
print("ok:", APP / "dados.json")

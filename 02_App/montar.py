# -*- coding: utf-8 -*-
"""Monta o HTML do app a partir do modelo (template), dos dados e das imagens."""
import base64
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]


def _uri(arquivo, mime):
    return f"data:{mime};base64," + base64.b64encode((AQUI / "assets" / arquivo).read_bytes()).decode()


def montar_html():
    dados = json.loads((AQUI / "dados" / "app_completo.json").read_text(encoding="utf-8"))
    html = (AQUI / "template" / "app.html").read_text(encoding="utf-8")
    ano, mes, dia = map(int, dados["data_hoje"].split("-"))
    trocas = {
        "%%POSTER%%": _uri("poster.jpg", "image/jpeg"),
        "%%HERO%%": _uri("hero.jpg", "image/jpeg"),
        "%%LOGO%%": _uri("logo.png", "image/png"),
        "%%DADOS%%": json.dumps(dados, ensure_ascii=False),
        "%%DATA_EXT%%": f"{dia} de {MESES[mes - 1]} de {ano}",
    }
    for chave, valor in trocas.items():
        html = html.replace(chave, valor)
    assert "%%" not in html, "marcador sem substituição no template"
    return html

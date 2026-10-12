# -*- coding: utf-8 -*-
"""FIAP | Radar de Volatilidade do Brent (Tech Challenge, Fase 4).

Executar:  streamlit run 02_App/app.py
(rodar este arquivo com `python` também funciona: ele chama o Streamlit)
"""
import base64
import json
import subprocess
import sys
from pathlib import Path


def _no_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _no_streamlit():
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]))

import streamlit as st

AQUI = Path(__file__).resolve().parent
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]

st.set_page_config(page_title="FIAP | Radar de Volatilidade do Brent", page_icon="🛢️", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
      .stApp { background: #07111F; }
      .block-container { padding: 0 !important; max-width: 100% !important; }
      iframe { width: 100% !important; height: 100vh !important; border: 0; display: block; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def pagina():
    """Junta a interface (interface.html), os números (dados.json) e as imagens em uma única página."""
    dados = json.loads((AQUI / "dados.json").read_text(encoding="utf-8"))
    html = (AQUI / "interface.html").read_text(encoding="utf-8")
    ano, mes, dia = map(int, dados["data_hoje"].split("-"))
    img = lambda nome, mime: f"data:{mime};base64," + base64.b64encode((AQUI / "imagens" / nome).read_bytes()).decode()
    trocas = {"%%POSTER%%": img("poster.jpg", "image/jpeg"), "%%HERO%%": img("hero.jpg", "image/jpeg"), "%%LOGO%%": img("logo.png", "image/png"),
              "%%DADOS%%": json.dumps(dados, ensure_ascii=False), "%%DATA_EXT%%": f"{dia} de {MESES[mes - 1]} de {ano}"}
    for chave, valor in trocas.items():
        html = html.replace(chave, valor)
    return html


if hasattr(st, "iframe"):                       # Streamlit recente (st.components.v1.html foi descontinuado)
    st.iframe(pagina(), height=900)
else:                                           # versões antigas
    import streamlit.components.v1 as components
    components.html(pagina(), height=900, scrolling=True)

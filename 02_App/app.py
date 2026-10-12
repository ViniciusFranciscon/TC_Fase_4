# -*- coding: utf-8 -*-
"""FIAP | Radar de Volatilidade do Brent: app do Tech Challenge (Fase 4).

Executar: streamlit run 02_App/app.py   (rodar este arquivo com `python` também funciona: ele chama o Streamlit)
"""
import subprocess
import sys


def _no_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _no_streamlit():
    # executado como script comum (por exemplo, pelo botão "Run Python File" do VS Code): abre o app no Streamlit
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]))

import streamlit as st

from montar import montar_html

st.set_page_config(page_title="FIAP | Radar de Volatilidade do Brent", page_icon="🛢️", layout="wide", initial_sidebar_state="collapsed")

# a interface inteira (capa, páginas, gráficos e simulador) é desenhada pelo template em 02_App/template/app.html;
# o Streamlit entrega a página em tela cheia, sem os elementos padrão do framework
st.markdown(
    """
    <style>
      header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
      .stApp { background: #07111F; }
      .block-container { padding: 0 !important; max-width: 100% !important; }
      [data-testid="stAppViewContainer"] > .main { padding: 0; }
      iframe { width: 100% !important; height: 100vh !important; border: 0; display: block; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def pagina():
    return montar_html()


if hasattr(st, "iframe"):                       # Streamlit recente: st.components.v1.html foi descontinuado
    st.iframe(pagina(), height=900)
else:                                           # versões antigas
    import streamlit.components.v1 as components
    components.html(pagina(), height=900, scrolling=True)

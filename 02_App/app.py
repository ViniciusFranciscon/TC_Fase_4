# -*- coding: utf-8 -*-
"""FIAP | Radar de Volatilidade do Brent: app do Tech Challenge (Fase 4).

Executar: streamlit run 02_App/app.py
"""
import streamlit as st
import streamlit.components.v1 as components

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


components.html(pagina(), height=900, scrolling=True)

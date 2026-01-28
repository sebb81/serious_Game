"""Point d'entrée de l'application Streamlit.
Configure l'application, initialise l'état, gère la navigation linéaire entre pages,
et affiche le bandeau des badges obtenus.
"""
from pathlib import Path

import streamlit as st

from utils import badges
import missions.introduction as introduction
import missions.mission1 as mission1
import missions.mission2 as mission2
import missions.mission3 as mission3
import missions.mission4 as mission4
import missions.conclusion as conclusion

# Configuration de la page (titre de l'onglet, icône, etc.)
st.set_page_config(page_title="Serious Game - IA Frugale", page_icon="🤖", layout="wide")

# Chargement du style externe
STYLE_PATH = Path(__file__).resolve().parent / "styles" / "theme.css"
if STYLE_PATH.exists():
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# Initialisation de l'état (page courante et badges) lors du premier chargement
if "page" not in st.session_state:
    st.session_state.page = "introduction"
if "badges" not in st.session_state:
    # Dictionnaire des badges débloqués (False par défaut au départ)
    st.session_state.badges = {key: False for key in badges.BADGES.keys()}

# Affichage du bandeau des badges en haut de l'application
badges.display_badges()

# Navigation conditionnelle en fonction de la page active
if st.session_state.page == "introduction":
    introduction.run()
elif st.session_state.page == "mission1":
    mission1.run()
elif st.session_state.page == "mission2":
    mission2.run()
elif st.session_state.page == "mission3":
    mission3.run()
elif st.session_state.page == "mission4":
    mission4.run()
elif st.session_state.page == "conclusion":
    conclusion.run()

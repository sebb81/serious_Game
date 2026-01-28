"""Utilitaires pour la gestion des badges (déblocage et navigation)."""
import streamlit as st

# Dictionnaire des badges disponibles (un par mission) avec emojis et titres
BADGES = {
    "mission1": {"title": "Geste", "emoji": "👍"},
    "mission2": {"title": "Émotion", "emoji": "😊"},
    "mission3": {"title": "Chatbot", "emoji": "💬"},
    "mission4": {"title": "RAG", "emoji": "📚"},
}


def unlock_badge(mission_key: str):
    """Marque le badge de la mission donnée comme débloqué dans l'état de session."""
    st.session_state.badges[mission_key] = True


def display_badges():
    """Affiche un bandeau horizontal cliquable avec les badges obtenus."""
    cols = st.columns(len(BADGES))
    for (key, info), col in zip(BADGES.items(), cols):
        unlocked = st.session_state.badges.get(key, False)
        label = f"{info['emoji']} {info['title']}" if unlocked else f"🔒 {info['title']}"
        if col.button(label, key=f"nav-{key}", disabled=not unlocked):
            st.session_state.page = key

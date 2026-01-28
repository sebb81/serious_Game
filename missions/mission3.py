"""Mission 3 - Chatbot IA local (LLM compact via llama.cpp)."""
import streamlit as st
from openai import OpenAI
from utils import badges


# -----------------------------------------------------------------------------
# Configuration LLM (llama.cpp server en mode OpenAI-compatible)
# -----------------------------------------------------------------------------
BASE_URL = "http://localhost:8033/v1"
LLM_MODEL = "mistral"

SYSTEM_PROMPT = (
    "Tu es un assistant IA local. "
    "Réponds en français, de manière claire et structurée. "
    "Si l'utilisateur demande du code, donne un exemple minimal et correct."
)


@st.cache_resource
def get_llm_client() -> OpenAI:
    # llama.cpp (server) accepte un api_key factice en mode local
    return OpenAI(base_url=BASE_URL, api_key="sk-no-key-needed")


def _init_state():
    if "m3_messages" not in st.session_state:
        st.session_state.m3_messages = []
    if "m3_system_prompt" not in st.session_state:
        st.session_state.m3_system_prompt = SYSTEM_PROMPT
    if "m3_badge_unlocked" not in st.session_state:
        st.session_state.m3_badge_unlocked = False


def _render_history():
    for msg in st.session_state.m3_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _unlock_badge_once():
    if not st.session_state.m3_badge_unlocked:
        badges.unlock_badge("mission3")
        st.session_state.m3_badge_unlocked = True
        st.success("🎉 Félicitations, vous avez débloqué le badge **Chatbot** 💬")


def run():
    _init_state()

    st.title("Mission 3 : Chatbot IA local 💬")
    st.write(
        "**Objectif** : Interagir avec un **assistant IA** fonctionnant entièrement en local "
        "(via un serveur llama.cpp compatible OpenAI)."
    )

    # Zone de configuration (sans sidebar)
    with st.expander("⚙️ Prompt système (optionnel)", expanded=False):
        st.text_area(
            "Le prompt système guide le comportement de l'assistant.",
            key="m3_system_prompt",
            height=160,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Réinitialiser le prompt système"):
                st.session_state.m3_system_prompt = SYSTEM_PROMPT
                st.rerun()
        with c2:
            if st.button("🧹 Effacer la conversation"):
                st.session_state.m3_messages = []
                st.rerun()

    # Affichage historique
    _render_history()

    # Le chat_input DOIT être le dernier élément
    prompt = st.chat_input("Posez votre question...")
    if prompt:
        st.session_state.m3_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        client = get_llm_client()

        # Construction du contexte: system + historique
        messages_for_llm = [{"role": "system", "content": st.session_state.m3_system_prompt}]
        messages_for_llm.extend(st.session_state.m3_messages)

        # Appel LLM en streaming (comme app_llamacpp_v3.py)
        with st.chat_message("assistant"):
            try:
                stream = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages_for_llm,
                    stream=True,
                    temperature=0.3,
                    top_p=0.9,
                    presence_penalty=0.6,
                    frequency_penalty=1.5,
                    max_tokens=2048,
                )
                response = st.write_stream(stream)
            except Exception as exc:
                response = (
                    "❌ Impossible de contacter le serveur llama.cpp.\n\n"
                    f"**Détail** : {exc}\n\n"
                    "Vérifiez que le serveur est lancé et accessible sur : "
                    f"`{BASE_URL}` (endpoint `/chat/completions`)."
                )
                st.error(response)

        st.session_state.m3_messages.append({"role": "assistant", "content": response})

        # Badge: considéré “accompli” après au moins une interaction
        _unlock_badge_once()

    # Navigation
    st.divider()
    if st.button("Accueil ➡️"):
        st.session_state.page = "introduction"

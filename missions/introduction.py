"""Page d'introduction du jeu - présente le contexte et lance les missions."""
import streamlit as st


def run():
    st.title("🤖 Bienvenue dans le jeu IA Frugale !")
    st.write(
        "**Découvrez l’IA frugale et locale à travers 5 missions interactives.** "
        "L’atelier est semi-dirigé : l’objectif est d’expérimenter, d’ajuster les paramètres "
        "et d’observer les compromis (précision vs rappel, latence vs qualité, etc.)."
    )
    st.info(
        "Parcourez les missions dans l’ordre que vous voulez. Pour chaque défi, lisez l’objectif "
        "puis testez différentes approches afin d’atteindre le meilleur équilibre."
    )

    missions = [
        {
            "key": "mission1",
            "title": "Mission 1 — Geste",
            "desc": "Détection de pouce levé et réglage du seuil de confiance.",
            "icon": "👍",
            "page": "mission1",
        },
        {
            "key": "mission2",
            "title": "Mission 2 — Émotion",
            "desc": "Face Mesh en temps réel et estimation d’émotion.",
            "icon": "😊",
            "page": "mission2",
        },
        {
            "key": "mission3",
            "title": "Mission 3 — Chatbot",
            "desc": "Interagir avec un assistant IA local (LLM compact).",
            "icon": "💬",
            "page": "mission3",
        },
        {
            "key": "mission4",
            "title": "Mission 4 — Documents",
            "desc": "RAG local : réponses basées sur vos documents.",
            "icon": "📚",
            "page": "mission4",
        },
        {
            "key": "mission5",
            "title": "Mission 5 — Audio",
            "desc": "Reconnaissance vocale / micro.",
            "icon": "🎤",
            "page": "mission5"
        },
    ]

    completed = sum(1 for m in missions if st.session_state.badges.get(m["key"], False))
    total = len(missions)
    progress = completed / total if total else 0

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"**Missions complétées : {completed}/{total}**")
        st.progress(progress)
    with right:
        st.metric("Score", f"{completed}/{total}")

    st.markdown("---")

    cols = st.columns(3)
    for idx, mission in enumerate(missions):
        col = cols[idx % 3]
        done = st.session_state.badges.get(mission["key"], False)
        coming_soon = mission.get("coming_soon", False)
        status_class = "mission-dot-done" if done else ("mission-dot-locked" if coming_soon else "mission-dot-pending")
        card_class = "mission-card done" if done else "mission-card locked"

        with col:
            col.markdown(
                f"""
                <div class="{card_class}">
                    <div class="mission-card-header">
                      <div class="mission-header-row">
                        <div class="mission-icon">{mission['icon']}</div>
                        <div class="mission-title">{mission['title']}</div>
                      </div>
                    </div>
                    <div class="mission-card-body">
                      <div class="mission-desc">{mission['desc']}</div>
                      <div class="mission-status">
                        <span class="mission-dot {status_class}"></span>
                      </div>
                    </div>
                    """,
                unsafe_allow_html=True,
            )
            if coming_soon:
                col.button("🔒 Bientôt", key=f"{mission['key']}-soon", disabled=True, use_container_width=True)
            else:
                if col.button("🚀 Commencer", key=f"{mission['key']}-open", use_container_width=True):
                    st.session_state.page = mission["page"]
                    st.rerun()
            col.markdown("</div>", unsafe_allow_html=True)

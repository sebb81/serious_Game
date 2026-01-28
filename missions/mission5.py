"""Mission 5 - Reconnaissance vocale locale (Whisper Tiny) + jeu de concision.

Objectif:
- Enregistrer une courte phrase via micro (st.audio_input)
- Transcrire en local avec Whisper Tiny
- Répéter 2-3 fois en raccourcissant la requête, comparer les transcriptions
- Débloquer le badge une fois l’exercice réalisé
"""
from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import streamlit as st
from utils import badges


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WHISPER_VARIANT = "tiny"  # openai-whisper: "tiny" / HF: "openai/whisper-tiny"
DEFAULT_LANGUAGE = "fr"

TARGET_SAMPLE_RATE = 16000  # conseillé pour Whisper


# -----------------------------------------------------------------------------
# Backend Whisper (local) : essaie d'abord openai-whisper, sinon transformers
# -----------------------------------------------------------------------------
@st.cache_resource
def get_asr_backend():
    """
    Retourne un tuple (backend_name, backend_object)

    - backend_name = "whisper" si lib openai-whisper dispo
    - backend_name = "transformers" si pipeline HF dispo
    """
    # 1) openai-whisper (plus simple)
    try:
        import whisper  # type: ignore

        model = whisper.load_model(WHISPER_VARIANT)
        return "whisper", model
    except Exception:
        pass

    # 2) transformers pipeline (fallback)
    try:
        from transformers import pipeline  # type: ignore

        asr = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{WHISPER_VARIANT}",
            device="cpu",
        )
        return "transformers", asr
    except Exception as e:
        return "none", str(e)


def transcribe_audio_bytes(audio_bytes: bytes, language: str = DEFAULT_LANGUAGE):
    """
    Transcrit un audio à partir de bytes.
    Retourne (texte, erreur).
    Compatible Windows: évite NamedTemporaryFile(delete=True) (fichier verrouillé).
    """
    import os
    import tempfile

    backend_name, backend = get_asr_backend()
    if backend_name == "none":
        return None, (
            "Aucun backend Whisper local disponible.\n\n"
            "Installez l’un des deux:\n"
            "- `pip install -U openai-whisper`\n"
            "ou\n"
            "- `pip install -U transformers accelerate` (et torch)\n\n"
            f"Détail: {backend}"
        )

    tmp_path = None
    try:
        # Crée un fichier temporaire, ferme immédiatement le handle
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        if backend_name == "whisper":
            result = backend.transcribe(tmp_path, language=language, fp16=False)
            text = (result.get("text") or "").strip()
            return text, None

        out = backend(tmp_path, generate_kwargs={"language": language})
        text = (out.get("text") if isinstance(out, dict) else str(out)).strip()
        return text, None

    except Exception as e:
        return None, f"Erreur de transcription: {e}"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Jeu de concision
# -----------------------------------------------------------------------------
@dataclass
class Attempt:
    label: str
    transcript: str

    @property
    def words(self) -> int:
        toks = re.findall(r"\b\w+\b", self.transcript, flags=re.UNICODE)
        return len(toks)

    @property
    def chars(self) -> int:
        return len(self.transcript.strip())


def concision_stars(word_count: int) -> str:
    # purely fun: plus c’est court, plus il y a d’étoiles
    if word_count <= 4:
        return "⭐⭐⭐⭐⭐"
    if word_count <= 6:
        return "⭐⭐⭐⭐"
    if word_count <= 8:
        return "⭐⭐⭐"
    if word_count <= 12:
        return "⭐⭐"
    return "⭐"


def keyword_preservation(a: str, b: str) -> float:
    """
    Mesure simple: proportion de mots "importants" de A retrouvés dans B.
    Important = mots >=4 lettres, hors stopwords FR minimaux.
    """
    stop = {
        "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
        "le", "la", "les", "un", "une", "des", "du", "de", "d",
        "et", "ou", "mais", "donc", "or", "ni", "car",
        "pour", "par", "avec", "sans", "dans", "sur", "sous", "chez",
        "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
        "son", "sa", "ses", "notre", "vos", "leur", "leurs",
        "est", "suis", "es", "sommes", "êtes", "sont",
        "a", "as", "avons", "avez", "ont",
        "que", "qui", "quoi", "dont", "où",
        "au", "aux",
    }

    def important_words(t: str) -> List[str]:
        toks = [w.lower() for w in re.findall(r"\b\w+\b", t, flags=re.UNICODE)]
        return [w for w in toks if len(w) >= 4 and w not in stop]

    base = set(important_words(a))
    if not base:
        return 1.0

    cand = set(important_words(b))
    kept = len(base.intersection(cand))
    return kept / max(1, len(base))


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
def _init_state():
    if "m5_attempts" not in st.session_state:
        st.session_state.m5_attempts: List[Attempt] = []
    if "m5_badge_unlocked" not in st.session_state:
        st.session_state.m5_badge_unlocked = False


def _unlock_badge_once():
    if not st.session_state.m5_badge_unlocked:
        badges.unlock_badge("mission5")
        st.session_state.m5_badge_unlocked = True
        st.success("🎉 Mission réussie : vous avez débloqué le badge **Audio** 🎙️")


def _add_attempt(label: str, transcript: str):
    st.session_state.m5_attempts.append(Attempt(label=label, transcript=transcript))


def run():
    _init_state()

    st.title("Mission 5 : Reconnaissance vocale locale 🎙️")
    st.write(
        "**Objectif** : dicter une requête, la faire transcrire en local avec **Whisper Tiny**, "
        "puis recommencer en **raccourcissant** la phrase tout en gardant le sens."
    )

    backend_name, _ = get_asr_backend()
    if backend_name == "none":
        st.warning(
            "Whisper local n’est pas disponible sur cet environnement. "
            "Voir les instructions dans le message d’erreur ci-dessous."
        )
        _, err = transcribe_audio_bytes(b"")
        if err:
            st.code(err)
        st.divider()
        if st.button("Accueil ➡️"):
            st.session_state.page = "introduction"
            st.rerun()
        return

    with st.expander("⚙️ Options (optionnel)", expanded=False):
        language = st.selectbox("Langue", ["fr", "en", "es", "de", "it"], index=0)
        st.caption("Le modèle Tiny est rapide mais moins précis que les modèles plus grands.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧹 Effacer les essais"):
                st.session_state.m5_attempts = []
                st.rerun()
        with c2:
            if st.button("Accueil ➡️"):
                st.session_state.page = "introduction"
                st.rerun()

    st.subheader("Consigne")
    st.markdown(
        "- **Étape 1** : énoncez une requête complète (phrase naturelle).\n"
        "- **Étape 2** : reformulez la même intention en moins de mots.\n"
        "- **Étape 3** : tentez avec un minimum de mots-clés.\n\n"
        "À chaque fois, comparez la transcription et observez jusqu’où la concision reste compréhensible."
    )

    st.subheader("Enregistrement")
    st.caption("Cliquez sur le micro, parlez quelques secondes, puis stoppez l’enregistrement.")

    # On guide le label automatiquement selon le nombre d'essais
    next_label = {0: "Phrase 1 (complète)", 1: "Phrase 2 (plus courte)", 2: "Phrase 3 (mots-clés)"}.get(
        len(st.session_state.m5_attempts),
        f"Phrase {len(st.session_state.m5_attempts) + 1}",
    )

    # IMPORTANT: st.audio_input reste ancré dans le flux et renvoie un fichier
    audio = st.audio_input(
        f"{next_label} — Enregistrer",
        # selon versions Streamlit, le paramètre sample_rate peut exister ou non
        # si ça plante chez toi, supprime sample_rate=...
        sample_rate=TARGET_SAMPLE_RATE,
    )

    if audio is not None:
        audio_bytes = audio.getvalue()
        with st.spinner("Transcription en cours (Whisper Tiny, local)…"):
            text, err = transcribe_audio_bytes(audio_bytes, language=language)

        if err:
            st.error(err)
        else:
            text = text or ""
            _add_attempt(next_label, text)

            st.success("📜 Transcription obtenue")
            st.write(text if text else "_(vide)_")

            # On débloque la mission dès 3 essais (ou dès 2 si tu préfères)
            if len(st.session_state.m5_attempts) >= 3:
                _unlock_badge_once()

            st.rerun()

    st.subheader("Comparaison")
    if not st.session_state.m5_attempts:
        st.info("Aucun essai pour le moment. Enregistrez une première phrase.")
        return

    # Affichage des essais (liste)
    for att in st.session_state.m5_attempts:
        st.markdown(f"**{att.label}**  \n"
                    f"- {att.words} mots • {att.chars} caractères • {concision_stars(att.words)}")
        st.write(att.transcript)

    # Feedback simple après au moins 2 essais
    if len(st.session_state.m5_attempts) >= 2:
        a0 = st.session_state.m5_attempts[0].transcript
        last = st.session_state.m5_attempts[-1].transcript
        score = keyword_preservation(a0, last)

        st.divider()
        st.subheader("Feedback")
        st.caption("Heuristique simple : retrouve-t-on les mots importants de la phrase initiale ?")

        st.metric("Conservation des mots-clés (approx.)", f"{int(round(score * 100, 0))}%")

        if score >= 0.70:
            st.success("👍 Pari réussi : même plus concise, la requête semble garder le même sens.")
        elif score >= 0.40:
            st.warning("🤔 Zone grise : la concision commence à faire perdre des éléments importants.")
        else:
            st.error("⚠️ Limite atteinte : en voulant trop condenser, la requête perd probablement son sens.")

    st.divider()
    if st.button("Accueil ➡️"):
        st.session_state.page = "introduction"
        st.rerun()

"""Mission 4 - Chatbot augmenté (Pipeline RAG local)."""
import math
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from utils import badges


# -----------------------------------------------------------------------------
# Configuration (llama.cpp server en mode OpenAI-compatible)
# -----------------------------------------------------------------------------
BASE_URL = "http://localhost:8033/v1"
LLM_MODEL = "mistral"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 6
DEFAULT_MIN_SCORE = 0.25

SYSTEM_PROMPT = (
    "Tu es un assistant IA local.\n"
    "Tu dois répondre en français.\n"
    "Si un CONTEXTE DOCUMENTAIRE est fourni, utilise-le en priorité et cite tes sources "
    "avec les numéros entre crochets (ex: [1], [2]).\n"
    "Si le contexte ne contient pas l'information, dis-le clairement et propose quoi chercher."
)


# -----------------------------------------------------------------------------
# Ressources & état
# -----------------------------------------------------------------------------
@st.cache_resource
def get_llm_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key="sk-no-key-needed")


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _init_state():
    if "m4_messages" not in st.session_state:
        st.session_state.m4_messages = []
    if "m4_system_prompt" not in st.session_state:
        st.session_state.m4_system_prompt = SYSTEM_PROMPT
    if "m4_badge_unlocked" not in st.session_state:
        st.session_state.m4_badge_unlocked = False

    # RAG store (en mémoire)
    if "rag_docs" not in st.session_state:
        st.session_state.rag_docs = []       # chunks (str)
        st.session_state.rag_sources = []    # source filename (str)
        st.session_state.rag_embeds = []     # list[list[float]] (normalisés)
        st.session_state.rag_norms = []      # float (norme)
        st.session_state.rag_hashes = set()  # anti-doublons (hash chunk)


def reset_rag_store():
    st.session_state.rag_docs = []
    st.session_state.rag_sources = []
    st.session_state.rag_embeds = []
    st.session_state.rag_norms = []
    st.session_state.rag_hashes = set()


def _unlock_badge_once():
    if not st.session_state.m4_badge_unlocked:
        badges.unlock_badge("mission4")
        st.session_state.m4_badge_unlocked = True
        st.success("🎉 Bravo ! Vous avez débloqué le badge **RAG** 📚")


# -----------------------------------------------------------------------------
# RAG utils (transposés depuis app_llamacpp_v3.py)
# -----------------------------------------------------------------------------
def text_from_bytes(name: str, data: bytes):
    name_lower = name.lower()

    # PDF via PyMuPDF (fitz)
    if name_lower.endswith(".pdf"):
        try:
            import fitz  # pymupdf
        except ImportError:
            return "", "Erreur : installez PyMuPDF via `pip install pymupdf`"

        try:
            doc = fitz.open(stream=data, filetype="pdf")
            pages = []
            for page in doc:
                pages.append(page.get_text())
            return "\n".join(pages), None
        except Exception as e:
            return "", f"Erreur de lecture PDF (PyMuPDF) : {e}"

    # Fichiers texte
    return data.decode("utf-8", errors="ignore"), None


def chunk_text(text: str, max_chars: int, overlap: int):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""

    effective_overlap = min(overlap, max_chars - 1) if max_chars > 1 else 0
    step = max(1, max_chars - effective_overlap)

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}" if current else paragraph
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            for i in range(0, len(paragraph), step):
                chunks.append(paragraph[i : i + max_chars])
            current = ""

    if current:
        chunks.append(current)

    return chunks


def dot_product(left, right):
    return sum(l * r for l, r in zip(left, right))


def vector_norm(v):
    return math.sqrt(sum(x * x for x in v))


def embed_texts(texts):
    model = get_embedding_model()
    embeds = model.encode(texts, normalize_embeddings=True)
    return embeds.tolist()


def retrieve_chunks(query: str, top_k: int, min_score: float):
    if not st.session_state.rag_docs:
        return []

    query_emb = embed_texts([query])[0]
    qn = vector_norm(query_emb)
    if qn == 0:
        return []

    # Bonus mots-clés frugal (hybrid search)
    query_keywords = [w.lower() for w in query.split() if len(w) > 3]

    scored = []
    for idx, emb in enumerate(st.session_state.rag_embeds):
        denom = qn * st.session_state.rag_norms[idx]
        cosine = dot_product(query_emb, emb) / denom if denom else 0.0

        chunk_lower = st.session_state.rag_docs[idx].lower()
        bonus = 0.0
        for kw in query_keywords:
            if kw in chunk_lower:
                bonus += 0.05
        bonus = min(bonus, 0.30)

        score = cosine + bonus
        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, idx in scored[:top_k]:
        if score < min_score:
            continue
        results.append(
            {
                "score": score,
                "text": st.session_state.rag_docs[idx],
                "source": st.session_state.rag_sources[idx],
            }
        )
    return results


def build_context_block(results):
    if not results:
        return ""

    lines = ["### CONTEXTE DOCUMENTAIRE"]
    for i, r in enumerate(results, start=1):
        src = r["source"]
        chunk = r["text"].strip()
        lines.append(f"[{i}] Source: {src}\n{chunk}\n")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def _render_history():
    for msg in st.session_state.m4_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def run():
    _init_state()

    st.title("Mission 4 : Chatbot augmenté par des documents 📚")
    st.write(
        "**Objectif** : Permettre à l'IA de répondre en s'appuyant sur une base de connaissances locale "
        "(pipeline **RAG** : indexation → recherche → génération)."
    )

    with st.expander("⚙️ Paramètres (optionnel)", expanded=False):
        st.text_area("Prompt système", key="m4_system_prompt", height=160)

        c1, c2, c3 = st.columns(3)
        with c1:
            chunk_size = st.number_input(
                "Taille des chunks (caractères)",
                min_value=400,
                max_value=4000,
                value=DEFAULT_CHUNK_SIZE,
                step=100,
            )
        with c2:
            overlap = st.number_input(
                "Chevauchement",
                min_value=0,
                max_value=1000,
                value=DEFAULT_CHUNK_OVERLAP,
                step=50,
            )
        with c3:
            top_k = st.number_input(
                "Top-K (passages)",
                min_value=1,
                max_value=20,
                value=DEFAULT_TOP_K,
                step=1,
            )

        min_score = st.slider("Score minimum", 0.0, 1.0, float(DEFAULT_MIN_SCORE), 0.01)

        a, b = st.columns(2)
        with a:
            if st.button("🧹 Effacer la conversation"):
                st.session_state.m4_messages = []
                st.rerun()
        with b:
            if st.button("♻️ Réinitialiser la base (RAG)"):
                reset_rag_store()
                st.toast("Base RAG réinitialisée.")
                st.rerun()

    st.subheader("1) Ajouter des documents")
    uploads = st.file_uploader(
        "Ajoutez un ou plusieurs fichiers (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploads:
        added_chunks = 0
        for up in uploads:
            data = up.getvalue()
            text, err = text_from_bytes(up.name, data)
            if err:
                st.warning(f"{up.name} — {err}")
                continue

            chunks = chunk_text(text, int(chunk_size), int(overlap))
            if not chunks:
                continue

            # Anti-doublons par chunk (hash)
            new_chunks = []
            new_sources = []
            for c in chunks:
                h = hash(c)
                if h in st.session_state.rag_hashes:
                    continue
                st.session_state.rag_hashes.add(h)
                new_chunks.append(c)
                new_sources.append(up.name)

            if not new_chunks:
                continue

            embeds = embed_texts(new_chunks)
            norms = [vector_norm(e) for e in embeds]

            st.session_state.rag_docs.extend(new_chunks)
            st.session_state.rag_sources.extend(new_sources)
            st.session_state.rag_embeds.extend(embeds)
            st.session_state.rag_norms.extend(norms)

            added_chunks += len(new_chunks)

        if added_chunks:
            st.success(f"✅ Index mis à jour : **{added_chunks}** chunks ajoutés.")
        st.caption(
            f"Chunks en base : {len(st.session_state.rag_docs)} (sources: {len(set(st.session_state.rag_sources))})"
        )

    st.subheader("2) Discuter avec l'assistant (avec RAG)")
    _render_history()

    # Input dans le flux (pas st.chat_input)
    st.divider()
    with st.form("m4_prompt_form", clear_on_submit=True):
        query = st.text_input("Votre question", placeholder="Posez une question…")
        send = st.form_submit_button("Envoyer")

    if send and query.strip():
        user_q = query.strip()
        st.session_state.m4_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # RAG: retrieve + context
        results = retrieve_chunks(user_q, int(top_k), float(min_score))
        context_block = build_context_block(results)

        # Messages pour le LLM
        messages_for_llm = [{"role": "system", "content": st.session_state.m4_system_prompt}]
        if context_block:
            messages_for_llm.append({"role": "system", "content": context_block})
        messages_for_llm.extend(st.session_state.m4_messages)

        client = get_llm_client()

        with st.chat_message("assistant"):
            try:
                stream = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages_for_llm,
                    stream=True,
                    temperature=0.3,
                    top_p=0.9,
                    presence_penalty=0.4,
                    frequency_penalty=0.8,
                    max_tokens=2048,
                )
                answer = st.write_stream(stream)
            except Exception as exc:
                answer = (
                    "❌ Impossible de contacter le serveur llama.cpp.\n\n"
                    f"**Détail** : {exc}\n\n"
                    f"Vérifiez : `{BASE_URL}` (endpoint `/chat/completions`)."
                )
                st.error(answer)

        st.session_state.m4_messages.append({"role": "assistant", "content": answer})

        # Afficher les sources retrouvées (lisible, optionnel)
        if results:
            with st.expander("📌 Passages utilisés (RAG)", expanded=False):
                for i, r in enumerate(results, start=1):
                    st.markdown(f"**[{i}] {r['source']}** — score: `{r['score']:.3f}`")
                    st.write(r["text"])

        # Badge: on considère réussi dès qu'on obtient une réponse (avec ou sans contexte)
        _unlock_badge_once()

    st.divider()
    if st.button("Accueil ➡️"):
        st.session_state.page = "introduction"

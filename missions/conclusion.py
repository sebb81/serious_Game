"""Page de conclusion - fin du jeu et récapitulatif."""
import streamlit as st
from utils import badges

def run():
    st.title("🏁 Mission accomplie !")
    st.subheader("Félicitations, vous avez relevé tous les défis de l'IA frugale.")
    st.write(
        "Vous avez obtenu l'ensemble des badges du jeu : " +
        ", ".join([f"{info['emoji']} **{info['title']}**" for key, info in badges.BADGES.items()])
    )
    st.write("Merci d'avoir participé à cette aventure pédagogique sur l’**IA frugale**. 🎓")
    st.write("*N'hésitez pas à partager vos retours ou à rejouer pour consolider vos connaissances.*")

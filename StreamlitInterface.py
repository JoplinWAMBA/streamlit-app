import streamlit as st
import requests
import plotly.express as px
import os

# =====================================================
# Configuration de l’application
# =====================================================
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

# =====================================================
# Choix dynamique de l'URL API (local vs production)
# =====================================================
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Vérification (affichée uniquement en mode debug)
st.sidebar.markdown(f"🔗 **API utilisée :** `{API_URL}`")

# Si tu veux, tu peux aussi afficher une alerte si API_URL pointe vers localhost sur Streamlit Cloud :
if "localhost" in API_URL:
    st.warning("⚠️ L'application est configurée pour le mode local.")



# =====================================================
# Sidebar (Informations et navigation)
# =====================================================
with st.sidebar:
    st.header("ℹ️ Informations")
    st.markdown("""
    Cette application utilise :
    - 🤖 *Machine Learning* pour la prédiction  
    - 🧩 *LIME* pour l’explicabilité  
    - ⚡ *FastAPI* pour les performances  
    - 🎨 *Streamlit* pour l’interface
    """)

    st.divider()

    st.subheader("🧠 Modèle")
    st.info("Type : **Logistic Regression**\n\nVocabulaire : 10 000 mots")

    st.divider()

    st.subheader("💡 Comment ça marche ?")
    st.markdown("""
    1️⃣ Saisissez votre tweet  
    2️⃣ Cliquez sur *Prédire le sentiment*  
    3️⃣ Explorez l’explication LIME  
    4️⃣ Comprenez la décision du modèle
    """)

    st.divider()
    st.subheader("📚 Exemples à tester")
    examples = [
        "J'adore ce produit, il est incroyable !",
        "Ce film est une perte de temps totale...",
        "Service client très professionnel 👍",
        "Je ne recommanderai jamais cet endroit 😡"
    ]
    example = st.selectbox("Choisir un exemple :", [""] + examples)
    if example:
        st.session_state["tweet_text"] = example


# =====================================================
# Corps principal
# =====================================================
st.title("💬 Twitter Sentiment Analyzer")
st.caption("Analyse de sentiment avec Intelligence Artificielle et explicabilité LIME")

# Test API connection
try:
    health = requests.get(f"{API_URL}/health")
    if health.status_code == 200:
        st.success("✅ API connectée avec succès")
    else:
        st.error("⚠️ API injoignable")
except Exception:
    st.error("❌ Impossible de se connecter à l’API. Vérifiez qu’elle est lancée.")


# =====================================================
# 1️⃣ Zone de saisie
# =====================================================
tweet_text = st.text_area(
    "📝 Saisissez votre tweet à analyser",
    value=st.session_state.get("tweet_text", ""),
    max_chars=280,
    placeholder="Tapez votre tweet ici... (280 caractères max)",
    help="Exemple : 'J'adore ce produit, il est fantastique !'"
)

# Compteur dynamique de caractères
char_count = len(tweet_text)
if char_count < 240:
    st.markdown(f"<span style='color:green'>{char_count}/280</span>", unsafe_allow_html=True)
elif char_count <= 280:
    st.markdown(f"<span style='color:orange'>{char_count}/280</span>", unsafe_allow_html=True)
else:
    st.markdown(f"<span style='color:red'>{char_count}/280</span>", unsafe_allow_html=True)

text_valid = 0 < char_count <= 280

# =====================================================
# 2️⃣ Boutons d’action
# =====================================================
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    predict_btn = st.button(
        "🎯 Prédire le sentiment",
        disabled=not text_valid,
        type="primary",
        use_container_width=True
    )
with col2:
    explain_btn = st.button(
        "🔍 LIME (30-60s)",
        disabled=not text_valid,
        use_container_width=True
    )
with col3:
    clear_btn = st.button(
        "🗑️ Effacer",
        use_container_width=True
    )

if clear_btn:
    st.session_state["tweet_text"] = ""
    st.rerun()

# =====================================================
# 3️⃣ Affichage des résultats de prédiction
# =====================================================
if predict_btn and text_valid:
    with st.spinner("Analyse du tweet en cours..."):
        try:
            response = requests.post(f"{API_URL}/predict", json={"text": tweet_text})
            if response.status_code == 200:
                data = response.json()
                sentiment = data["sentiment"]
                confidence = data["confidence"]
                prob_pos = data["probability_positive"]
                prob_neg = data["probability_negative"]

                if sentiment.lower().startswith("pos"):
                    st.success(f"😊 **POSiTIF** ({confidence:.1%})")
                else:
                    st.error(f"😞 **NÉGATIF** ({confidence:.1%})")

                # Graphique interactif Plotly
                fig = px.bar(
                    x=["Négatif", "Positif"],
                    y=[prob_neg, prob_pos],
                    color=["Négatif", "Positif"],
                    color_discrete_map={"Négatif": "red", "Positif": "green"},
                    labels={"x": "Classe", "y": "Probabilité"}
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("Erreur lors de la requête à l’API.")
        except Exception as e:
            st.error(f"Erreur : {e}")


# =====================================================
# 4️⃣ Explicabilité LIME
# =====================================================
if explain_btn and text_valid:
    with st.spinner("Génération de l’explication LIME... (30–60s)"):
        try:
            resp = requests.post(f"{API_URL}/explain", json={"text": tweet_text})
            if resp.status_code == 200:
                explanation_data = resp.json()
                st.subheader("📘 Explication LIME")
                st.components.v1.html(
                    explanation_data["html_explanation"],
                    height=400,
                    scrolling=True
                )

                with st.expander("📊 Détails des poids des mots"):
                    for item in explanation_data["explanation"]:
                        color = "green" if item["weight"] > 0 else "red"
                        st.markdown(f"- <span style='color:{color}'>{item['word']}</span> → {item['weight']:.3f}",
                                    unsafe_allow_html=True)
            else:
                st.error("⚠️ Erreur lors de la génération de l’explication LIME.")
        except Exception as e:
            st.error(f"Erreur : {e}")

# =====================================================
# Footer
# =====================================================
st.divider()
st.caption("🧠 Application développée dans le cadre du projet MLOps — Streamlit + FastAPI + MLflow + LIME")

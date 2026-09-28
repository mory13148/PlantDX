import sys
import json
from pathlib import Path

import streamlit as st
from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# MODEL
# ============================================================

from inference import (
    predict_image,
    format_class_name,
)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="PlantDx",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOAD DISEASE DATABASE
# ============================================================

@st.cache_data
def load_disease_database():
    possible_paths = [
        ROOT_DIR / "data" / "diseases.json",
        ROOT_DIR / "diseases.json",
        ROOT_DIR / "src" / "diseases.json",
    ]

    for path in possible_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    return {}


# ============================================================
# HELPERS
# ============================================================

def clean_list_item(item):
    """
    Nettoie les éléments venant du JSON pour éviter
    l'affichage de '- item' ou '* item' dans les listes.
    """
    if isinstance(item, str):
        return item.lstrip("-* ").strip()

    return str(item)


def display_list(items):
    """
    Affiche proprement une liste.
    """
    if not items:
        st.write("Aucune information disponible.")

        return

    for item in items:
        st.markdown(f"- {clean_list_item(item)}")


# ============================================================
# DISEASE INFORMATION
# ============================================================

def display_disease_info(class_name, diseases_db):
    """
    Affiche les informations en français concernant
    la maladie détectée.
    """

    if not diseases_db:
        return

    disease = diseases_db.get(class_name)

    if disease is None:
        return

    st.divider()

    st.subheader("🩺 Informations")

    # Nom français
    french_name = disease.get(
        "french_name",
        class_name
    )

    st.markdown(f"### 🌿 {french_name}")

    # Plante
    plant = disease.get("plant")

    if plant:
        st.markdown("**🌱 Plante**")
        st.write(plant)

    # Symptômes
    symptoms = disease.get("symptoms")

    if symptoms:
        st.markdown("**🔎 Symptômes**")
        display_list(symptoms)

    # Traitement
    treatment = disease.get("treatment")

    if treatment:
        st.markdown("**💊 Traitement recommandé**")
        display_list(treatment)


# ============================================================
# DISPLAY PREDICTION
# ============================================================

def display_prediction(result, diseases_db):
    """
    Affiche le résultat final de la prédiction.
    """

    if result is None:
        return

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    if result.get("is_unknown", False):

        confidence = result.get("confidence", 0) * 100

        st.warning(
            "⚠️ Impossible d'identifier la maladie avec "
            "suffisamment de confiance."
        )

        st.metric(
            "Confiance",
            f"{confidence:.2f}%"
        )

        st.info(
            "Essayez avec une photo plus nette et centrée "
            "sur la feuille."
        )

        return

    # --------------------------------------------------------
    # KNOWN CLASS
    # --------------------------------------------------------

    class_name = result.get("class")

    if not class_name:
        st.error("Impossible de récupérer la prédiction.")
        return

    confidence = result.get("confidence", 0) * 100

    readable_name = format_class_name(class_name)

    st.divider()

    st.subheader("🌿 Résultat de l'analyse")

    st.success(
        f"Maladie détectée : **{readable_name}**"
    )

    st.metric(
        "Confiance du modèle",
        f"{confidence:.2f}%"
    )

    # Informations maladie
    display_disease_info(
        class_name,
        diseases_db
    )


# ============================================================
# MAIN APP
# ============================================================

def main():

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("🌱 PlantDx")

    st.caption(
        "Détection intelligente des maladies des plantes"
    )

    st.divider()

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    diseases_db = load_disease_database()

    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    st.subheader("📷 Importer une feuille")

    uploaded_file = st.file_uploader(
        "Choisissez une image de feuille",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        label_visibility="collapsed",
    )

    # --------------------------------------------------------
    # NO IMAGE
    # --------------------------------------------------------

    if uploaded_file is None:

        st.info(
            "Importez une photo de feuille pour commencer l'analyse."
        )

        return

    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Impossible de lire cette image."
        )

        return

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    st.subheader("👀 Aperçu")

    st.image(
        image,
        use_container_width=True
    )

    # --------------------------------------------------------
    # ANALYSE BUTTON
    # --------------------------------------------------------

    analyze = st.button(
        "🔍 Analyser la feuille",
        type="primary",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # WAIT FOR BUTTON
    # --------------------------------------------------------

    if not analyze:
        return

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with st.spinner(
        "Analyse de la feuille en cours..."
    ):

        try:

            result = predict_image(image)

        except Exception as e:

            st.error(
                "Une erreur est survenue pendant l'analyse."
            )

            st.exception(e)

            return

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    display_prediction(
        result,
        diseases_db
    )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "PlantDx · Computer Vision & Deep Learning"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
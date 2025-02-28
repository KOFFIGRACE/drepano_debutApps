import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import os
import uuid
from datetime import datetime
import time

# Configuration de la page
st.set_page_config(page_title="Suivi Drépanocytose", layout="wide")

# Initialisation des données de session
if 'families' not in st.session_state:
    st.session_state.families = {}

if 'current_family_id' not in st.session_state:
    st.session_state.current_family_id = None

# Fonction pour générer un identifiant unique pour la famille
def generate_family_id():
    return str(uuid.uuid4())[:8].upper()

# Création et entraînement du modèle
def train_model():
    # Créer des données d'entraînement simulées
    X_train = []
    y_train = []
    
    # Générer des exemples pour chaque combinaison possible
    for mere in range(4):  # HbAA, HbAS, HbSC, HbSS
        for pere in range(4):  # HbAA, HbAS, HbSC, HbSS
            for antecedent in range(2):  # Oui, Non
                
                # Logique simplifiée pour la prédiction
                if mere == 0 and pere == 0:  # Deux parents HbAA
                    enfant = 0  # HbAA
                elif (mere == 0 and pere == 3) or (mere == 3 and pere == 0):  # HbAA + HbSS
                    enfant = 1  # HbAS
                elif (mere == 1 and pere == 1):  # Deux parents HbAS
                    # 25% HbAA, 50% HbAS, 25% HbSS
                    for _ in range(4):
                        if _ == 0:
                            X_train.append([mere, pere, antecedent])
                            y_train.append(0)  # HbAA
                        elif _ == 3:
                            X_train.append([mere, pere, antecedent])
                            y_train.append(3)  # HbSS
                        else:
                            X_train.append([mere, pere, antecedent])
                            y_train.append(1)  # HbAS
                    continue
                elif (mere == 1 and pere == 3) or (mere == 3 and pere == 1):  # HbAS + HbSS
                    # 50% HbAS, 50% HbSS
                    X_train.append([mere, pere, antecedent])
                    y_train.append(1)  # HbAS
                    X_train.append([mere, pere, antecedent])
                    y_train.append(3)  # HbSS
                    continue
                elif (mere == 2 or pere == 2):  # Si l'un des parents est HbSC
                    if mere == 0 or pere == 0:  # HbSC + HbAA
                        enfant = 1  # HbAS (simplifié)
                    elif mere == 1 or pere == 1:  # HbSC + HbAS
                        enfant = 2  # HbSC (simplifié)
                    else:  # HbSC + HbSS ou HbSC + HbSC
                        enfant = 2  # HbSC (simplifié)
                else:
                    enfant = 1  # HbAS par défaut pour les autres cas
                
                X_train.append([mere, pere, antecedent])
                y_train.append(enfant)
    
    # Convertir en arrays numpy
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=600, min_samples_split=2, min_samples_leaf=1, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Dictionnaires de mapping
statut_mere_mapping = {'HbAA': 0, 'HbAS': 1, 'HbSC': 2, 'HbSS': 3}
statut_pere_mapping = {'HbAA': 0, 'HbAS': 1, 'HbSC': 2, 'HbSS': 3}
AC_mapping = {'Oui': 0, 'Non': 1}
Statut_Enfant_mapping = {0: 'HbAA', 1: 'HbAS', 2: 'HbSC', 3: 'HbSS'}

# Entraînement du modèle
model = train_model()

# Fonction pour créer un bouton stylisé
def styled_button(label, key=None, type="primary"):
    button_styles = {
        "primary": "background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; width: 100%; margin: 5px 0;",
        "secondary": "background-color: #2196F3; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; width: 100%; margin: 5px 0;",
        "warning": "background-color: #ff9800; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; width: 100%; margin: 5px 0;",
        "danger": "background-color: #f44336; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; width: 100%; margin: 5px 0;"
    }
    
    button_html = f"""
    <button style="{button_styles[type]}" id="{key}">{label}</button>
    """
    return st.markdown(button_html, unsafe_allow_html=True)

# Interface principale avec menu latéral
# ------ SIDEBAR ------
with st.sidebar:
    # Toujours afficher l'image en haut du sidebar
    if os.path.exists("FEMME.png"):
        st.image("FEMME.png", width=200)
    else:
        st.warning("Image 'FEMME.png' non trouvée.")

    st.title("Menu Principal")

    # Container pour les boutons de menu
    menu_container = st.container()
    
    with menu_container:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Accueil", use_container_width=True):
                st.session_state.page = "Accueil"
                st.rerun()
                
            if st.button("🔮 Prédiction", use_container_width=True):
                st.session_state.page = "Nouvelle Prédiction"
                st.rerun()
                
            if st.button("🏥 Suivi", use_container_width=True):
                st.session_state.page = "Suivi Médical" 
                st.rerun()
                
        with col2:
            if st.button("🔍 Recherche", use_container_width=True):
                st.session_state.page = "Recherche Famille"
                st.rerun()
                
            if st.button("ℹ️ Informations", use_container_width=True):
                st.session_state.page = "Informations"
                st.rerun()
                
            if st.button("📊 Statistiques", use_container_width=True):
                st.session_state.page = "Statistiques"
                st.rerun()

    # Afficher l'ID de la famille courante si elle existe
    if st.session_state.current_family_id:
        st.info(f"**Famille en cours**: {st.session_state.current_family_id}")
        if st.button("Effacer sélection", key="clear_family"):
            st.session_state.current_family_id = None
            st.rerun()

# Initialiser la page si elle n'est pas définie
if 'page' not in st.session_state:
    st.session_state.page = "Accueil"

# ------ PAGE D'ACCUEIL ------
if st.session_state.page == "Accueil":
    st.title("Système de Prédiction et Suivi de la Drépanocytose")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if os.path.exists("FEMME.png"):
            st.image("FEMME.png", width=300)
    
    with col2:
        st.header("Bienvenue dans l'application de suivi de drépanocytose")
        st.markdown("""
        Cette application permet aux professionnels de santé de:
        * **Prédire** le statut génétique des enfants à naître
        * **Suivre** l'état de santé des patients atteints de drépanocytose
        * **Générer** des recommandations personnalisées
        * **Maintenir** un dossier médical complet
        """)
        
        # Boutons d'action rapide
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔮 Nouvelle Prédiction", use_container_width=True):
                st.session_state.page = "Nouvelle Prédiction"
                st.rerun()
        with col2:
            if st.button("🔍 Rechercher un Dossier", use_container_width=True):
                st.session_state.page = "Recherche Famille"
                st.rerun()
    
    st.subheader("Qu'est-ce que la Drépanocytose ?")
    st.write("""
        La drépanocytose est une maladie génétique qui affecte les globules rouges du sang, leur donnant une forme de faucille. 
        Cela peut provoquer des douleurs, des infections fréquentes, et d'autres complications. 
        Elle est surtout présente chez les personnes d'origine africaine, méditerranéenne et du Moyen-Orient. 
        Un suivi médical est essentiel pour gérer cette maladie et améliorer la qualité de vie des personnes atteintes.
    """)
    
    # Afficher les statistiques des familles
    st.subheader("Statistiques des familles suivies")
    stats_container = st.container()
    
    with stats_container:
        if len(st.session_state.families) > 0:
            st.success(f"Nombre total de familles suivies: {len(st.session_state.families)}")
            
            # Calculer des statistiques sur les statuts prédits
            statuts_predits = []
            statuts_confirmes = []
            for family_id, family_data in st.session_state.families.items():
                if 'predicted_status' in family_data:
                    statuts_predits.append(family_data['predicted_status'])
                if 'confirmed_status' in family_data and family_data['confirmed_status']:
                    statuts_confirmes.append(family_data['confirmed_status'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if statuts_predits:
                    st.write("### Statuts prédits")
                    statut_counts = pd.Series(statuts_predits).value_counts()
                    st.bar_chart(statut_counts)
            
            with col2:
                if statuts_confirmes:
                    st.write("### Statuts confirmés")
                    statut_confirmed_counts = pd.Series(statuts_confirmes).value_counts()
                    st.bar_chart(statut_confirmed_counts)
        else:
            st.info("Aucune famille n'est encore enregistrée dans le système.")
            
        # Bouton pour aller aux statistiques détaillées
        if st.button("Voir statistiques détaillées", use_container_width=True):
            st.session_state.page = "Statistiques"
            st.rerun()

# ------ PAGE DE NOUVELLE PRÉDICTION ------
elif st.session_state.page == "Nouvelle Prédiction":
    st.title("Nouvelle Prédiction")
    
    # Onglets pour différentes méthodes de création
    tabs = st.tabs(["Nouvelle Famille", "Famille Existante"])
    
    with tabs[0]:
        # Formulaire de prédiction pour nouvelle famille
        with st.form("prediction_form"):
            st.subheader("Informations du couple")
            
            col1, col2 = st.columns(2)
            
            with col1:
                nom_famille = st.text_input("Nom de Famille")
                statut_mere = st.selectbox("Statut Génétique de la Mère", options=['HbAA', 'HbAS', 'HbSC', 'HbSS'])
                antecedents = st.selectbox("Antécédents Familiaux de Drépanocytose", options=['Oui', 'Non'])
            
            with col2:
                prenoms_parents = st.text_input("Prénoms des Parents")
                statut_pere = st.selectbox("Statut Génétique du Père", options=['HbAA', 'HbAS', 'HbSC', 'HbSS'])
                date_consultation = st.date_input("Date de la Consultation")
            
            notes = st.text_area("Notes Additionnelles", height=100)
            
            submitted = st.form_submit_button("Prédire le Statut de l'Enfant")
        
        if submitted and nom_famille:
            with st.spinner("Analyse en cours..."):
                time.sleep(1)  # Simulation d'un traitement
                
                # Conversion des entrées en valeurs numériques
                nouvelle_donnee = [[statut_mere_mapping[statut_mere], statut_pere_mapping[statut_pere], AC_mapping[antecedents]]]
                
                # Prédiction
                prediction = model.predict(nouvelle_donnee)
                statut_enfant_pred = Statut_Enfant_mapping[prediction[0]]
                
                # Génération d'un ID de famille unique
                family_id = generate_family_id()
                
                # Enregistrement de la famille
                st.session_state.families[family_id] = {
                    'name': nom_famille,
                    'prenoms': prenoms_parents,
                    'predicted_status': statut_enfant_pred,
                    'statut_mere': statut_mere,
                    'statut_pere': statut_pere,
                    'antecedents': antecedents,
                    'creation_date': date_consultation.strftime('%Y-%m-%d'),
                    'consultations': [],
                    'baby_born': False,
                    'confirmed_status': None
                }
                
                # Ajouter cette consultation
                st.session_state.families[family_id]['consultations'].append({
                    'date': date_consultation.strftime('%Y-%m-%d'),
                    'type': 'Prédiction',
                    'notes': notes,
                    'recommendations': []
                })
                
                # Définir cette famille comme la famille courante
                st.session_state.current_family_id = family_id
            
            # Affichage des résultats dans une carte stylisée
            st.success("Prédiction réalisée avec succès!")
            
            result_container = st.container()
            with result_container:
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #f9f9f9; margin-bottom: 20px;">
                    <h3 style="color: #2c3e50;">Résultats de la Prédiction</h3>
                    <p><strong>Famille:</strong> {nom_famille} - {prenoms_parents}</p>
                    <p><strong>ID Famille:</strong> {family_id}</p>
                    <p><strong>Statut prédit:</strong> <span style="font-size: 18px; font-weight: bold; color: {'red' if statut_enfant_pred in ['HbSS', 'HbSC'] else 'green'};">{statut_enfant_pred}</span></p>
                    <p><small>Conservez cet identifiant pour les consultations futures.</small></p>
                </div>
                """, unsafe_allow_html=True)
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.subheader("Risques associés")
                    if statut_enfant_pred in ['HbSS', 'HbSC']:
                        st.warning("⚠️ Risque élevé de complications liées à la drépanocytose")
                    elif statut_enfant_pred == 'HbAS':
                        st.info("ℹ️ Porteur du trait drépanocytaire, mais risque de complications réduit")
                    else:
                        st.success("✅ Aucun risque de drépanocytose détecté")
                
                with result_col2:
                    # Recommandations en fonction du statut prédit
                    st.subheader("Recommandations médicales")
                    
                    if statut_enfant_pred in ['HbSS', 'HbSC']:
                        recommendations = [
                            "Suivi médical rapproché avec un hématologue spécialiste",
                            "Supplémentation en acide folique et en fer",
                            "Alimentation riche en fer et en vitamines",
                            "Examens sanguins réguliers",
                            "Vaccination complète incluant le pneumocoque",
                            "Éviter les situations à risque d'infection"
                        ]
                    else:
                        recommendations = [
                            "Suivi prénatal standard",
                            "Alimentation équilibrée",
                            "Hydratation adéquate",
                            "Repos suffisant",
                            "Éviter le stress"
                        ]
                    
                    for i, rec in enumerate(recommendations):
                        st.markdown(f"<p style='margin:5px;'>🔸 {rec}</p>", unsafe_allow_html=True)
                        
                    # Enregistrer ces recommandations
                    st.session_state.families[family_id]['consultations'][-1]['recommendations'] = recommendations
            
            # Boutons d'action rapide
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("📋 Imprimer le rapport", use_container_width=True):
                    st.balloons()
                    st.success("Impression du rapport en cours...")
            
            with action_col2:
                if st.button("🏥 Aller au suivi médical", use_container_width=True):
                    st.session_state.page = "Suivi Médical"
                    st.rerun()
            
            with action_col3:
                if st.button("🔮 Nouvelle prédiction", use_container_width=True):
                    st.session_state.current_family_id = None
                    st.experimental_rerun()
    
    with tabs[1]:
        # Recherche de famille existante pour prédiction
        st.subheader("Rechercher une famille existante")
        
        search_method = st.radio("Méthode de recherche", ["Par ID", "Par Nom"])
        
        if search_method == "Par ID":
            family_id = st.text_input("Identifiant de la famille")
            search = st.button("Rechercher")
            
            if search and family_id:
                if family_id in st.session_state.families:
                    family = st.session_state.families[family_id]
                    st.success(f"Famille trouvée: {family['name']} - {family['prenoms']}")
                    
                    # Afficher informations et permettre nouvelle consultation
                    st.subheader("Informations actuelles")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Statut mère**: {family['statut_mere']}")
                        st.write(f"**Statut père**: {family['statut_pere']}")
                    
                    with col2:
                        st.write(f"**Statut prédit précédent**: {family['predicted_status']}")
                        st.write(f"**Antécédents**: {family['antecedents']}")
                    
                    if st.button("Accéder au dossier de suivi", use_container_width=True):
                        st.session_state.current_family_id = family_id
                        st.session_state.page = "Suivi Médical"
                        st.rerun()
                else:
                    st.error("Identifiant non trouvé.")
        else:
            name = st.text_input("Nom de famille")
            search = st.button("Rechercher")
            
            if search and name:
                found_families = []
                for fid, fdata in st.session_state.families.items():
                    if name.lower() in fdata['name'].lower():
                        found_families.append((fid, fdata))
                
                if found_families:
                    st.success(f"{len(found_families)} famille(s) trouvée(s)")
                    for fid, fdata in found_families:
                        with st.expander(f"{fdata['name']} - {fdata['prenoms']}"):
                            st.write(f"**ID**: {fid}")
                            st.write(f"**Statut prédit**: {fdata['predicted_status']}")
                            st.write(f"**Date de création**: {fdata['creation_date']}")
                            
                            if st.button(f"Sélectionner cette famille", key=f"select_{fid}"):
                                st.session_state.current_family_id = fid
                                st.session_state.page = "Suivi Médical"
                                st.rerun()
                else:
                    st.error("Aucune famille trouvée avec ce nom.")

# ------ PAGE DE SUIVI MÉDICAL ------
elif st.session_state.page == "Suivi Médical":
    st.title("Suivi Médical")
    
    # Si une famille est déjà sélectionnée
    if st.session_state.current_family_id and st.session_state.current_family_id in st.session_state.families:
        family_id = st.session_state.current_family_id
        family = st.session_state.families[family_id]
        
        # Affichage des informations de la famille dans une carte stylisée
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #f9f9f9; margin-bottom: 20px;">
            <h3 style="color: #2c3e50;">{family['name']} - {family['prenoms']}</h3>
            <p><strong>ID Famille:</strong> {family_id}</p>
            <p><strong>Statut prédit:</strong> <span style="font-weight: bold; color: {'red' if family['predicted_status'] in ['HbSS', 'HbSC'] else 'green'};">{family['predicted_status']}</span></p>
            <p><strong>Date de création:</strong> {family['creation_date']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Onglets pour différentes sections du suivi
        tabs = st.tabs(["État de Santé", "Historique", "Plan de Traitement", "Documents"])
        
        with tabs[0]:
            st.subheader("État de Santé Actuel")
            
            # Vérifier si l'enfant est né
            baby_born = st.checkbox("L'enfant est né", value=family['baby_born'])
            
            if baby_born:
                if not family['baby_born']:
                    st.session_state.families[family_id]['baby_born'] = True
                
                # Formulaire pour le suivi post-naissance
                with st.form("health_monitoring_form"):
                    st.write("### Suivi post-naissance")
                    
                    # Statut confirmé après naissance
                    confirmed_status = st.selectbox(
                        "Statut confirmé après naissance", 
                        options=['HbAA', 'HbAS', 'HbSC', 'HbSS'],
                        index=['HbAA', 'HbAS', 'HbSC', 'HbSS'].index(family['confirmed_status']) if family['confirmed_status'] else 0
                    )
                    
                    # Si le statut est différent de celui prédit, afficher une alerte
                    if family['confirmed_status'] and confirmed_status != family['confirmed_status']:
                        st.warning(f"Attention: Vous êtes en train de modifier le statut confirmé de l'enfant de {family['confirmed_status']} à {confirmed_status}.")
                    
                    # Paramètres de santé avec explications
                    st.write("### Paramètres de santé")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        poids = st.number_input("Poids (kg)", min_value=0.0, max_value=20.0, step=0.1)
                        taille = st.number_input("Taille (cm)", min_value=0.0, max_value=150.0, step=0.5)
                        temperature = st.number_input("Température (°C)", min_value=35.0, max_value=42.0, step=0.1, value=37.0)
                    
                    with col2:
                        taux_hb = st.number_input("Taux d'hémoglobine (g/dL)", min_value=0.0, max_value=20.0, step=0.1)
                        st.info("Normal: 11-13 g/dL")
                        
                        globules_blancs = st.number_input("Globules blancs (G/L)", min_value=0.0, max_value=50.0, step=0.1)
                        st.info("Normal: 5-20 G/L")
                    
                    with col3:
                        plaquettes = st.number_input("Plaquettes (G/L)", min_value=0.0, max_value=1000.0, step=1.0)
                        st.info("Normal: 150-400 G/L")
                        
                        saturation = st.number_input("Saturation en oxygène (%)", min_value=80.0, max_value=100.0, step=0.5, value=97.0)
                        st.info("Normal: >95%")
                    
                    # Symptômes
                    st.write("### Symptômes observés")
                    
                    symptomes_options = [
                        "Douleurs", "Fièvre", "Fatigue", "Pâleur", "Ictère", 
                        "Infections", "Crises vaso-occlusives", "Douleurs abdominales",
                        "Douleurs thoraciques", "Difficultés respiratoires", "Aucun"
                    ]
                    
                    symptomes = st.multiselect(
                        "Sélectionnez tous les symptômes observés",
                        options=symptomes_options
                    )
                    
                    # Niveau de douleur si "Douleurs" est sélectionné
                    if "Douleurs" in symptomes or "Crises vaso-occlusives" in symptomes:
                        douleur_niveau = st.slider(
                            "Niveau de douleur (0-10)", 
                            min_value=0, 
                            max_value=10, 
                            value=5
                        )
                        
                        localisation_douleur = st.multiselect(
                            "Localisation de la douleur",
                            options=["Bras", "Jambes", "Dos", "Abdomen", "Thorax", "Tête", "Articulations"]
                        )
                    else:
                        douleur_niveau = 0
                        localisation_douleur = []
                    
                    # Respect des recommandations
                    respect_recommandations = st.slider(
                        "Respect des recommandations médicales", 
                        min_value=0, 
                        max_value=10, 
                        value=5,
                        help="0 = Non respectées, 10 = Parfaitement respectées"
                    )
                    
                    # Notes de consultation et date
                    date_consultation = st.date_input("Date de cette consultation", value=datetime.now())
                    notes_consultation = st.text_area("Notes de consultation", height=100)
                    
                    # Bouton de soumission
                    submitted = st.form_submit_button("Enregistrer la consultation", use_container_width=True)
                
                if submitted:
                    # Mise à jour du statut confirmé
                    st.session_state.families[family_id]['confirmed_status'] = confirmed_status
                    
                                        # Enregistrement de la consultation
                    nouvelle_consultation = {
                        'date': date_consultation.strftime('%Y-%m-%d'),
                        'type': 'Suivi post-naissance',
                        'poids': poids,
                        'taille': taille,
                        'temperature': temperature,
                        'taux_hb': taux_hb,
                        'globules_blancs': globules_blancs,
                        'plaquettes': plaquettes,
                        'saturation': saturation,
                        'symptomes': symptomes,
                        'douleur_niveau': douleur_niveau,
                        'localisation_douleur': localisation_douleur,
                        'respect_recommandations': respect_recommandations,
                        'notes': notes_consultation,
                        'recommendations': []
                    }

                    # Ajouter la consultation à l'historique
                    st.session_state.families[family_id]['consultations'].append(nouvelle_consultation)

                    # Affichage d'un message de succès
                    st.success("Consultation enregistrée avec succès!")

            else:
                # Si l'enfant n'est pas encore né
                st.info("L'enfant n'est pas encore né. Veuillez revenir après la naissance pour enregistrer les informations de santé.")

        with tabs[1]:
            # Historique des consultations
            st.subheader("Historique des Consultations")

            if family['consultations']:
                for i, consultation in enumerate(family['consultations']):
                    with st.expander(f"Consultation du {consultation['date']} - {consultation['type']}"):
                        st.write(f"**Type:** {consultation['type']}")
                        st.write(f"**Notes:** {consultation['notes']}")
                        if 'poids' in consultation:
                            st.write(f"**Poids:** {consultation['poids']} kg")
                            st.write(f"**Taille:** {consultation['taille']} cm")
                            st.write(f"**Température:** {consultation['temperature']} °C")
                            st.write(f"**Taux d'hémoglobine:** {consultation['taux_hb']} g/dL")
                            st.write(f"**Globules blancs:** {consultation['globules_blancs']} G/L")
                            st.write(f"**Plaquettes:** {consultation['plaquettes']} G/L")
                            st.write(f"**Saturation en oxygène:** {consultation['saturation']} %")
                            st.write(f"**Symptômes:** {', '.join(consultation['symptomes']) if consultation['symptomes'] else 'Aucun'}")
                            if 'douleur_niveau' in consultation:
                                st.write(f"**Niveau de douleur:** {consultation['douleur_niveau']}/10")
                                st.write(f"**Localisation de la douleur:** {', '.join(consultation['localisation_douleur']) if consultation['localisation_douleur'] else 'Aucune'}")
                            st.write(f"**Respect des recommandations:** {consultation['respect_recommandations']}/10")
            else:
                st.info("Aucune consultation enregistrée pour cette famille.")

        with tabs[2]:
            # Plan de traitement
            st.subheader("Plan de Traitement")

            if family['confirmed_status']:
                st.write(f"Statut confirmé de l'enfant: **{family['confirmed_status']}**")
                if family['confirmed_status'] in ['HbSS', 'HbSC']:
                    st.warning("⚠️ Plan de traitement intensif requis")
                    st.markdown("""
                    **Recommandations:**
                    - Suivi médical régulier avec un hématologue
                    - Supplémentation en acide folique et en fer
                    - Vaccination complète (pneumocoque, méningocoque, etc.)
                    - Éviter les situations à risque d'infection
                    - Gestion des crises vaso-occlusives avec des antalgiques
                    """)
                else:
                    st.success("✅ Aucun plan de traitement spécifique requis")
                    st.markdown("""
                    **Recommandations:**
                    - Suivi médical standard
                    - Alimentation équilibrée
                    - Hydratation adéquate
                    - Repos suffisant
                    """)
            else:
                st.info("Veuillez confirmer le statut de l'enfant après la naissance pour générer un plan de traitement.")

        with tabs[3]:
            # Documents
            st.subheader("Documents")

            st.info("Cette section est en cours de développement. Elle permettra de stocker et de visualiser les documents médicaux liés à la famille.")

    else:
        st.warning("Veuillez sélectionner une famille pour accéder au suivi médical.")

# ------ PAGE DE RECHERCHE FAMILLE ------
elif st.session_state.page == "Recherche Famille":
    st.title("Recherche Famille")

    search_method = st.radio("Méthode de recherche", ["Par ID", "Par Nom"])

    if search_method == "Par ID":
        family_id = st.text_input("Identifiant de la famille")
        search = st.button("Rechercher")

        if search and family_id:
            if family_id in st.session_state.families:
                family = st.session_state.families[family_id]
                st.success(f"Famille trouvée: {family['name']} - {family['prenoms']}")

                # Afficher informations et permettre nouvelle consultation
                st.subheader("Informations actuelles")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Statut mère:** {family['statut_mere']}")
                    st.write(f"**Statut père:** {family['statut_pere']}")

                with col2:
                    st.write(f"**Statut prédit précédent:** {family['predicted_status']}")
                    st.write(f"**Antécédents:** {family['antecedents']}")

                if st.button("Accéder au dossier de suivi", use_container_width=True):
                    st.session_state.current_family_id = family_id
                    st.session_state.page = "Suivi Médical"
                    st.rerun()
            else:
                st.error("Identifiant non trouvé.")
    else:
        name = st.text_input("Nom de famille")
        search = st.button("Rechercher")

        if search and name:
            found_families = []
            for fid, fdata in st.session_state.families.items():
                if name.lower() in fdata['name'].lower():
                    found_families.append((fid, fdata))

            if found_families:
                st.success(f"{len(found_families)} famille(s) trouvée(s)")
                for fid, fdata in found_families:
                    with st.expander(f"{fdata['name']} - {fdata['prenoms']}"):
                        st.write(f"**ID:** {fid}")
                        st.write(f"**Statut prédit:** {fdata['predicted_status']}")
                        st.write(f"**Date de création:** {fdata['creation_date']}")

                        if st.button(f"Sélectionner cette famille", key=f"select_{fid}"):
                            st.session_state.current_family_id = fid
                            st.session_state.page = "Suivi Médical"
                            st.rerun()
            else:
                st.error("Aucune famille trouvée avec ce nom.")

# ------ PAGE D'INFORMATIONS ------
elif st.session_state.page == "Informations":
    st.title("Informations sur la Drépanocytose")

    st.markdown("""
    ### Qu'est-ce que la drépanocytose ?
    La drépanocytose est une maladie génétique qui affecte les globules rouges du sang, leur donnant une forme de faucille. 
    Cela peut provoquer des douleurs, des infections fréquentes, et d'autres complications. 
    Elle est surtout présente chez les personnes d'origine africaine, méditerranéenne et du Moyen-Orient. 
    Un suivi médical est essentiel pour gérer cette maladie et améliorer la qualité de vie des personnes atteintes.
    """)

    st.markdown("""
    ### Comment fonctionne cette application ?
    Cette application permet aux professionnels de santé de:
    - **Prédire** le statut génétique des enfants à naître
    - **Suivre** l'état de santé des patients atteints de drépanocytose
    - **Générer** des recommandations personnalisées
    - **Maintenir** un dossier médical complet
    """)

# ------ PAGE DE STATISTIQUES ------
elif st.session_state.page == "Statistiques":
    st.title("Statistiques")

    if len(st.session_state.families) > 0:
        st.success(f"Nombre total de familles suivies: {len(st.session_state.families)}")

        # Calculer des statistiques sur les statuts prédits
        statuts_predits = []
        statuts_confirmes = []
        for family_id, family_data in st.session_state.families.items():
            if 'predicted_status' in family_data:
                statuts_predits.append(family_data['predicted_status'])
            if 'confirmed_status' in family_data and family_data['confirmed_status']:
                statuts_confirmes.append(family_data['confirmed_status'])

        col1, col2 = st.columns(2)

        with col1:
            if statuts_predits:
                st.write("### Statuts prédits")
                statut_counts = pd.Series(statuts_predits).value_counts()
                st.bar_chart(statut_counts)

        with col2:
            if statuts_confirmes:
                st.write("### Statuts confirmés")
                statut_confirmed_counts = pd.Series(statuts_confirmes).value_counts()
                st.bar_chart(statut_confirmed_counts)
    else:
        st.info("Aucune famille n'est encore enregistrée dans le système.")
# Création d'una application avec Streamlit dans laquelle nous allons présenter notre notebook

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="Cancer du poumon : Analyse et Visualisation 🩺", layout="wide")

st.markdown("<h1 style='color:#1976D2;'>Cancer du poumon : Analyse et Visualisation 🩺</h1>", 
            unsafe_allow_html=True)


# Notre application sera constituer de 3 pages

# Création d'une barre horizontale pour la navigation entre les pages
tabs = st.tabs(["Le Jeu de données", "Visualisation du jeu de données", "L'article"])


# Page 1 : Le Jeu de données
with tabs[0]:
    with st.sidebar:
        st.header("Jeu de données 🧮")
        st.write("Ici, vous pouvez explorer le jeu de données.")
        show_intro1 = st.checkbox("Présentation", value=True)
        show_head = st.checkbox("Aperçu du jeu de données")
        show_shape = st.checkbox("Les informations du jeu de données")
        show_describe = st.checkbox("Statistiques descriptives")
        show_cat = st.checkbox("Les variables catégorielles")
        show_num = st.checkbox("Les variables numériques")
        show_date = st.checkbox("Les variables temporelles")

    if show_intro1:
        st.header("Le Jeu de données")
        st.write("Dans cette page, nous allons présenter et analyser le jeu de données utilisé dans notre notebook.")
        st.write("Le jeu de données utilisé est le suivant :")
        st.write("https://www.kaggle.com/datasets/amankumar094/lung-cancer-dataset")
        st.write("Ce jeu de données contient des informations sur les patients atteints de cancer du poumon, y compris des caractéristiques démographiques, des résultats de tests médicaux et des diagnostics.")
        st.write("Nous allons explorer les données, effectuer des analyses statistiques et visualiser les résultats pour mieux comprendre les facteurs associés au cancer du poumon.")


    # On importe le jeux de données
    df = pd.read_pickle("dataset_med_cleaned.pkl")


    if show_head:
        # Affichage des premières lignes du dataset
        st.header("Les premières lignes du jeu de données :")
        st.dataframe(df.head(10))

    if show_shape:
        # Affichage des informations sur le dataset
        st.header("Les informations sur le jeus de données :")
        # Les colonnes du jeu de données
        st.write("Le jeu de données contient", df.shape[0], "lignes et", df.shape[1], "colonnes.")
        st.write("Les colonnes du jeu de données sont les suivantes :")
        st.write(df.columns.tolist())
        # Les types de données de chaque colonne
        st.write("Les types de données de chaque colonne sont les suivants :")
        st.write(df.dtypes)
        # Les types de données de chaque colonne, le nombre d'éléments non nuls et l'utilisation de la mémoire
        st.write("Les types de données de chaque colonne, le nombre d'éléments non nuls et l'utilisation de la mémoire :")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    
    if show_describe:
        # statistiques descriptives sans la colonne 'id'
        st.header("Statistiques descriptives du jeu de données :")
        st.write(df.drop(columns=['id', 'diagnosis_date', 'end_treatment_date'], errors='ignore').describe())

    if show_cat:
        # Les variables catégorielles
        st.header("Les variables catégorielles :")
        st.write("Les variables catégorielles du jeu de données sont les suivantes :")
        st.write(df.select_dtypes(include=['object']).columns.tolist())

        ## Nous pouvons faire des observations sur les décès en fonction du genre
        st.write("Observation sur les décès en fonction du genre :")
        st.write(df[df['survived'] == 'No'].groupby('gender').size().reset_index(name='count'))
        # On peut aussi faire des observations sur les pays d'origine des patients
        # En triant les pays par le nombre de patients du plus élevé au plus bas
        st.write("Les pays d'origine des patients sont les suivants :")
        st.write(df.groupby('country').size().reset_index(name='count').sort_values(by='count', ascending=False))
    
    if show_num:
        # Les variables numériques
        st.header("Les variables numériques :")
        st.write("Les variables numériques du jeu de données sont les suivantes :")
        st.write(df.select_dtypes(include=['int64', 'float64']).columns.tolist())
        # On peut calculer l'age moyen des patients
        st.write("L'âge moyen des patients est de :", df['age'].mean())
        # On peut calculer quelle est la moyenne des patients qui ont survécu en n'ayant pas d'autre cancer
        # Conversion temporaire pour l'affichage, sans modifier le DataFrame
        surv_no = df.loc[df['other_cancer'] == 'No', 'survived'].map({'Yes': 1, 'No': 0}).mean()
        surv_yes = df.loc[df['other_cancer'] == 'Yes', 'survived'].map({'Yes': 1, 'No': 0}).mean()

        st.write("La moyenne des patients qui ont survécu en n'ayant pas d'autre cancer est de :", surv_no)
        st.write("La moyenne des patients qui ont survécu en ayant un autre cancer est de :", surv_yes)
    
    if show_date:
        # Les variables temporelles
        st.header("Les variables temporelles :")
        # On va d'abord identifier les variables temporelles
        st.write("Les variables temporelles du jeu de données sont les suivantes :")
        st.write(df.select_dtypes(include=['datetime64[ns]']).columns.tolist())  
        # On peut regarder les extremes des dates de diagnostic et de fin de traitement
        st.write("Les dates minimales et maximales de diagnostic et de fin de traitement sont les suivantes :")
        # les mettre dans un dataframe pour les afficher
        date_info = {
            'diagnosis_date': [df['diagnosis_date'].min(), df['diagnosis_date'].max()],
            'end_treatment_date': [df['end_treatment_date'].min(), df['end_treatment_date'].max()]
        }       
        date_df = pd.DataFrame(date_info, index=['min', 'max'])
        st.dataframe(date_df)

# Page 2 : Visualition du jeu de données
with tabs[1]:
    with st.sidebar:
        st.header("Visualisation 📊")
        st.write("Ici, vous pouvez visualiser le jeu de données.")
        show_intro2 = st.checkbox("Présentation", value=True, key="intro2")
        with st.expander("Graphiques disponibles"):
            show_graph1 = st.checkbox("Graphique 1 : Diagnostics par année", value=False, key="graph1")
            show_graph2 = st.checkbox("Graphique 2 : Décès par genre", value=False, key="graph2")
            show_graph3 = st.checkbox("Graphique 3 : Type de traitement et genre", value=False, key="graph3")
            show_graph4 = st.checkbox("Graphique 4 : Durée par stade et comorbidité", value=False, key="graph4")
            show_graph5 = st.checkbox("Graphique 5 : Durée par type de traitement", value=False, key="graph5")
            show_graph6 = st.checkbox("Graphique 6 : Matrice de corrélation", value=False, key="graph6")


    if show_intro2:
        st.header("Visualition du jeu de données")
        st.write("Dans cette page, nous allons visualiser le jeu de données et créer des graphiques pour mieux comprendre les données.")
    
    if show_graph1:
            st.header("Évolution du nombre de diagnostics par année")


            # Filtre dynamique par tranche d'âge
            age_groups = df['age_group'].dropna().unique().tolist()
            selected_age_groups = st.multiselect(
                "Filtrer par tranche d'âge",
                options=age_groups,
                default=age_groups,
                key="age_group_graph1"
            )

            # Filtre dynamique par pays
            countries = df['country'].dropna().unique().tolist()
            selected_countries = st.multiselect(
                "Filtrer par pays",
                options=countries,
                default=countries,
                key="country_graph1"
            )

            # Filtrer le DataFrame selon la tranche d'âge et le pays sélectionnés
            df_filtered = df[
                (df['age_group'].isin(selected_age_groups)) &
                (df['country'].isin(selected_countries))
            ].copy()
            df_filtered['diagnosis_year'] = df_filtered['diagnosis_date'].dt.year

            # Déterminer les bornes du slider
            min_year = int(df_filtered['diagnosis_year'].min())
            max_year = int(df_filtered['diagnosis_year'].max())

            # --- Slider pour sélectionner la période à zoomer ---
            zoom_years = st.slider(
                "Sélectionner la période à zoomer",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                step=1,
                key="slider_graph1"
            )

            diagnosis_per_year = df_filtered['diagnosis_year'].value_counts().sort_index()

            # Masque pour la période de zoom
            mask = (diagnosis_per_year.index >= zoom_years[0]) & (diagnosis_per_year.index <= zoom_years[1])

            # Création des subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Évolution globale", f"Zoom {zoom_years[0]}-{zoom_years[1]}"),
                shared_yaxes=False
            )

            # Courbe globale
            fig.add_trace(
                go.Scatter(
                    x=diagnosis_per_year.index,
                    y=diagnosis_per_year.values,
                    mode='lines+markers',
                    name='Total',
                    line=dict(color='royalblue', width=2)
                ),
                row=1, col=1
            )

            # Courbe zoomée
            fig.add_trace(
                go.Scatter(
                    x=diagnosis_per_year.index[mask],
                    y=diagnosis_per_year.values[mask],
                    mode='lines+markers',
                    name=f'Zoom {zoom_years[0]}-{zoom_years[1]}',
                    line=dict(color='crimson', width=2)
                ),
                row=1, col=2
            )

            fig.update_layout(
                title={
                    'text': "Évolution du nombre de diagnostics de cancer par année (par tranche d'âge et pays)",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Année de diagnostic",
                yaxis_title="Nombre de diagnostics",
                width=1000,
                height=400
            )
            fig.update_xaxes(title_text="Année de diagnostic", row=1, col=1)
            fig.update_xaxes(title_text="Année de diagnostic", row=1, col=2)
            fig.update_yaxes(title_text="Nombre de diagnostics", row=1, col=1)
            fig.update_yaxes(title_text="Nombre de diagnostics", row=1, col=2)

            st.plotly_chart(fig, use_container_width=True)
    
    if show_graph2:
            st.header("Décès par cancer en fonction du genre")

            # Filtres dynamiques avec clés uniques
            pays2 = df['country'].dropna().unique().tolist()
            selected_pays2 = st.multiselect(
                "Filtrer par pays",
                options=pays2,
                default=pays2,
                key="pays_graph2"
            )

            age_groups2 = df['age_group'].dropna().unique().tolist()
            selected_age_groups2 = st.multiselect(
                "Filtrer par tranche d'âge",
                options=age_groups2,
                default=age_groups2,
                key="age_group_graph2"
            )

            bmi_categories2 = df['bmi_category'].dropna().unique().tolist()
            selected_bmi_categories2 = st.multiselect(
                "Filtrer par catégorie d'IMC",
                options=bmi_categories2,
                default=bmi_categories2,
                key="bmi_graph2"
            )

            # Appliquer les filtres
            df_graph2 = df[
                (df['country'].isin(selected_pays2)) &
                (df['age_group'].isin(selected_age_groups2)) &
                (df['bmi_category'].isin(selected_bmi_categories2))
            ]
            deces = df_graph2[df_graph2['survived'] == 'No']

            fig = px.bar(
                deces.groupby('gender').size().reset_index(name='count'),
                x='gender',
                y='count',
                color='gender',
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Décès par cancer en fonction du genre",
                labels={'gender': 'Genre', 'count': 'Nombre de décès'}
            )
            fig.update_layout(
                title={
                    'text': "Décès par cancer en fonction du genre",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Genre',
                yaxis_title='Nombre de décès'
            )
            st.plotly_chart(fig, use_container_width=True)

    if show_graph3:
            st.header("Durée de traitement par type de traitement et genre")
            # Filtres dynamiques pour Graphique 3
            years3 = df['diagnosis_date'].dt.year.dropna().astype(int)
            min_year3 = int(years3.min())
            max_year3 = int(years3.max())
            annee_debut3, annee_fin3 = st.slider(
                "Filtrer par période de diagnostic",
                min_value=min_year3,
                max_value=max_year3,
                value=(min_year3, max_year3),
                step=1,
                key="slider_graph3"
            )

            genres3 = df['gender'].dropna().unique().tolist()
            selected_genres3 = st.multiselect(
                "Filtrer par genre",
                options=genres3,
                default=genres3,
                key="genre_graph3"
            )

            pays3 = df['country'].dropna().unique().tolist()
            selected_pays3 = st.multiselect(
                "Filtrer par pays",
                options=pays3,
                default=pays3,
                key="pays_graph3"
            )

            other_pathologies3 = df['autre_pathologie'].dropna().unique().tolist()
            selected_other_pathologies3 = st.multiselect(
                "Filtrer par autre pathologie",
                options=other_pathologies3,
                default=other_pathologies3,
                key="other_pathology_graph3"
            )

            # Appliquer les filtres
            df_graph3 = df[
                (df['diagnosis_date'].dt.year >= annee_debut3) &
                (df['diagnosis_date'].dt.year <= annee_fin3) &
                (df['gender'].isin(selected_genres3)) &
                (df['country'].isin(selected_pays3)) &
                (df['autre_pathologie'].isin(selected_other_pathologies3))
            ]

            import plotly.express as px
            fig = px.histogram(
                df_graph3,
                x="treatment_type",
                color="gender",
                barmode="group",
                title="Évolution de la durée de traitement en fonction du type de traitement et du genre",
                labels={"treatment_type": "Type de traitement", "count": "Nombre de patients", "gender": "Genre"},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig.update_layout(
                title={
                    'text': "Évolution de la durée de traitement en fonction du type de traitement et du genre",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Type de traitement",
                yaxis_title="Nombre de patients",
                legend_title="Genre"
            )
            st.plotly_chart(fig, use_container_width=True)

    if show_graph4:
            st.header("Heatmap interactive des facteurs de risque par pays")

            # Filtres dynamiques
            pays4 = df['country'].dropna().unique().tolist()
            selected_pays4 = st.multiselect(
                "Filtrer par pays",
                options=pays4,
                default=pays4,
                key="pays_graph4"
            )

            facteurs = ['family_history', 'cholesterol_level', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer']
            selected_facteurs = st.multiselect(
                "Filtrer les facteurs de risque",
                options=facteurs,
                default=facteurs,
                key="facteurs_graph4"
            )

            # Calculer les données pour la heatmap filtrée
            heatmap_data = df[df['country'].isin(selected_pays4)].groupby('country').agg({
                'family_history': lambda x: (x == 'Yes').mean() * 100 if 'family_history' in selected_facteurs else None,
                'cholesterol_level': 'mean' if 'cholesterol_level' in selected_facteurs else None,
                'hypertension': lambda x: (x == 'Yes').mean() * 100 if 'hypertension' in selected_facteurs else None,
                'asthma': lambda x: (x == 'Yes').mean() * 100 if 'asthma' in selected_facteurs else None,
                'cirrhosis': lambda x: (x == 'Yes').mean() * 100 if 'cirrhosis' in selected_facteurs else None,
                'other_cancer': lambda x: (x == 'Yes').mean() * 100 if 'other_cancer' in selected_facteurs else None
            })

            # Garder uniquement les colonnes sélectionnées
            heatmap_data = heatmap_data[[col for col in selected_facteurs if col in heatmap_data.columns]]

            # Renommer les colonnes pour plus de clarté
            rename_dict = {
                'family_history': '% Antécédents familiaux',
                'cholesterol_level': 'Cholestérol (moyenne)',
                'hypertension': '% Hypertension',
                'asthma': '% Asthme',
                'cirrhosis': '% Cirrhose',
                'other_cancer': '% Autres cancers'
            }
            heatmap_data.rename(columns=rename_dict, inplace=True)


            fig = px.imshow(
                heatmap_data.values,
                labels=dict(x="Facteurs", y="Pays", color="Valeur"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale="viridis",
                text_auto=".1f",
                aspect="auto"
            )

            fig.update_layout(
                 title={
                    'text': "Heatmap interactive des facteurs de risque par pays",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                title_x=0.5,
                xaxis_title="Facteurs",
                yaxis_title="Pays",
                width=1200,
                height=30 * len(heatmap_data),
                font=dict(size=12),
                coloraxis_colorbar=dict(
                    len=1,
                    y=0.5,
                    thickness=20
                )
            )

            fig.update_xaxes(tickangle=0)

            st.plotly_chart(fig, use_container_width=True)

    if show_graph5:
            st.header("Durée du traitement en fonction du type de traitement")

            # --- Filtres dynamiques pour Graphique 5 ---
            genres5 = df['gender'].dropna().unique().tolist()
            selected_genres5 = st.multiselect(
                "Filtrer par genre",
                options=genres5,
                default=genres5,
                key="genre_graph5"
            )

            pays5 = df['country'].dropna().unique().tolist()
            selected_pays5 = st.multiselect(
                "Filtrer par pays",
                options=pays5,
                default=pays5,
                key="pays_graph5"
            )

            treatments5 = df['treatment_type'].dropna().unique().tolist()
            selected_treatments5 = st.multiselect(
                "Filtrer par type de traitement",
                options=treatments5,
                default=treatments5,
                key="treatment_graph5"
            )

            # Appliquer les filtres
            df_graph5 = df[
                (df['gender'].isin(selected_genres5)) &
                (df['country'].isin(selected_pays5)) &
                (df['treatment_type'].isin(selected_treatments5))
            ]

            fig = px.box(
                df_graph5,
                x="treatment_type",
                y="treatment_duration",
                color="treatment_type",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Durée du traitement en fonction du type de traitement",
                labels={
                    "treatment_type": "Type de traitement",
                    "treatment_duration": "Durée du traitement (jours)"
                }
            )
            fig.update_layout(
                title={
                    'text': "Durée du traitement en fonction du type de traitement",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Type de traitement",
                yaxis_title="Durée du traitement (jours)"
            )
            st.plotly_chart(fig, use_container_width=True)

    if show_graph6:
                st.header("Matrice de corrélation des variables")
                # Filtres dynamiques
                genres6 = df['gender'].dropna().unique().tolist()
                selected_genres6 = st.multiselect(
                    "Filtrer par genre",
                    options=genres6,
                    default=genres6,
                    key="genre_graph6"
                )

                pays6 = df['country'].dropna().unique().tolist()
                selected_pays6 = st.multiselect(
                    "Filtrer par pays",
                    options=pays6,
                    default=pays6,
                    key="pays_graph6"
                )

                age_groups6 = df['age_group'].dropna().unique().tolist()
                selected_age_groups6 = st.multiselect(
                    "Filtrer par tranche d'âge",
                    options=age_groups6,
                    default=age_groups6,
                    key="age_group_graph6"
                )

                # Appliquer les filtres
                df_corr = df[
                    (df['gender'].isin(selected_genres6)) &
                    (df['country'].isin(selected_pays6)) &
                    (df['age_group'].isin(selected_age_groups6))
                ].copy()

                # Colonnes à encoder
                cols_to_encode = ['hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'survived', 'autre_pathologie', 'family_history']
                for col in cols_to_encode:
                    if col in df_corr.columns:
                        df_corr[col] = df_corr[col].map({'Yes': 1, 'No': 0})

                # Exclure la variable 'treatment_duration' de la matrice de corrélation
                if 'treatment_duration' in df_corr.columns:
                    df_corr = df_corr.drop(columns=['treatment_duration'])

                correlation_matrix = df_corr.corr(numeric_only=True)

                fig = px.imshow(
                    correlation_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu",  # ou "Blues" pour bleu clair
                    aspect="auto"
                )

                fig.update_layout(
                    title={
                        'text': "Matrice de corrélation des variables",
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    width=800,
                    height=700,
                    xaxis_title="Variables",
                    yaxis_title="Variables"
                )
                st.plotly_chart(fig, use_container_width=True)
   

# Page 3 : L'article
with tabs[2]:
    with st.sidebar:
        st.header("Article 📰")
        st.write("Ici, vous pouvez explorer l'article en lien avec le jeu de données.")
        show_intro3 = st.checkbox("Présentation", value=True, key="intro3")
        show_article = st.checkbox("Afficher l'article")
        show_nuage = st.checkbox("Afficher le nuage de mots")
    
    if show_intro3:
        st.header("L'article")
        st.write("Dans cette page, nous allons présenter l'article en lien avec notre jeu de données que nous avons choisit.")

        # Lien vers l'article
        st.write("Voici le lien vers l'article :")
        st.write("https://www.pourquoidocteur.fr/Articles/Question-d-actu/51754-Cancer-poumon-focalisation-tabac-a-t-elle-cache-d-autres-facteurs-risque")

    if show_article:
        # Afficher l'article
        with open("article.md", "r", encoding="utf-8") as f:
            article_md = f.read()
        st.markdown(article_md, unsafe_allow_html=True)

    if show_nuage:
        # Afficher le nuage de mot que nous avons créer à partir de du corpus lemmatisé et enregistré dans le fichier nuage_mots.png
        st.write("Voici le nuage de mots créé à partir de l'article :")
        st.image("nuage_mots.png", caption="Nuage de mots", use_container_width=True)



#L'application est terminée
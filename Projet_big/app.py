import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Fonction pour nettoyer les noms de colonnes : minuscules, pas d'espaces, underscore à la place
def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
    return df

@st.cache_data
def load_data():
    clients_0 = pd.read_csv("Clients_0.csv", sep=",", encoding="latin1")
    clients_8 = pd.read_csv("Clients_8.csv", sep=",", encoding="latin1")
    catalogue = pd.read_csv("Catalogue.csv", sep=",", encoding="latin1")
    co2 = pd.read_csv("CO2.csv", sep=",", encoding="latin1")
    marketing = pd.read_csv("Marketing.csv", sep=",", encoding="latin1")
    immatriculation = pd.read_csv("Immatriculations.csv", sep=",", encoding="latin1")


    # Nettoyage colonnes catalogue et co2 (et autres si besoin)
    catalogue = clean_columns(catalogue)
    co2 = clean_columns(co2)
    clients_0.columns = clients_0.columns.str.strip()
    clients_8.columns = clients_8.columns.str.strip()
    marketing.columns = marketing.columns.str.strip()
    immatriculation.columns = immatriculation.columns.str.strip()

    return clients_0, clients_8, catalogue, co2, marketing, immatriculation

clients_df, clients_8_df, catalogue_df, co2_df, marketing_df, immatriculation_df = load_data()

# Renommage manuel s’il y a encore des erreurs dans co2
co2_df.rename(columns={
    "marque___modele": "marque_modele",
    "bonus___malus": "bonus_malus",
    "cout_enerie": "cout_energie",
    "rejets_co2_g/km": "rejets_co2_g_km"
}, inplace=True)

st.title("🚗 Système de recommandation de véhicules électriques")

# Affichage des données clients
st.header("📋 Données clients")
st.dataframe(clients_df)

# Statistiques générales
st.subheader("📊 Statistiques générales")
st.write("Répartition par sexe :")
st.bar_chart(clients_df["sexe"].value_counts())

st.write("Répartition par situation familiale :")
st.bar_chart(clients_df["situationFamiliale"].value_counts())

st.write("Répartition du nombre d'enfants à charge :")
st.bar_chart(clients_df["nbEnfantsAcharge"].value_counts())

# Préparation des données pour clustering
features = ['age', 'taux', 'nbEnfantsAcharge']
X = clients_df[features].copy()

# Nettoyage : remplacer '?' et espaces vides par NaN
X.replace(['?', ' '], np.nan, inplace=True)
X = X.astype(float)
X.dropna(inplace=True)

clients_df_cleaned = clients_df.loc[X.index].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clients_df_cleaned['cluster'] = kmeans.fit_predict(X_scaled)

st.subheader("🧠 Segmentation des clients")
st.dataframe(clients_df_cleaned[['immatriculation', 'cluster'] + features])

# Sélection client
st.header("🔍 Recommandation de véhicules")
selected_client = st.selectbox("Sélectionnez un client :", clients_df_cleaned["immatriculation"])
client_data = clients_df_cleaned[clients_df_cleaned["immatriculation"] == selected_client].iloc[0]
st.write("Données du client sélectionné :")
st.json(client_data.to_dict())

cluster = client_data['cluster']
st.subheader(f"🚘 Recommandations pour le cluster {cluster}")

# Nettoyer colonnes catalogue et co2 (déjà fait, mais au cas où)
catalogue_df = clean_columns(catalogue_df)
co2_df = clean_columns(co2_df)

# Fusion catalogue + co2 sur 'marque_modele'
catalogue_co2_df = pd.merge(
    catalogue_df,
    co2_df[['marque_modele', 'bonus_malus', 'cout_energie']],
    on='marque_modele',
    how='left'
)

# Tri selon cluster
if cluster == 0:
    recommandations = catalogue_co2_df.sort_values('cout_energie').head(5)
elif cluster == 1:
    recommandations = catalogue_co2_df.sort_values('bonus_malus', ascending=False).head(5)
else:
    recommandations = catalogue_co2_df.head(5)

st.dataframe(recommandations[["marque_modele", "cout_energie", "bonus_malus"]])

# Analyse des émissions de CO2
st.subheader("🌿 Analyse des émissions de CO2")
merged_df = pd.merge(clients_df_cleaned, co2_df, on='immatriculation', how='left')

if "rejets_co2_g_km" in merged_df.columns:
    fig, ax = plt.subplots()
    sns.histplot(merged_df["rejets_co2_g_km"].dropna(), bins=20, kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.warning("Colonne 'rejets_co2_g_km' introuvable dans CO2.")

st.success("✅ Recommandations générées avec succès !")

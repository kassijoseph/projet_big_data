import streamlit as st
import pandas as pd
import re

@st.cache_data
def load_clients():
    return pd.read_parquet("results/clients_clustered.parquet")

@st.cache_data
def load_co2():
    df = pd.read_csv("data/co2.csv", encoding='latin1')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')

    def clean_bonus_malus(val):
        if pd.isna(val):
            return 0
        match = re.search(r'-?\d+[\.,]?\d*', str(val))
        if match:
            num = match.group(0).replace(',', '.')
            return float(num)
        return 0

    def clean_cout_energie(val):
        if pd.isna(val):
            return 0
        match = re.search(r'\d+[\.,]?\d*', str(val))
        if match:
            num = match.group(0).replace(',', '.')
            return float(num)
        return 0

    df['bonus_malus'] = df['bonus_malus'].apply(clean_bonus_malus)
    df['cout_energie'] = df['cout_energie'].apply(clean_cout_energie)

    return df

def recommend_vehicles(cluster, co2_df):
    if cluster == 0:
        filtered = co2_df.sort_values('cout_energie').head(5)
    elif cluster == 1:
        filtered = co2_df.sort_values('bonus_malus', ascending=False).head(5)
    elif cluster == 2:
        filtered = co2_df.sort_values(['cout_energie', 'bonus_malus']).head(5)
    else:
        filtered = co2_df.head(5)
    return filtered

def main():
    st.title("Dashboard Big Data - Clients et Véhicules")

    clients = load_clients()
    co2 = load_co2()

    st.sidebar.header("Filtres Clients")
    clusters = clients["cluster"].unique()
    selected_clusters = st.sidebar.multiselect("Choisir clusters", options=clusters, default=clusters)

    filtered_clients = clients[clients["cluster"].isin(selected_clusters)]
    st.subheader("Clients Clusterisés")
    st.dataframe(filtered_clients)

    st.subheader("Véhicules disponibles")
    st.dataframe(co2)

    st.markdown("""
    ---
    _Les véhicules affichés ici sont tous disponibles. Ci-dessous, vous pouvez voir des recommandations personnalisées selon le cluster client._
    """)

    for cluster in selected_clusters:
        st.write(f"### Recommandations pour le cluster {cluster}")
        recs = recommend_vehicles(cluster, co2)
        st.dataframe(recs)

if __name__ == "__main__":
    main()

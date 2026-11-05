"""La base de données du projet (SQLite) : fusion, vues et vues matérialisées.

Consigne : « faire la fusion, créer des vues pour exploiter le potentiel réel de la base de données,
générer le script qui crée ces tables virtuelles et les stocke avant de les manipuler ».

Étapes (python src/base_de_donnees.py) :
1. créer les tables (sql/01_tables.sql) ;
2. les remplir : vols (fichier nettoyé), compagnies, destinations, calendrier, événements, météo (fusion) ;
3. créer les vues, une par question (sql/02_vues.sql) ;
4. matérialiser chaque vue : CREATE TABLE mv_<nom> AS SELECT * FROM v_<nom> (les résultats sont stockés) ;
5. vérifier que la base donne les mêmes chiffres que Python (tests/test_base_de_donnees.py).

SQLite est intégré à Python : la base est un seul fichier (data/atlanta_vols.sqlite), sans serveur.
"""

import json
import sqlite3
import time

import pandas as pd

from config import RACINE
from contexte import COMPAGNIES_CONTEXTE, EVENEMENTS
from preprocessing import NOMS_JOURS, SAISONS, charger_vols_propres

FICHIER_BASE = RACINE / "data/atlanta_vols.sqlite"
DOSSIER_SQL = RACINE / "sql"
DOSSIER_EXTERNE = RACINE / "data/external"

# Jours fériés fédéraux américains de la période (date observée quand le jour tombe un week-end)
JOURS_FERIES = {
    "2021-09-06": "Labor Day", "2021-10-11": "Columbus Day", "2021-11-11": "Veterans Day",
    "2021-11-25": "Thanksgiving", "2021-12-24": "Noël (observé)", "2021-12-25": "Noël",
    "2021-12-31": "Nouvel An (observé)", "2022-01-01": "Nouvel An", "2022-01-17": "Martin Luther King Day",
    "2022-02-21": "Presidents' Day", "2022-05-30": "Memorial Day", "2022-06-19": "Juneteenth",
    "2022-06-20": "Juneteenth (observé)", "2022-07-04": "Independence Day",
}


def table_compagnies(vols):
    lignes = []
    for compagnie, groupe in vols.groupby("compagnie"):
        marque, base, remarque = COMPAGNIES_CONTEXTE[compagnie]
        lignes.append({"compagnie": compagnie, "nom_bts": groupe["Airline"].iloc[0],
                       "code_iata": groupe["IATA_Code_Operating_Airline"].iloc[0],
                       "type_compagnie": groupe["type_compagnie"].iloc[0], "vend_sous_la_marque": marque,
                       "base_principale": base, "remarque": remarque})
    return pd.DataFrame(lignes)


def table_destinations(vols):
    return (vols.groupby("Dest").agg(ville=("DestCityName", "first"), etat=("DestStateName", "first"),
                                     distance_miles=("Distance", "median"))
            .reset_index().rename(columns={"Dest": "dest"}))


def table_calendrier(vols):
    dates = pd.date_range(vols["FlightDate"].min(), vols["FlightDate"].max(), freq="D")
    return pd.DataFrame({
        "date_vol": dates.strftime("%Y-%m-%d"),
        "jour_semaine": [NOMS_JOURS[d.isoweekday()] for d in dates],
        "num_jour_semaine": [d.isoweekday() for d in dates],
        "annee_mois": dates.strftime("%Y-%m"),
        "saison": [SAISONS[d.month] for d in dates],
        "jour_ferie": [JOURS_FERIES.get(d.strftime("%Y-%m-%d")) for d in dates],
    })


def table_evenements():
    lignes = []
    for numero, ev in enumerate(EVENEMENTS, start=1):
        for jour in pd.date_range(ev["debut"], ev["fin"], freq="D"):
            lignes.append({"date_vol": jour.strftime("%Y-%m-%d"), "numero": numero, "nom": ev["nom"],
                           "compagnies_concernees": "toutes" if ev["compagnies"] is None else ", ".join(ev["compagnies"]),
                           "source": ev["source"]})
    return pd.DataFrame(lignes)


def tables_meteo():
    """Météo d'Atlanta (Open-Meteo, archive historique), téléchargée dans data/external/."""
    jour = pd.DataFrame(json.loads((DOSSIER_EXTERNE / "meteo_atlanta_jour.json").read_text(encoding="utf-8"))["daily"])
    jour = jour.rename(columns={"time": "date_vol", "weather_code": "code_meteo", "temperature_2m_max": "temperature_max",
                                "temperature_2m_min": "temperature_min", "precipitation_sum": "pluie_mm",
                                "snowfall_sum": "neige_cm", "precipitation_hours": "heures_de_pluie",
                                "wind_speed_10m_max": "vent_max_kmh", "wind_gusts_10m_max": "rafales_max_kmh"})
    jour = jour.drop(columns=["rain_sum"])
    heure = pd.DataFrame(json.loads((DOSSIER_EXTERNE / "meteo_atlanta_heure.json").read_text(encoding="utf-8"))["hourly"])
    temps = pd.to_datetime(heure.pop("time"))
    heure.insert(0, "date_vol", temps.dt.strftime("%Y-%m-%d"))
    heure.insert(1, "heure", temps.dt.hour)
    heure = heure.rename(columns={"weather_code": "code_meteo", "precipitation": "pluie_mm", "snowfall": "neige_cm",
                                  "wind_speed_10m": "vent_kmh", "wind_gusts_10m": "rafales_kmh",
                                  "visibility": "visibilite_m", "cloud_cover": "nuages_pct"})
    return jour, heure


def table_vols(vols):
    """Les vols, avec des types simples pour SQLite (dates en texte, booléens en 0/1)."""
    vols = vols.copy()
    vols["FlightDate"] = vols["FlightDate"].dt.strftime("%Y-%m-%d")
    for colonne in vols.select_dtypes(include="bool").columns:
        vols[colonne] = vols[colonne].astype(int)
    for colonne in vols.select_dtypes(include="category").columns:
        vols[colonne] = vols[colonne].astype(str)
    return vols


def creer_base(fichier=FICHIER_BASE, vols=None, verbeux=True):
    """Crée (ou recrée) la base complète : tables, fusion, vues, vues matérialisées."""
    debut = time.time()
    vols = charger_vols_propres() if vols is None else vols
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.unlink(missing_ok=True)
    with sqlite3.connect(fichier) as base:
        base.executescript((DOSSIER_SQL / "01_tables.sql").read_text(encoding="utf-8"))
        meteo_jour, meteo_heure = tables_meteo()
        for nom, table in [("compagnies", table_compagnies(vols)), ("destinations", table_destinations(vols)),
                           ("calendrier", table_calendrier(vols)), ("evenements", table_evenements()),
                           ("meteo_jour", meteo_jour), ("meteo_heure", meteo_heure)]:
            table.to_sql(nom, base, if_exists="append", index=False)
        table_vols(vols).to_sql("vols", base, index=False, chunksize=20000)
        for index in ["FlightDate", "compagnie", "Dest", "Tail_Number, FlightDate, CRSDepTime"]:
            nom_index = "idx_vols_" + index.split(",")[0].strip().lower()
            base.execute(f"CREATE INDEX {nom_index} ON vols ({index})")
        base.executescript((DOSSIER_SQL / "02_vues.sql").read_text(encoding="utf-8"))
        vues = [ligne[0] for ligne in base.execute("SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name")]
        for vue in vues:  # matérialisation : on stocke le résultat de chaque vue avant de le manipuler
            base.execute(f"DROP TABLE IF EXISTS mv_{vue[2:]}")
            base.execute(f"CREATE TABLE mv_{vue[2:]} AS SELECT * FROM {vue}")
        base.commit()
    if verbeux:
        print(f"Base créée : {fichier.name} ({fichier.stat().st_size / 1e6:.0f} Mo), {len(vues)} vues matérialisées, "
              f"{time.time() - debut:.0f} s")
        print("   vues :", ", ".join(vues))
    return fichier


def requete(sql, fichier=FICHIER_BASE, parametres=()):
    """Exécute une requête SQL sur la base et renvoie un DataFrame."""
    with sqlite3.connect(fichier) as base:
        return pd.read_sql_query(sql, base, params=parametres)


def lire(nom, fichier=FICHIER_BASE):
    """Lit une vue matérialisée (ex. lire('ponctualite_compagnie') lit mv_ponctualite_compagnie)."""
    return requete(f"SELECT * FROM mv_{nom}", fichier)


if __name__ == "__main__":
    creer_base()

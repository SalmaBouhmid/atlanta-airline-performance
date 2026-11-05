"""Étape 1 du pipeline : extraction du périmètre d'étude.

Les fichiers Kaggle complets contiennent tous les vols intérieurs des États-Unis
(environ 6 millions par an). On ne garde que les vols au départ d'Atlanta (ATL)
entre DEBUT_ETUDE et FIN_ETUDE (voir config.py) : c'est notre population d'étude.

Les deux années n'ont pas le même format :
- 2022 : fichier Parquet, lu directement avec un filtre (seules les lignes utiles sont chargées) ;
- 2021 : fichier CSV de 2,2 Go dans un zip, lu par morceaux de 500 000 lignes
  pour ne jamais charger tout le fichier en mémoire.
"""

import pandas as pd

from config import (AEROPORT, DEBUT_ETUDE, FICHIER_ETUDE, FICHIERS_KAGGLE, FIN_ETUDE,
                    RACINE)  # noqa: F401  (RACINE est réutilisé par les autres modules)

TAILLE_MORCEAU = 500_000  # lignes lues à la fois dans le CSV


def mois_par_annee(debut, fin):
    """Liste les mois nécessaires pour chaque année : {2021: [8, ..., 12], 2022: [1, ..., 7]}."""
    mois = pd.period_range(debut, fin, freq="M")
    return {annee: sorted(mois[mois.year == annee].month) for annee in sorted(set(mois.year))}


def lire_annee(annee, mois, aeroport=AEROPORT):
    """Lit les vols au départ de l'aéroport pour une année et une liste de mois."""
    fichier = FICHIERS_KAGGLE[annee]
    if fichier.suffix == ".parquet":
        return pd.read_parquet(fichier, filters=[("Origin", "==", aeroport), ("Month", "in", mois)])

    # CSV (éventuellement zippé) : on filtre chaque morceau avant de le garder
    morceaux = []
    for morceau in pd.read_csv(fichier, chunksize=TAILLE_MORCEAU, low_memory=False):
        garder = (morceau["Origin"] == aeroport) & morceau["Month"].isin(mois)
        morceaux.append(morceau[garder])
    vols = pd.concat(morceaux)
    vols["FlightDate"] = pd.to_datetime(vols["FlightDate"])
    return vols


def extraire_vols(debut=DEBUT_ETUDE, fin=FIN_ETUDE, aeroport=AEROPORT):
    """Assemble les vols au départ de l'aéroport entre deux dates, toutes années confondues.

    Les vols sont triés (date, heure prévue, compagnie, numéro de vol) pour que
    l'ordre des lignes soit toujours le même, quel que soit le format du fichier source.
    """
    annees = [lire_annee(annee, mois, aeroport) for annee, mois in mois_par_annee(debut, fin).items()]
    vols = pd.concat(annees, ignore_index=True)
    vols = vols[vols["FlightDate"].between(debut, fin)]
    vols = vols.sort_values(["FlightDate", "CRSDepTime", "Airline", "Flight_Number_Operating_Airline"])
    return vols.reset_index(drop=True)


def extraire_vols_atlanta():
    """Extrait la population d'étude et l'enregistre (à lancer une seule fois)."""
    vols = extraire_vols()
    vols.to_parquet(FICHIER_ETUDE, index=False)
    return vols


def charger_vols_atlanta():
    """Charge la population d'étude : les vols au départ d'Atlanta d'août 2021 à juillet 2022."""
    return pd.read_parquet(FICHIER_ETUDE)


if __name__ == "__main__":
    vols = extraire_vols_atlanta()
    print(f"{len(vols)} lignes, {vols.shape[1]} colonnes enregistrées dans {FICHIER_ETUDE}")
    print(f"Du {vols['FlightDate'].min():%d/%m/%Y} au {vols['FlightDate'].max():%d/%m/%Y}")

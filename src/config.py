"""Paramètres du projet, réunis à un seul endroit.

Pour changer la période d'étude, il suffit de modifier DEBUT_ETUDE et FIN_ETUDE,
puis de relancer le pipeline : toutes les analyses sont recalculées sur la nouvelle période.
"""

from pathlib import Path

# Dossier racine du projet : les chemins marchent quel que soit le dossier d'où on lance le code
RACINE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Périmètre de l'étude
# ---------------------------------------------------------------------------
AEROPORT = "ATL"  # Atlanta, l'aéroport le plus fréquenté du monde

# Population P12 : 12 mois consécutifs complets, les plus récents disponibles sur Kaggle
# (le fichier 2022 s'arrête en juillet). Elle contient l'été 2022 étudié dans la version 1.
DEBUT_ETUDE = "2021-08-01"
FIN_ETUDE = "2022-07-31"

# Période de la version 1 du projet (juin-juillet 2022), gardée comme « zoom été »
# et pour vérifier que le nouveau chargement redonne exactement les mêmes vols.
DEBUT_ETE_2022 = "2022-06-01"
FIN_ETE_2022 = "2022-07-31"

# ---------------------------------------------------------------------------
# Fichiers
# ---------------------------------------------------------------------------
# Fichiers Kaggle complets, un par année (téléchargés à la main, ignorés par git).
# Le format diffère selon l'année : Kaggle refusait le téléchargement du Parquet 2021,
# on lit donc le CSV 2021 directement dans son zip.
FICHIERS_KAGGLE = {
    2021: RACINE / "data/raw/Combined_Flights_2021.csv.zip",
    2022: RACINE / "data/raw/Combined_Flights_2022.parquet",
}

# Extraction (vols au départ d'Atlanta sur la période d'étude, toutes les colonnes)
FICHIER_ETUDE = RACINE / "data/raw/vols_atlanta_2021-08_2022-07.parquet"
# Version nettoyée et enrichie, utilisée par le notebook et le dashboard
FICHIER_PROPRE = RACINE / "data/processed/vols_atlanta_2021-08_2022-07_propre.parquet"
# Les fichiers de la version 1 (été 2022, CSV) restent disponibles dans git : tag « v1-ete-2022 ».

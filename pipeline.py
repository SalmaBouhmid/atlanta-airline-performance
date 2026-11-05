"""Lance toutes les étapes du projet dans l'ordre, en une seule commande :

    python pipeline.py              # nettoyage + résultats (utilise l'extraction existante)
    python pipeline.py --extraire   # refait aussi l'extraction depuis les fichiers Kaggle complets

Étapes :
    1. Extraction   : vols au départ d'Atlanta sur la période d'étude (data/raw/)
    2-4. Qualité, nettoyage et préparation : fichier propre enrichi (data/processed/)
    5. Base de données : SQLite, fusion (compagnies, destinations, calendrier, événements, météo) et vues (sql/)
    6-9. Statistiques, comparaisons, modèle et analyses : tous les chiffres dans reports/resultats.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from base_de_donnees import creer_base  # noqa: E402
from config import FICHIER_ETUDE, FICHIER_PROPRE  # noqa: E402
from data_loading import charger_vols_atlanta, extraire_vols_atlanta  # noqa: E402
from preprocessing import nettoyer_vols  # noqa: E402
from results import FICHIER_RESULTATS, calculer_resultats, enregistrer_resultats  # noqa: E402


def main(extraire=False):
    if extraire or not FICHIER_ETUDE.exists():
        print("1. Extraction depuis les fichiers Kaggle (environ 2 minutes)...")
        vols_bruts = extraire_vols_atlanta()
    else:
        print("1. Extraction déjà faite : lecture de", FICHIER_ETUDE.name)
        vols_bruts = charger_vols_atlanta()
    print(f"   {len(vols_bruts)} vols, {vols_bruts.shape[1]} colonnes")

    print("2-4. Qualité, nettoyage et préparation...")
    vols = nettoyer_vols(vols_bruts)
    FICHIER_PROPRE.parent.mkdir(parents=True, exist_ok=True)
    vols.to_parquet(FICHIER_PROPRE, index=False)
    print(f"   {vols.shape[1]} colonnes enregistrées dans {FICHIER_PROPRE.name}")

    print("5. Base de données SQLite : tables, fusion, vues matérialisées...")
    creer_base(vols=vols)

    print("6-9. Statistiques, comparaisons, modèle et analyses (environ 2 minutes)...")
    enregistrer_resultats(calculer_resultats(vols))
    print(f"   Résultats enregistrés dans {FICHIER_RESULTATS.relative_to(FICHIER_RESULTATS.parents[1])}")


if __name__ == "__main__":
    main(extraire="--extraire" in sys.argv)

"""Tests de la base SQLite : la base (vues SQL) et Python doivent donner exactement les mêmes chiffres."""

import pandas as pd
import pytest

from base_de_donnees import FICHIER_BASE, creer_base, lire, requete
from results import charger_resultats


@pytest.fixture(scope="module")
def base(vols):
    """La base du projet ; elle est construite si elle n'existe pas encore (environ 30 secondes)."""
    if not FICHIER_BASE.exists():
        creer_base(vols=vols, verbeux=False)
    return FICHIER_BASE


@pytest.fixture(scope="module")
def resultats():
    return charger_resultats()


def test_toutes_les_vues_sont_materialisees(base):
    vues = set(requete("SELECT name FROM sqlite_master WHERE type = 'view'")["name"])
    tables = set(requete("SELECT name FROM sqlite_master WHERE type = 'table'")["name"])
    assert len(vues) >= 10
    assert {f"mv_{v[2:]}" for v in vues} <= tables


def test_fusion_ne_perd_aucun_vol(base, vols):
    assert requete("SELECT COUNT(*) AS n FROM vols")["n"].iloc[0] == len(vols)
    assert requete("SELECT COUNT(*) AS n FROM vols v JOIN compagnies c USING (compagnie)")["n"].iloc[0] == len(vols)
    assert requete("SELECT COUNT(*) AS n FROM vols v JOIN calendrier c ON c.date_vol = v.FlightDate")["n"].iloc[0] == len(vols)


def test_sql_et_python_donnent_les_memes_chiffres_par_compagnie(base, resultats):
    sql = lire("ponctualite_compagnie").set_index("compagnie")
    python = pd.DataFrame(resultats["compagnies"]).T
    for colonne in ["nb_vols", "pct_retard_ou_annule", "pct_retard", "pct_annules", "pct_grand_retard"]:
        assert (sql[colonne].astype(float) - python[colonne].astype(float)).abs().max() < 0.05, colonne


def test_standardisation_sql_identique(base, resultats):
    sql = lire("composition_heure").set_index("compagnie")
    python = pd.DataFrame(resultats["composition_tranche_horaire"]).T
    assert (sql["pct_retard_standardise"] - python["taux_standardise_pct"]).abs().max() < 0.05


def test_domino_sql_identique(base, resultats):
    domino = requete("SELECT AVG(CASE WHEN precedent_en_retard = 0 THEN DepDel15 END) * 100 AS a, "
                     "AVG(CASE WHEN precedent_en_retard = 1 THEN DepDel15 END) * 100 AS b FROM mv_domino")
    assert domino["a"].iloc[0] == pytest.approx(resultats["indicateurs"]["domino_pct_si_precedent_a_l_heure"], abs=0.05)
    assert domino["b"].iloc[0] == pytest.approx(resultats["indicateurs"]["domino_pct_si_precedent_en_retard"], abs=0.05)


def test_vue_jour_fusionne_meteo_et_evenements(base):
    jour = lire("jour")
    assert len(jour) == 365
    assert jour["pluie_mm"].notna().all()
    assert jour.loc[jour["date_vol"] == "2022-01-16", "neige_cm"].iloc[0] > 1   # tempête Izzy
    assert jour["evenements"].notna().sum() >= 10

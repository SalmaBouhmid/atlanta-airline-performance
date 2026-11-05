"""Tests de l'approfondissement (jour par jour, liens bivariés, tests par paires) et du contexte réel."""

import pandas as pd
import pytest

import approfondissement as a
from contexte import EVENEMENTS, FAITS_METIER, verifier_evenements


def test_association_nulle_et_totale():
    """V de Cramér : 0 si les deux variables sont indépendantes, 1 si l'une détermine l'autre."""
    independantes = pd.DataFrame([[50, 50], [50, 50]])
    liees = pd.DataFrame([[100, 0], [0, 100]])
    assert a.association_categorielle(independantes)["v_cramer"] == pytest.approx(0, abs=1e-9)
    assert a.association_categorielle(liees)["v_cramer"] == pytest.approx(1, abs=0.02)


def test_paires_bonferroni_sur_un_exemple():
    """A et B identiques : non distinctes ; C très différente : distincte des deux."""
    vols = pd.DataFrame({"compagnie": ["A"] * 1000 + ["B"] * 1000 + ["C"] * 1000,
                         "retard_ou_annule": [0, 1] * 500 + [0, 1] * 500 + [1] * 900 + [0] * 100})
    paires, seuil = a.paires_non_distinctes(vols)
    assert seuil == pytest.approx(0.05 / 3)
    resultat = {(p.compagnie_a, p.compagnie_b): p.distinctes for p in paires.itertuples()}
    assert resultat[("A", "B")] is False
    assert resultat[("A", "C")] and resultat[("B", "C")]


def test_serie_journaliere_couvre_toute_l_annee(vols):
    jour = a.serie_journaliere(vols)
    assert len(jour) == 365
    assert jour["nb_vols"].sum() == len(vols)
    assert jour["pct_retard"].between(0, 100).all()


def test_vue_par_jour_plus_variable_que_par_mois(vols):
    """La moyenne du mois efface les crises courtes."""
    resume = a.resume_journalier(vols)
    assert resume["ecart_type_jours"] > 1.5 * resume["ecart_type_mois"]
    assert resume["jour_max"]["pct"] > 2 * resume["mediane_jours"]


def test_conclusion_types_robuste_aux_versions(vols):
    """« Bas coût > majeures > régionales » tient dans les 3 versions de la classification."""
    assert a.sensibilite_types(vols)["ordre_conserve"].all()


def test_domino_tient_a_chaque_heure(vols):
    tableau = a.domino_par_tranche_et_rang(vols)["domino_par_tranche"]
    taux, effectifs = tableau["mean"], tableau["size"]
    fiables = effectifs[1.0] >= 30
    assert (taux.loc[fiables, 1.0] > taux.loc[fiables, 0.0] + 0.15).all()


def test_aucune_erreur_de_coherence_arrivee(vols):
    assert a.valeurs_extremes(vols)["nb_arrivee_incoherente"] == 0


def test_chaque_evenement_reel_est_visible_dans_les_donnees(vols):
    """Règle du projet : un événement documenté n'est retenu que s'il se voit dans nos données."""
    verification = verifier_evenements(vols)
    assert len(verification) == len(EVENEMENTS)
    assert verification["visible"].all()


def test_chaque_fait_et_evenement_a_une_source():
    for element in FAITS_METIER + EVENEMENTS:
        assert element["source"].startswith("https://")


def test_les_vols_d_un_meme_jour_se_ressemblent_plus_que_ceux_d_une_compagnie(vols):
    from approfondissement import decomposition_variance
    d = decomposition_variance(vols)
    assert d["FlightDate"]["part_inter_pct"] > d["compagnie"]["part_inter_pct"]
    assert d["grappes_jours"]["effet_de_plan"] > 10          # les grappes de jours perdent beaucoup de précision
    assert d["grappes_jours"]["taille_efficace"] < 500

"""Tests des 6 méthodes d'échantillonnage et de la mesure de représentativité."""

import pytest

from sampling import (METHODES, TAILLE_ECHANTILLON, classement_des_methodes, echantillon_bootstrap,
                      echantillon_par_grappes, echantillon_stratifie, echantillon_stratifie_non_proportionnel,
                      echantillon_systematique, erreurs_par_rapport_a_la_population, pourcentage_retard_pondere,
                      score_representativite, stabilite_des_methodes)


def test_six_methodes_du_cours():
    assert list(METHODES) == ["Aléatoire simple", "Systématique", "Stratifié par compagnie", "Par grappes (jours)",
                              "Temporel (par jour)", "Répété (bootstrap)"]


@pytest.mark.parametrize("nom", list(METHODES))
def test_chaque_methode_est_reproductible(vols, nom):
    """Même graine → exactement le même échantillon."""
    methode = METHODES[nom]
    assert methode(vols, graine=7).index.equals(methode(vols, graine=7).index)


@pytest.mark.parametrize("nom", ["Aléatoire simple", "Systématique", "Répété (bootstrap)"])
def test_taille_demandee_respectee(vols, nom):
    assert len(METHODES[nom](vols)) == TAILLE_ECHANTILLON


def test_indices_systematiques_valides(vols):
    echantillon = echantillon_systematique(vols)
    assert echantillon.index.isin(vols.index).all()
    assert echantillon.index.is_unique
    assert echantillon["FlightDate"].nunique() == vols["FlightDate"].nunique()  # toute la période couverte


def test_stratifie_proportionnel_conserve_les_proportions(vols):
    parts_population = vols["compagnie"].value_counts(normalize=True)
    parts_echantillon = echantillon_stratifie(vols)["compagnie"].value_counts(normalize=True)
    ecarts = (parts_echantillon - parts_population).abs()
    assert ecarts.max() < 0.005  # moins de 0,5 point d'écart pour chaque compagnie


def test_stratifie_par_compagnie_meme_nombre_par_groupe(vols):
    assert METHODES["Stratifié par compagnie"] is echantillon_stratifie_non_proportionnel
    comptes = echantillon_stratifie_non_proportionnel(vols)["compagnie"].value_counts()
    assert comptes.nunique() == 1
    assert comptes.iloc[0] == TAILLE_ECHANTILLON // vols["compagnie"].nunique()


@pytest.mark.parametrize("nom", list(METHODES))
def test_grande_taille_ne_plante_pas(vols, nom):
    """Même avec 15 000 vols demandés (plus que certaines compagnies n'en ont), aucune méthode ne plante."""
    echantillon = METHODES[nom](vols, taille=15000)
    assert 0 < len(echantillon) <= 15500


def test_repondération_corrige_le_stratifie_par_compagnie(vols):
    """Repondéré, le % de retard du stratifié par compagnie (à part égale) est proche de la population."""
    echantillon = echantillon_stratifie_non_proportionnel(vols)
    pourcentage_population = vols[~vols["Cancelled"]]["DepDel15"].mean() * 100
    assert abs(pourcentage_retard_pondere(echantillon, vols) - pourcentage_population) < 2


def test_bootstrap_autorise_les_repetitions(vols):
    echantillon = echantillon_bootstrap(vols)
    assert not echantillon.index.is_unique


def test_grappes_existent_et_sont_completes(vols):
    echantillon = echantillon_par_grappes(vols)
    jours = echantillon["FlightDate"].unique()
    assert set(jours) <= set(vols["FlightDate"].unique())
    for jour in jours:  # une grappe = tous les vols du jour
        assert (echantillon["FlightDate"] == jour).sum() == (vols["FlightDate"] == jour).sum()


def test_erreurs_et_score(vols):
    echantillon = echantillon_stratifie(vols)
    erreurs = erreurs_par_rapport_a_la_population(vols, echantillon)
    assert len(erreurs) == 6
    assert (erreurs["erreur absolue"] >= 0).all()
    assert 0 <= score_representativite(vols, echantillon) <= 100
    assert score_representativite(vols, vols) == 100  # la population est parfaitement représentative d'elle-même


def test_classement_met_les_methodes_biaisees_en_dernier(vols):
    classement = classement_des_methodes(stabilite_des_methodes(vols, nb_tirages=5))
    assert list(classement["rang"]) == list(range(1, len(METHODES) + 1))
    assert classement.index[-1] == "Stratifié par compagnie"  # naïf : Delta passe de 64 % à 7 %

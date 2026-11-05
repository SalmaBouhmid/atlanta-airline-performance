"""Tests de l'étude d'échantillonnage (E1 à E7)."""

import numpy as np
import pytest

from sampling import poids_de_sondage
from sampling_study import (CONCLUSIONS, centile_pondere, distance_ks, etude_methodes, mesurer_echantillon,
                            references_population, synthese)


@pytest.fixture(scope="module")
def reference(vols):
    return references_population(vols)


def test_distance_ks():
    """Deux distributions identiques : 0. Décalées : écart positif."""
    population = np.sort(np.arange(100, dtype=float))
    grille = np.unique(population)
    repartition = np.searchsorted(population, grille, side="right") / len(population)
    poids = np.ones(100)
    assert distance_ks(population.copy(), poids, grille, repartition) == pytest.approx(0)
    assert distance_ks(population + 50, poids, grille, repartition) == pytest.approx(0.5, abs=0.01)


def test_centile_pondere():
    """Un poids double revient à compter la valeur deux fois."""
    valeurs = np.array([1.0, 2.0, 3.0])
    assert centile_pondere(valeurs, np.ones(3), 0.5) == 2
    assert centile_pondere(valeurs, np.array([1.0, 1.0, 4.0]), 0.5) == 3


def test_poids_de_sondage_proportionnel_presque_constants(vols):
    from sampling import echantillon_stratifie
    poids = poids_de_sondage(echantillon_stratifie(vols), vols)
    assert poids.max() / poids.min() < 1.1


def test_population_parfaitement_preservee(vols, reference):
    """Mesurée contre elle-même, la population n'a aucune erreur et garde toutes ses conclusions."""
    mesures = mesurer_echantillon(vols, reference)
    erreurs = [valeur for nom, valeur in mesures.items()
               if "erreur" in nom or "ecart" in nom or nom.startswith("E1_ks")]
    assert max(erreurs) == pytest.approx(0, abs=1e-9)
    assert all(mesures[c] for c in CONCLUSIONS)


def test_etude_methodes_produit_une_ligne_par_tirage(vols, reference):
    mesures = etude_methodes(vols, reference, taille=2000, nb_tirages=2,
                             methodes=["Aléatoire simple", "Stratifié par compagnie"])
    # aléatoire : 2 tirages ; stratifié par compagnie : 2 tirages × (naïf + repondéré)
    assert len(mesures) == 6
    resume = synthese(mesures)
    assert resume["pct_conclusions_conservees"].between(0, 100).all()


def test_repondération_corrige_le_pourcentage_global(vols, reference):
    """Naïf, le stratifié par compagnie se trompe sur le % global ; repondéré, beaucoup moins."""
    mesures = etude_methodes(vols, reference, taille=5000, nb_tirages=5,
                             methodes=["Stratifié par compagnie"])
    erreurs = mesures.groupby("estimateur")["E6_erreur_pct_retard_global"].mean()
    assert erreurs["repondéré"] < erreurs["naïf"]


def test_perdre_des_compagnies_ne_rend_pas_le_classement_plus_facile(vols, reference):
    """Piège corrigé : sans Mesa, on ne peut plus affirmer qu'Endeavor est la plus ponctuelle DE TOUTES."""
    sans_mesa = vols[vols["compagnie"] != "Mesa"]
    mesures = mesurer_echantillon(sans_mesa, reference)
    assert mesures["E6_nb_compagnies_comparables"] == 13
    assert mesures["C3"] is False


def test_conclusions_vraies_accepte_un_echantillon_sans_american(vols, reference):
    """Si American est absente de l'échantillon, C4 et C5 sont simplement « non conservées »."""
    sans_american = vols[vols["compagnie"] != "American"]
    mesures = mesurer_echantillon(sans_american, reference)
    assert mesures["C4"] is False and mesures["C5"] is False


def test_formule_taille_echantillon():
    """Cas d'école : p = 0,5, e = 0,05, z = 1,96, population infinie → n₀ = 384,16."""
    from sampling import marge_erreur, taille_echantillon
    assert taille_echantillon(10**12, 0.5, 0.05) == pytest.approx(384.16, abs=0.01)
    # la correction de population finie réduit n ; la marge d'erreur est la formule inverse
    n = taille_echantillon(319335, 0.186, 0.01)
    assert 5700 < n < 5720
    assert marge_erreur(n, 319335, 0.186) == pytest.approx(0.01, abs=1e-4)

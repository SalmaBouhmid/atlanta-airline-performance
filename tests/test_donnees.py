"""Tests du chargement et du nettoyage des données."""

import hashlib

import pandas as pd
import pytest

from config import (DEBUT_ETE_2022, DEBUT_ETUDE, FICHIER_ETUDE, FICHIERS_KAGGLE, FIN_ETE_2022,
                    FIN_ETUDE)
from data_loading import extraire_vols, mois_par_annee
from preprocessing import colonnes_constantes


def test_dataset_brut_respecte_la_consigne(vols_bruts):
    """Au moins 30 000 lignes et 50 colonnes (cahier des charges, section 2)."""
    assert vols_bruts.shape == (319335, 61)


def test_periode_complete_de_12_mois(vols_bruts):
    """La population couvre 12 mois consécutifs, tous les jours, sans trou."""
    dates = vols_bruts["FlightDate"]
    assert dates.min() == pd.Timestamp(DEBUT_ETUDE)
    assert dates.max() == pd.Timestamp(FIN_ETUDE)
    assert dates.dt.to_period("M").nunique() == 12
    assert dates.dt.date.nunique() == 365


def test_mois_par_annee():
    assert mois_par_annee("2021-08-01", "2022-07-31") == {2021: [8, 9, 10, 11, 12],
                                                         2022: [1, 2, 3, 4, 5, 6, 7]}


@pytest.mark.skipif(not FICHIERS_KAGGLE[2022].exists(), reason="fichier Kaggle 2022 absent")
def test_extraction_redonne_la_version_1_sur_l_ete_2022():
    """Non-régression : sur juin-juillet 2022, le nouveau chargement redonne les vols de la version 1.

    Version 1 du projet : 54 461 vols, dont 14 039 partis avec plus de 15 min de retard.
    """
    ete = extraire_vols(DEBUT_ETE_2022, FIN_ETE_2022)
    assert ete.shape == (54461, 61)
    assert ete["DepDel15"].sum() == 14039


def test_nettoyage_ne_modifie_pas_le_fichier_brut(vols_bruts):
    """Le fichier brut sur le disque est identique avant et après le nettoyage."""
    empreinte_avant = hashlib.md5(FICHIER_ETUDE.read_bytes()).hexdigest()
    from preprocessing import nettoyer_vols
    nettoyer_vols(vols_bruts)
    assert hashlib.md5(FICHIER_ETUDE.read_bytes()).hexdigest() == empreinte_avant


def test_nettoyage_garde_toutes_les_lignes(vols_bruts, vols_nettoyes_a_la_volee):
    """Aucun vol n'est supprimé : les annulés et les retards extrêmes sont gardés volontairement."""
    assert len(vols_nettoyes_a_la_volee) == len(vols_bruts)
    assert vols_nettoyes_a_la_volee["Cancelled"].sum() == vols_bruts["Cancelled"].sum()


def test_pas_de_doublons(vols):
    assert vols.duplicated().sum() == 0


def test_colonnes_constantes_supprimees(vols_bruts, vols):
    constantes = colonnes_constantes(vols_bruts)
    assert len(constantes) == 9  # les 9 colonnes qui décrivent l'aéroport de départ (toujours ATL)
    assert not set(constantes) & set(vols.columns)


def test_types_coherents(vols):
    assert pd.api.types.is_datetime64_any_dtype(vols["FlightDate"])
    assert vols["heure_depart_prevue"].between(0, 23).all()
    assert set(vols["statut"]) == {"À l'heure", "En retard", "Annulé", "Dévié"}
    assert vols["compagnie"].notna().all()


def test_valeurs_manquantes_seulement_pour_les_vols_annules(vols):
    """Un vol parti a toujours un retard au départ ; un vol annulé n'en a pas forcément."""
    partis = vols[~vols["Cancelled"]]
    assert partis["DepDelay"].notna().all()


def test_effet_domino_coherent(vols):
    """Le 1er départ de la journée d'un avion n'a pas de vol précédent ; les suivants en ont un (0 ou 1)."""
    premiers = vols[vols["rang_rotation"] == 1]
    assert premiers["vol_precedent_en_retard"].isna().all()
    suivants = vols[vols["rang_rotation"] > 1]["vol_precedent_en_retard"].dropna()
    assert set(suivants.unique()) <= {0.0, 1.0}


def test_type_compagnie_lu_dans_les_donnees(vols):
    """Une compagnie régionale vend ses vols sous la marque d'une autre (« _CODESHARE »)."""
    assert set(vols["type_compagnie"]) == {"Majeure", "Régionale", "Bas coût"}
    regionales = vols[vols["type_compagnie"] == "Régionale"]
    assert regionales["Operated_or_Branded_Code_Share_Partners"].str.endswith("_CODESHARE").all()
    assert (vols.loc[vols["compagnie"] == "Endeavor (Delta Connection)", "reseau_commercial"] == "DL").all()


def test_classe_retard_coherente(vols):
    """Chaque vol a exactement une classe ; les annulés sont dans « Annulé »."""
    assert vols["classe_retard"].notna().all()
    assert (vols.loc[vols["Cancelled"], "classe_retard"] == "Annulé").all()
    grands = vols[vols["classe_retard"] == "Grand retard (> 45 min)"]
    assert (grands["DepDelayMinutes"] > 45).all()


def test_retard_ou_annule_corrige_le_biais_du_survivant(vols):
    """Vol perturbé = parti avec plus de 15 min de retard, ou annulé.

    Attention : 64 vols annulés ont quand même quitté la porte (puis ont été annulés),
    et certains ont DepDel15 = 1 : on ne doit pas les compter deux fois.
    """
    partis_en_retard = (vols["DepDel15"] == 1) & ~vols["Cancelled"]
    assert vols["retard_ou_annule"].sum() == partis_en_retard.sum() + vols["Cancelled"].sum()


def test_nettoyage_reproductible(vols, vols_nettoyes_a_la_volee):
    """Le fichier nettoyé enregistré correspond au nettoyage recalculé."""
    assert list(vols.columns) == list(vols_nettoyes_a_la_volee.columns)
    assert vols["DepDel15"].sum() == vols_nettoyes_a_la_volee["DepDel15"].sum()

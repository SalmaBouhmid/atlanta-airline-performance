"""Configuration commune des tests : accès aux modules de src/ et données chargées une seule fois."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loading import charger_vols_atlanta  # noqa: E402
from preprocessing import charger_vols_propres, nettoyer_vols  # noqa: E402


@pytest.fixture(scope="session")
def vols_bruts():
    """Le dataset brut (319 335 vols × 61 colonnes, août 2021 - juillet 2022)."""
    return charger_vols_atlanta()


@pytest.fixture(scope="session")
def vols():
    """Le dataset nettoyé, utilisé par l'analyse et le dashboard."""
    return charger_vols_propres()


@pytest.fixture(scope="session")
def vols_nettoyes_a_la_volee(vols_bruts):
    """Le nettoyage recalculé pendant le test (pour vérifier qu'il est reproductible)."""
    return nettoyer_vols(vols_bruts)

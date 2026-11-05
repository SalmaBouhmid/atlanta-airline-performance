"""Tests des calculs d'analyse et du démarrage du dashboard."""

from pathlib import Path

import pytest

from analysis import effet_domino, indicateurs_cles, resume_qualite, tableau_croise_retard, taux_de_retard_par
from config import DEBUT_ETE_2022, FIN_ETE_2022

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


def test_pourcentages_entre_0_et_100(vols):
    for colonne in ["heure_depart_prevue", "compagnie", "jour_semaine", "Dest"]:
        tableau = taux_de_retard_par(vols, colonne)
        assert not tableau.empty
        assert tableau["pct_retard"].between(0, 100).all()


def test_proportions_totalisent_100(vols):
    assert vols["statut"].value_counts(normalize=True).sum() == pytest.approx(1)
    assert vols["compagnie"].value_counts(normalize=True).sum() == pytest.approx(1)


def test_indicateurs_cles_non_vides(vols):
    kpi = indicateurs_cles(vols)
    assert kpi["nb_vols"] == 319335
    assert kpi["pct_retard"] == pytest.approx(18.6, abs=0.1)
    assert kpi["retard_moyen_des_vols_en_retard"] > 15  # un vol « en retard » a plus de 15 min de retard


def test_effet_domino_sur_12_mois(vols):
    """Sur 12 mois : 40,5 % de retard si le vol précédent était en retard, contre 15,4 %."""
    domino = effet_domino(vols)["pct_retard"]
    assert domino["Vol précédent à l'heure"] == pytest.approx(15.4, abs=0.1)
    assert domino["Vol précédent en retard"] == pytest.approx(40.5, abs=0.1)


def test_zoom_ete_2022_redonne_la_version_1(vols):
    """Non-régression : le sous-ensemble juin-juillet 2022 redonne les résultats de la version 1."""
    ete = vols[vols["FlightDate"].between(DEBUT_ETE_2022, FIN_ETE_2022)]
    assert indicateurs_cles(ete)["nb_vols"] == 54461
    assert indicateurs_cles(ete)["pct_retard"] == pytest.approx(26.2, abs=0.1)
    domino = effet_domino(ete)["pct_retard"]
    assert domino["Vol précédent à l'heure"] == pytest.approx(24.1, abs=0.1)
    assert domino["Vol précédent en retard"] == pytest.approx(50.0, abs=0.1)


def test_tableau_croise_non_vide(vols):
    carte = tableau_croise_retard(vols, "jour_semaine", "heure_depart_prevue")
    assert carte.shape[0] == 7
    assert carte.stack().between(0, 100).all()


def test_resume_qualite(vols_bruts):
    resume = resume_qualite(vols_bruts)
    assert resume["Nombre de lignes"] == 319335
    assert resume["Lignes en double"] == 0


def test_dashboard_demarre_sans_erreur():
    """Le dashboard s'ouvre sur la page d'accueil, sans exception."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()
    assert not app.exception
    assert any("319 335" in m.value for m in app.markdown)       # le bandeau d'accueil


def test_chaque_chapitre_s_affiche_sans_erreur():
    """Les 10 pages (accueil + 9 parties) s'exécutent sans exception."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()
    chapitres = app.radio(key="chapitre").options
    assert len(chapitres) == 10
    for chapitre in chapitres:
        app.radio(key="chapitre").set_value(chapitre).run()
        assert not app.exception, chapitre


def test_bouton_suivant_avance_d_un_chapitre():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()
    app.button[0].click().run()
    assert app.radio(key="chapitre").value.startswith("1 ")


def test_laboratoire_d_echantillonnage_reagit_a_la_methode():
    """Changer la méthode dans le laboratoire change réellement le résultat (compagnies comparables)."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()
    app.radio(key="chapitre").set_value("4 · Échantillonnage").run()
    methode = next(s for s in app.selectbox if s.label == "Méthode d'échantillonnage")
    comparables = lambda: next(m for m in app.metric if m.label == "Compagnies comparables").value  # noqa: E731
    assert methode.value == "Aléatoire simple"
    assert comparables() != "14 / 14"   # au hasard, les petites compagnies ont moins de 30 vols
    methode.set_value("Stratifié par compagnie").run()
    assert not app.exception
    assert comparables() == "14 / 14"   # stratifié : toutes les compagnies sont mesurables


def test_le_simulateur_retrouve_les_scenarios_du_modele():
    """Le risque calculé par le simulateur (page 8) est le même que celui du modèle (1,6 % … 81,5 %)."""
    import json
    import math
    resultats = json.loads((APP.parent.parent / "reports" / "resultats.json").read_text(encoding="utf-8"))
    modele = resultats["equipe"]["modele"]
    coef = modele["coefficients"]

    def risque(compagnie, tranche, saison, jour, precedent, pluie, journee):
        z = coef["Intercept"]
        for cle in [f"C(compagnie, Treatment('Delta'))[T.{compagnie}]", f"C(tranche, Treatment('Matin'))[T.{tranche}]",
                    f"C(saison, Treatment('Automne'))[T.{saison}]", f"C(jour, Treatment('Mardi'))[T.{jour}]",
                    f"C(precedent, Treatment('a_l_heure'))[T.{precedent}]"]:
            z += coef.get(cle, 0.0)
        z += coef["pluie_depart"] * pluie + coef["jour_pluvieux"] * journee
        return round(100 / (1 + math.exp(-z)), 1)

    attendus = list(modele["scenarios"].values())
    assert risque("Endeavor (Delta Connection)", "Matin", "Automne", "Mardi", "a_l_heure", 0, 0) == attendus[0]
    assert risque("JetBlue", "Soir", "Été", "Vendredi", "en_retard", 1, 1) == attendus[2]


def test_le_cube_accepte_d_autres_dimensions():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()
    app.radio(key="chapitre").set_value(app.radio(key="chapitre").options[1]).run()
    app.selectbox(key="cube_lignes").set_value("saison").run()
    app.selectbox(key="cube_colonnes").set_value("tranche_horaire").run()
    app.selectbox(key="cube_mesure").set_value("pct_annules").run()
    assert not app.exception

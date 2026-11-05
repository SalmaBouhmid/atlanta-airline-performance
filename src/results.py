"""Tous les chiffres cités dans le projet, calculés à un seul endroit.

Le rapport, le guide personnel et la présentation lisent reports/resultats.json :
aucun chiffre n'est recopié à la main, donc aucun document ne peut contredire les calculs.
Si la période d'étude change, on relance le pipeline et tous les documents suivent.
"""

import json

import pandas as pd

from analysis import effet_domino, frequences, indicateurs_cles, statistiques_descriptives, vols_partis
from comparison import (comparaison_brut_standardise, correlation_globale_et_par_groupe,
                        effet_domino_par_compagnie, profil_compagnies, taux_par_compagnie_et_periode)
from conclusions import conclusions_par_compagnie
from config import DEBUT_ETE_2022, DEBUT_ETUDE, FIN_ETE_2022, FIN_ETUDE, RACINE

FICHIER_RESULTATS = RACINE / "reports/resultats.json"

# Une conclusion est jugée « robuste » à une taille donnée si elle reste vraie
# dans au moins 95 % des tirages (même seuil que le niveau de confiance usuel).
SEUIL_ROBUSTESSE_PCT = 95

# Mesures où « plus petit = meilleur » (erreurs, écarts) ; les autres : « plus grand = meilleur »
MESURES_CLES = {
    "E1_ks_retard": "min",
    "E2_ecart_max_parts_compagnies": "min",
    "E2_nb_compagnies_moins_30_vols": "min",
    "E3_erreur_ecart_domino": "min",
    "E4_erreur_moyenne_compagnie_saison": "min",
    "E5_ecart_moyen_serie_mensuelle": "min",
    "E6_nb_compagnies_comparables": "max",
    "E6_erreur_max_compagnie": "min",
    "E6_spearman_classement_compagnies": "max",
    "E6_erreur_pct_retard_global": "min",
    "pct_conclusions_conservees": "max",
}


def resume_echantillonnage():
    """Synthèse de l'étude d'échantillonnage (si elle a été lancée : sampling_study.py)."""
    from sampling_study import CONCLUSIONS, DOSSIER_ECHANTILLONNAGE, synthese

    fichier_methodes = DOSSIER_ECHANTILLONNAGE / "mesures_methodes.parquet"
    fichier_tailles = DOSSIER_ECHANTILLONNAGE / "mesures_tailles.parquet"
    if not (fichier_methodes.exists() and fichier_tailles.exists()):
        return None

    par_methode = synthese(pd.read_parquet(fichier_methodes)).droplevel("taille")
    par_methode.index = [f"{m} ({e})" if e == "repondéré" else m for m, e in par_methode.index]
    meilleure = {
        mesure: (par_methode[mesure].idxmin() if sens == "min" else par_methode[mesure].idxmax())
        for mesure, sens in MESURES_CLES.items()
    }

    par_taille = synthese(pd.read_parquet(fichier_tailles))
    conclusions_par_taille = par_taille[[f"pct_tirages_{c}" for c in CONCLUSIONS]]
    taille_minimale = {}
    for (methode, estimateur), tableau in conclusions_par_taille.groupby(level=["methode", "estimateur"]):
        nom = f"{methode} ({estimateur})" if estimateur == "repondéré" else methode
        tableau = tableau.droplevel(["methode", "estimateur"])
        taille_minimale[nom] = {
            conclusion: (int(tableau.index[tableau[f"pct_tirages_{conclusion}"] >= SEUIL_ROBUSTESSE_PCT].min())
                         if (tableau[f"pct_tirages_{conclusion}"] >= SEUIL_ROBUSTESSE_PCT).any() else None)
            for conclusion in CONCLUSIONS
        }

    return {
        "conclusions": CONCLUSIONS,
        "synthese_5000": en_dictionnaire(par_methode),
        "meilleure_methode_par_mesure": meilleure,
        # clé « méthode | estimateur | taille », ex. « Aléatoire simple | naïf | 5000 »
        "par_taille": en_dictionnaire(par_taille[list(MESURES_CLES)].set_axis(
            [" | ".join(map(str, cle)) for cle in par_taille.index])),
        "taille_minimale_robuste": taille_minimale,
    }


def en_dictionnaire(tableau):
    """Convertit un tableau pandas en dictionnaire JSON (les cases vides deviennent null)."""
    tableau = tableau.copy()
    tableau.index = tableau.index.map(str)
    tableau.columns = tableau.columns.map(str)
    return json.loads(tableau.to_json(orient="index", force_ascii=False))


def resume_population(vols):
    """Ce qui décrit la population : taille, période, compagnies, destinations."""
    return {
        "debut": DEBUT_ETUDE,
        "fin": FIN_ETUDE,
        "nb_vols": len(vols),
        "nb_jours": int(vols["FlightDate"].dt.date.nunique()),
        "nb_mois": int(vols["annee_mois"].nunique()),
        "nb_compagnies": int(vols["Airline"].nunique()),
        "nb_destinations": int(vols["Dest"].nunique()),
        "part_delta_pct": round((vols["compagnie"] == "Delta").mean() * 100, 1),
        "part_par_type_pct": frequences(vols, "type_compagnie")["pourcentage"].to_dict(),
    }


def resume_qualite(vols):
    """Faits de qualité utiles à l'interprétation (voir l'étape qualité du notebook)."""
    annules = vols[vols["Cancelled"]]
    return {
        "nb_annules": int(len(annules)),
        "nb_annules_sans_retard_mesure": int(annules["DepDel15"].isna().sum()),
        "nb_annules_partis_de_la_porte": int(annules["DepTime"].notna().sum()),
        "nb_sans_immatriculation": int(vols["Tail_Number"].isna().sum()),
        "nb_lignes_en_double": int(vols.duplicated().sum()),
    }


def resume_indicateurs(vols):
    """Indicateurs globaux de ponctualité, sur une population donnée."""
    kpi = indicateurs_cles(vols)
    domino = effet_domino(vols)["pct_retard"]
    return {
        "nb_vols": kpi["nb_vols"],
        "pct_retard": round(kpi["pct_retard"], 1),
        "pct_retard_ou_annule": round(vols["retard_ou_annule"].mean() * 100, 1),
        "pct_annules": round(kpi["pct_annules"], 1),
        "retard_median_min": kpi["retard_median_min"],
        "retard_moyen_des_vols_en_retard_min": round(kpi["retard_moyen_des_vols_en_retard"], 1),
        "domino_pct_si_precedent_a_l_heure": domino["Vol précédent à l'heure"],
        "domino_pct_si_precedent_en_retard": domino["Vol précédent en retard"],
    }


def resume_correlations(vols):
    """Relations entre variables numériques (Pearson = linéaire ; Spearman = rangs, résiste aux extrêmes)."""
    from figures import VARIABLES_CORRELATION
    partis = vols_partis(vols)
    matrice = partis[list(VARIABLES_CORRELATION)].corr(method="spearman").round(3)
    return {
        "depart_arrivee_pearson": round(partis["DepDelay"].corr(partis["ArrDelay"]), 4),
        "depart_arrivee_spearman": round(partis["DepDelay"].corr(partis["ArrDelay"], method="spearman"), 4),
        "distance_rattrapage_spearman": round(partis["Distance"].corr(partis["minutes_rattrapees"], method="spearman"), 4),
        "matrice_spearman": en_dictionnaire(matrice.rename(index=VARIABLES_CORRELATION, columns=VARIABLES_CORRELATION)),
    }


def resume_taille(vols):
    """Justification par le calcul de la taille d'échantillon (n = 5 000)."""
    from sampling import TAILLE_ECHANTILLON, marge_erreur, taille_echantillon
    N = len(vols)
    p = float(vols_partis(vols)["DepDel15"].mean())
    mesa = vols[vols["compagnie"] == "Mesa"]
    p_mesa = float(vols_partis(mesa)["DepDel15"].mean())
    return {
        "N": N, "p": round(p, 4), "z": 1.96,
        "n_pour_1_point": round(taille_echantillon(N, p, 0.01)),
        "n_pour_1_point_p_0_5": round(taille_echantillon(N, 0.5, 0.01)),
        "n_pour_2_points": round(taille_echantillon(N, p, 0.02)),
        "n_retenu": TAILLE_ECHANTILLON,
        "fraction_pct": round(100 * TAILLE_ECHANTILLON / N, 2),
        "marge_n_retenu_points": round(100 * marge_erreur(TAILLE_ECHANTILLON, N, p), 2),
        "mesa": {"N_h": len(mesa), "p_h": round(p_mesa, 3),
                 "n_h_pour_5_points": round(taille_echantillon(len(mesa), p_mesa, 0.05)),
                 "n_h_si_aleatoire_5000": round(TAILLE_ECHANTILLON * len(mesa) / N)},
    }


def resume_approfondissement(vols):
    """Les analyses ajoutées lors de la deuxième itération (voir approfondissement.py et contexte.py)."""
    import approfondissement as a
    from contexte import FAITS_METIER, JOURS_NON_EXPLIQUES, verifier_evenements

    jours = a.par_jour_semaine(vols)
    paires, seuil = a.paires_non_distinctes(vols)
    domino = a.domino_par_tranche_et_rang(vols)
    taux_domino = domino["domino_par_tranche"]["mean"] * 100
    effectifs_domino = domino["domino_par_tranche"]["size"]
    extremes = a.valeurs_extremes(vols)
    return {
        "journalier": a.resume_journalier(vols),
        "decomposition_variance": a.decomposition_variance(vols),
        "reponses_echantillon": a.reponses_des_echantillons(vols),
        "jour_semaine": {**en_dictionnaire(jours["tableau"]),
                         "association": jours["association"],
                         "mardi_ou_mercredi_meilleur_chaque_saison": jours["mardi_ou_mercredi_meilleur_chaque_saison"]},
        "jour_semaine_par_saison": en_dictionnaire(jours["par_saison"]),
        "quantiles_retard_par_compagnie": en_dictionnaire(a.quantiles_retard_par_compagnie(vols)),
        "classes_par_type": en_dictionnaire(a.classes_par_type(vols)),
        "liens_bivaries": a.liens_bivaries(vols),
        "paires_compagnies": {
            "nb_paires": len(paires), "seuil_bonferroni": seuil,
            "nb_non_distinctes": int((~paires["distinctes"]).sum()),
            "non_distinctes": paires.loc[~paires["distinctes"], ["compagnie_a", "compagnie_b", "pct_a", "pct_b"]]
                                    .to_dict(orient="records"),
        },
        "domino_rang": {
            "pct_retard_par_rang": {str(int(k)): v for k, v in domino["pct_retard_par_rang"].items()},
            "pct_premiers_departs_apres_14h": domino["pct_premiers_departs_apres_14h"],
            "pct_retard_premier_depart_soir": domino["pct_retard_premier_depart_soir"],
            # tranches avec au moins 30 cas de vol précédent en retard (le matin n'en a que 9)
            "domino_par_tranche": {t: {"si_precedent_a_l_heure": round(taux_domino.loc[t, 0.0], 1),
                                       "si_precedent_en_retard": round(taux_domino.loc[t, 1.0], 1)}
                                   for t in taux_domino.index if effectifs_domino.loc[t, 1.0] >= 30},
        },
        "sensibilite_types": en_dictionnaire(a.sensibilite_types(vols)),
        "valeurs_extremes": {k: v for k, v in extremes.items() if k != "plus_grands"},
        "evenements": verifier_evenements(vols).to_dict(orient="records"),
        "jours_non_expliques": JOURS_NON_EXPLIQUES,
        "faits_metier": FAITS_METIER,
    }


def resume_equipe():
    """Data scientist (modèle) et data analyst (destinations, créneaux, météo, records), lus dans la base SQLite."""
    import analyste
    import modele
    from base_de_donnees import FICHIER_BASE
    from contexte import COMPAGNIES_CONTEXTE, SOURCES_COMPAGNIES
    if not FICHIER_BASE.exists():
        return None
    d, c = analyste.destinations(), analyste.creneaux()
    colonnes_dest = ["dest", "ville", "nb_vols", "pct_retard", "pct_annules"]
    jours, mediane = analyste.jours_non_expliques_avec_meteo()
    return {
        "modele": modele.resume(),
        "destinations": {"nb": d["nb_destinations"], "meilleures": d["meilleures"][colonnes_dest].to_dict(orient="records"),
                         "pires": d["pires"][colonnes_dest].to_dict(orient="records")},
        "creneaux": {"nb": c["nb_creneaux"], "meilleurs": c["meilleurs"].to_dict(orient="records"),
                     "pires": c["pires"].to_dict(orient="records")},
        "meteo_par_classe": en_dictionnaire(analyste.meteo_par_classe()),
        "correlations_meteo": analyste.correlations_meteo().to_dict(),
        "records": analyste.records(),
        "jours_non_expliques_meteo": jours.to_dict(orient="records"),
        "meteo_mediane": {k: float(v) for k, v in mediane.items()},
        "compagnies_contexte": {k: {"marque": m, "base": b, "remarque": r} for k, (m, b, r) in COMPAGNIES_CONTEXTE.items()},
        "sources_compagnies": SOURCES_COMPAGNIES,
    }


def calculer_resultats(vols):
    """Calcule l'ensemble des résultats chiffrés du projet."""
    ete = vols[vols["FlightDate"].between(DEBUT_ETE_2022, FIN_ETE_2022)]
    correlation_globale, correlation_par_compagnie = correlation_globale_et_par_groupe(
        vols, "TaxiOut", "DepDelayMinutes")
    taux_saison, rangs_saison = taux_par_compagnie_et_periode(vols, "saison")
    retard_par_mois = vols_partis(vols).groupby("annee_mois")["DepDel15"].mean().mul(100).round(1)

    resultats = {
        "population": resume_population(vols),
        "qualite": resume_qualite(vols),
        "indicateurs": resume_indicateurs(vols),
        "zoom_ete_2022": resume_indicateurs(ete),
        "descriptifs": en_dictionnaire(statistiques_descriptives(vols)),
        "classes_retard": frequences(vols, "classe_retard")["pourcentage"].to_dict(),
        "retard_par_mois_pct": retard_par_mois.to_dict(),
        "retard_par_saison_global_pct": vols_partis(vols).groupby("saison")["DepDel15"].mean().mul(100).round(1).to_dict(),
        "retard_par_tranche_pct": vols_partis(vols).groupby("tranche_horaire")["DepDel15"].mean().mul(100).round(1).to_dict(),
        "compagnies": en_dictionnaire(profil_compagnies(vols)),
        "composition_tranche_horaire": en_dictionnaire(comparaison_brut_standardise(vols, "tranche_horaire")),
        "composition_mois": en_dictionnaire(comparaison_brut_standardise(vols, "annee_mois")),
        "correlation_roulage_retard": {
            "globale": round(correlation_globale, 3),
            "par_compagnie": en_dictionnaire(correlation_par_compagnie),
        },
        "correlations": resume_correlations(vols),
        "retard_par_saison_pct": en_dictionnaire(taux_saison),
        "rang_par_saison": en_dictionnaire(rangs_saison),
        "domino_par_compagnie": en_dictionnaire(effet_domino_par_compagnie(vols)),
        "taille_echantillon": resume_taille(vols),
        "echantillonnage": resume_echantillonnage(),
        "approfondissement": resume_approfondissement(vols),
        "equipe": resume_equipe(),
    }
    resultats["conclusions_compagnies"] = conclusions_par_compagnie(resultats)
    return resultats


def enregistrer_resultats(resultats, fichier=FICHIER_RESULTATS):
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.write_text(json.dumps(resultats, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def charger_resultats(fichier=FICHIER_RESULTATS):
    return json.loads(fichier.read_text(encoding="utf-8"))


if __name__ == "__main__":
    from preprocessing import charger_vols_propres
    enregistrer_resultats(calculer_resultats(charger_vols_propres()))
    print(f"Résultats enregistrés dans {FICHIER_RESULTATS}")

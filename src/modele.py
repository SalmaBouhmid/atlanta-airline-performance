"""Le regard du data scientist : un modèle qui mesure chaque facteur À CONDITIONS ÉGALES.

Les analyses précédentes regardent les facteurs un par un (ou deux par deux). Mais la compagnie, l'heure,
la saison et la météo se mélangent : JetBlue vole-t-elle plus le soir ? les orages tombent-ils l'été ?
Une régression logistique estime l'effet de chaque facteur en tenant compte de tous les autres.

- Variable expliquée : DepDel15 (1 = départ avec 15 min de retard ou plus), vols partis.
- Variables explicatives : compagnie, tranche horaire, saison, jour de la semaine, vol précédent du même avion
  (à l'heure / en retard / pas de vol précédent), pluie à l'heure du départ, rafales fortes, journée pluvieuse.
- Données : la vue SQL v_depart_meteo (fusion vols × météo horaire × météo du jour), lue dans la base.

Lecture d'un « rapport de cotes » (odds ratio, OR) : OR = 2 → la cote du retard est doublée par rapport
à la référence, toutes choses égales par ailleurs ; OR = 1 → aucun effet ; OR < 1 → moins de retard.
C'est une ASSOCIATION, pas une preuve de cause (les causes officielles ne sont pas dans le fichier).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from base_de_donnees import lire

SEUIL_PLUIE_MM = 0.5      # pluie à l'heure du départ (mm dans l'heure)
SEUIL_RAFALES_KMH = 40    # rafales fortes à l'heure du départ
SEUIL_HEURES_PLUIE = 6    # journée pluvieuse : au moins 6 heures de pluie

REFERENCES = {"compagnie": "Delta", "tranche": "Matin", "saison": "Automne", "jour": "Mardi", "precedent": "a_l_heure"}
BLOCS = {
    "Compagnie": "C(compagnie, Treatment('Delta'))",
    "Heure de départ": "C(tranche, Treatment('Matin'))",
    "Saison": "C(saison, Treatment('Automne'))",
    "Jour de la semaine": "C(jour, Treatment('Mardi'))",
    "Vol précédent du même avion": "C(precedent, Treatment('a_l_heure'))",
    "Météo (pluie, rafales)": "pluie_depart + rafales_fortes + jour_pluvieux",
}
NOMS_PRECEDENT = {"a_l_heure": "vol précédent à l'heure", "en_retard": "vol précédent en retard",
                  "aucun": "pas de vol précédent (1er départ)"}


def preparer_donnees():
    """Une ligne par vol parti, avec des variables simples pour le modèle."""
    d = lire("depart_meteo").dropna(subset=["DepDel15"])
    d["tranche"] = d["tranche_horaire"].str.split(" ").str[0]          # Matin, Midi, Après-midi, Soir
    d["jour"] = d["jour_semaine"]
    d["precedent"] = d["vol_precedent_en_retard"].map({0.0: "a_l_heure", 1.0: "en_retard"}).fillna("aucun")
    d["pluie_depart"] = (d["pluie_heure_mm"] >= SEUIL_PLUIE_MM).astype(int)
    d["rafales_fortes"] = (d["rafales_heure_kmh"] >= SEUIL_RAFALES_KMH).astype(int)
    d["jour_pluvieux"] = (d["heures_de_pluie_jour"] >= SEUIL_HEURES_PLUIE).astype(int)
    return d


def formule(blocs=BLOCS):
    return "DepDel15 ~ " + " + ".join(blocs.values())


def nom_lisible(terme):
    """« C(compagnie, Treatment('Delta'))[T.JetBlue] » → « Compagnie : JetBlue »."""
    if terme == "Intercept":
        return "Constante"
    noms = {"pluie_depart": "Pluie à l'heure du départ", "rafales_fortes": "Rafales fortes (≥ 40 km/h)",
            "jour_pluvieux": "Journée pluvieuse (≥ 6 h de pluie)"}
    if terme in noms:
        return noms[terme]
    variable = terme.split("(")[1].split(",")[0]
    modalite = terme.split("[T.")[1].rstrip("]")
    etiquette = {"compagnie": "Compagnie", "tranche": "Heure", "saison": "Saison", "jour": "Jour",
                 "precedent": "Vol précédent"}[variable]
    return f"{etiquette} : {NOMS_PRECEDENT.get(modalite, modalite)}"


def ajuster(donnees=None):
    """Ajuste le modèle complet et mesure l'importance de chaque bloc de facteurs."""
    d = preparer_donnees() if donnees is None else donnees
    modele = smf.logit(formule(), d).fit(disp=0)
    ic = np.exp(modele.conf_int())
    rapports = pd.DataFrame({"rapport_de_cotes": np.exp(modele.params), "ic_bas": ic[0], "ic_haut": ic[1],
                             "p_valeur": modele.pvalues})
    rapports.index = [nom_lisible(t) for t in rapports.index]
    rapports = rapports.drop(index="Constante")
    # Importance : ce que le modèle perd (en pseudo-R²) quand on retire un bloc de facteurs
    importance = {}
    for nom in BLOCS:
        reduit = smf.logit(formule({k: v for k, v in BLOCS.items() if k != nom}), d).fit(disp=0)
        importance[nom] = modele.prsquared - reduit.prsquared
    importance = pd.Series(importance).sort_values(ascending=False)
    # Évaluation honnête : on entraîne sur 70 % des vols et on mesure sur les 30 % jamais vus
    entrainement, test = train_test_split(d, test_size=0.3, random_state=42, stratify=d["DepDel15"])
    modele_test = smf.logit(formule(), entrainement).fit(disp=0)
    prediction_test = modele_test.predict(test)
    return {
        "modele": modele, "rapports": rapports, "importance": importance,
        "pseudo_r2": modele.prsquared, "auc": roc_auc_score(d["DepDel15"], modele.predict(d)), "n": int(modele.nobs),
        "part_pluie_depart": d["pluie_depart"].mean(), "part_rafales": d["rafales_fortes"].mean(),
        "auc_test": roc_auc_score(test["DepDel15"], prediction_test),
        "brier_test": brier_score_loss(test["DepDel15"], prediction_test),
        "brier_reference": brier_score_loss(test["DepDel15"], np.full(len(test), entrainement["DepDel15"].mean())),
    }


def scenarios(modele):
    """Probabilité de retard prédite pour quelques vols « types » (meilleur et pire cas)."""
    exemples = pd.DataFrame([
        {"nom": "Endeavor, mardi matin d'automne, temps sec, avion à l'heure", "compagnie": "Endeavor (Delta Connection)",
         "tranche": "Matin", "saison": "Automne", "jour": "Mardi", "precedent": "a_l_heure",
         "pluie_depart": 0, "rafales_fortes": 0, "jour_pluvieux": 0},
        {"nom": "Delta, jeudi midi de printemps, temps sec, avion à l'heure", "compagnie": "Delta",
         "tranche": "Midi", "saison": "Printemps", "jour": "Jeudi", "precedent": "a_l_heure",
         "pluie_depart": 0, "rafales_fortes": 0, "jour_pluvieux": 0},
        {"nom": "JetBlue, vendredi soir d'été, pluie, avion déjà en retard", "compagnie": "JetBlue",
         "tranche": "Soir", "saison": "Été", "jour": "Vendredi", "precedent": "en_retard",
         "pluie_depart": 1, "rafales_fortes": 0, "jour_pluvieux": 1},
    ])
    exemples["probabilite_retard_pct"] = (modele.predict(exemples) * 100).round(1)
    return exemples[["nom", "probabilite_retard_pct"]]


def resume(resultat=None):
    """Résumé JSON du modèle pour resultats.json."""
    r = ajuster() if resultat is None else resultat
    return {
        "n": r["n"], "pseudo_r2": round(r["pseudo_r2"], 3), "auc": round(r["auc"], 3),
        "auc_test": round(r["auc_test"], 3), "brier_test": round(r["brier_test"], 4),
        "brier_reference": round(r["brier_reference"], 4),
        "part_vols_pluie_depart_pct": round(r["part_pluie_depart"] * 100, 1),
        "rapports_de_cotes": {k: {c: round(float(v), 3) if c != "p_valeur" else float(v) for c, v in ligne.items()}
                              for k, ligne in r["rapports"].iterrows()},
        "importance_pseudo_r2": {k: round(float(v), 4) for k, v in r["importance"].items()},
        "scenarios": {ligne.nom: float(ligne.probabilite_retard_pct) for ligne in scenarios(r["modele"]).itertuples()},
        "references": REFERENCES,
        # coefficients bruts (log de la cote) : servent au simulateur de risque du dashboard
        "coefficients": {k: float(v) for k, v in r["modele"].params.items()},
    }


if __name__ == "__main__":
    r = ajuster()
    pd.set_option("display.width", 200)
    print(f"n = {r['n']}, pseudo-R² = {r['pseudo_r2']:.3f}, AUC = {r['auc']:.3f}")
    print(r["rapports"].round(2).to_string())
    print(r["importance"].round(4))
    print(scenarios(r["modele"]).to_string())

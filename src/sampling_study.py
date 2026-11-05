"""Étude d'échantillonnage : chaque méthode préserve-t-elle ce dont la question a besoin ?

Principe : on connaît la population entière (319 335 vols), donc on peut comparer chaque
échantillon à la vérité. Pour chaque méthode, on répète le tirage 100 fois (guide, étape 8 :
« plusieurs échantillons plutôt qu'un seul ») et on mesure, à chaque tirage :

E1. les DISTRIBUTIONS des variables numériques (centiles, distance KS) ;
E2. les PROPORTIONS et les GROUPES RARES (parts des compagnies, compagnies « perdues ») ;
E3. les RELATIONS entre deux variables (profil horaire, effet domino, corrélation) ;
E4. les STRUCTURES à plusieurs variables et la COMPOSITION (compagnie × saison,
    corrélation globale contre corrélation par compagnie) ;
E5. le TEMPS (série mensuelle, jours couverts) ;
E6. les CONCLUSIONS de la question principale (restent-elles vraies dans l'échantillon ?) ;
E7. tout cela pour plusieurs TAILLES d'échantillon (courbe de perte d'information).

Une méthode n'est jamais félicitée pour ce qu'elle impose par construction : par exemple,
le temporel (par jour) couvre forcément tous les jours de l'année (E5).

Estimateur « naïf » : on calcule sur l'échantillon comme si c'était la population.
Estimateur « repondéré » : chaque vol compte avec son poids de sondage (pour les méthodes
qui déforment volontairement les proportions : le stratifié par compagnie).
"""

import numpy as np
import pandas as pd

from comparison import taux_standardise
from config import RACINE
from sampling import METHODES, METHODES_A_REPONDERER, poids_de_sondage

DOSSIER_ECHANTILLONNAGE = RACINE / "reports/echantillonnage"
NB_TIRAGES = 100
TAILLES_ETUDIEES = [1000, 2000, 5000, 10000, 20000]
METHODES_ETUDE_TAILLE = ["Aléatoire simple", "Stratifié par compagnie", "Temporel (par jour)"]
N_MIN_GROUPE = 30  # en dessous, la marge d'erreur d'un pourcentage dépasse ± 18 points

VARIABLES_DISTRIBUTION = {"DepDelayMinutes": "retard", "TaxiOut": "roulage",
                          "minutes_rattrapees": "rattrapage"}

# Ce que mesure chaque colonne de résultat : (expérience, description, unité)
DESCRIPTION_MESURES = {
    **{f"E1_ks_{nom}": ("E1", f"Distance KS sur la distribution du {nom}", "0 à 1")
       for nom in VARIABLES_DISTRIBUTION.values()},
    **{f"E1_erreur_p{q}_{nom}": ("E1", f"Erreur sur le centile {q} du {nom}", "min")
       for nom in VARIABLES_DISTRIBUTION.values() for q in (50, 90, 99)},
    "E2_ecart_max_parts_compagnies": ("E2", "Plus grand écart sur la part d'une compagnie", "points"),
    "E2_ecart_max_classes_retard": ("E2", "Plus grand écart sur la part d'une classe de retard", "points"),
    "E2_erreur_pct_grand_retard": ("E2", "Erreur sur le % de grands retards (> 45 min)", "points"),
    "E2_nb_compagnies_moins_30_vols": ("E2", "Compagnies avec moins de 30 vols dans l'échantillon", "nombre"),
    "E3_ecart_moyen_profil_horaire": ("E3", "Écart moyen sur la courbe % de retard par heure", "points"),
    "E3_erreur_ecart_domino": ("E3", "Erreur sur l'écart domino", "points"),
    "E3_erreur_corr_depart_arrivee": ("E3", "Erreur sur r(retard départ, retard arrivée)", "r"),
    "E4_erreur_moyenne_compagnie_saison": ("E4", "Erreur moyenne du tableau compagnie × saison", "points"),
    "E4_pct_cases_moins_30_vols": ("E4", "% des cases compagnie × saison avec moins de 30 vols", "%"),
    "E4_corr_globale_roulage_retard": ("E4", "Corrélation globale roulage-retard (population : 0,002)", "ρ"),
    "E4_erreur_corr_par_compagnie": ("E4", "Erreur moyenne de la corrélation roulage-retard par compagnie", "ρ"),
    "E5_ecart_moyen_serie_mensuelle": ("E5", "Écart moyen sur la série mensuelle du % de retard", "points"),
    "E5_nb_jours_couverts": ("E5", "Nombre de jours couverts (sur 365)", "jours"),
    "E6_nb_compagnies_comparables": ("E6", "Compagnies comparables (au moins 30 vols partis), sur 14", "nombre"),
    "E6_spearman_classement_compagnies": ("E6", "Corrélation de rang du classement des compagnies comparables",
                                          "ρ"),
    "E6_erreur_max_compagnie": ("E6", "Plus grande erreur sur le % de retard d'une compagnie", "points"),
    "E6_erreur_pct_retard_global": ("E6", "Erreur sur le % de retard global", "points"),
}

# Les conclusions de la question principale, établies sur la population (voir resultats.json)
CONCLUSIONS = {
    "C1": "Bas coût > Majeures > Régionales (% de retard)",
    "C2": "Southwest plus en retard que Delta",
    "C3": "Endeavor est la compagnie la plus ponctuelle",
    "C4": "American paraît plus ponctuelle que Delta, mais plus en retard à tranche horaire égale",
    "C5": "American < Delta en retard seul, mais > Delta en « retardé ou annulé »",
    "C6": "Effet domino : écart de plus de 15 points",
    "C7": "Roulage-retard : corrélation globale ≈ 0 mais positive dans la plupart des compagnies",
    "C8": "Le retard est plus fréquent le soir que le matin",
}
# Noms courts pour les graphiques
CONCLUSIONS_COURTES = {
    "C1": "Bas coût > Majeures\n> Régionales",
    "C2": "Southwest\n> Delta",
    "C3": "Endeavor la plus\nponctuelle",
    "C4": "American : effet\nde l'heure",
    "C5": "American / Delta :\nbiais du survivant",
    "C6": "Domino\n> 15 points",
    "C7": "Roulage-retard :\ntype Simpson",
    "C8": "Soir\n> matin",
}


# --------------------------------------------------------------------------- Calculs pondérés

def moyenne_ponderee(valeurs, poids):
    garder = ~np.isnan(valeurs)
    return np.average(valeurs[garder], weights=poids[garder]) if garder.any() else np.nan


def centile_pondere(valeurs, poids, q):
    """Centile q (entre 0 et 1) où chaque valeur compte avec son poids."""
    garder = ~np.isnan(valeurs)
    ordre = np.argsort(valeurs[garder])
    valeurs_triees, poids_tries = valeurs[garder][ordre], poids[garder][ordre]
    cumul = np.cumsum(poids_tries) / poids_tries.sum()
    return valeurs_triees[min(np.searchsorted(cumul, q), len(cumul) - 1)]


def distance_ks(valeurs, poids, grille, repartition_population):
    """Distance de Kolmogorov-Smirnov : le plus grand écart vertical entre les courbes cumulées
    (fonctions de répartition) de l'échantillon et de la population. 0 = identiques."""
    garder = ~np.isnan(valeurs)
    ordre = np.argsort(valeurs[garder])
    valeurs_triees = valeurs[garder][ordre]
    cumul = np.cumsum(poids[garder][ordre]) / poids[garder].sum()
    positions = np.searchsorted(valeurs_triees, grille, side="right") - 1
    repartition = np.where(positions >= 0, cumul[np.clip(positions, 0, None)], 0)
    return np.abs(repartition - repartition_population).max()


def taux_par_groupe(table, colonne_groupe, colonne_valeur="DepDel15", ponderer=True):
    """% moyen de colonne_valeur par groupe, pondéré par la colonne « poids » (ou non)."""
    donnees = table[table[colonne_valeur].notna()]
    poids = donnees["poids"] if ponderer else pd.Series(1.0, index=donnees.index)
    somme = (donnees[colonne_valeur] * poids).groupby(donnees[colonne_groupe], observed=True).sum()
    return somme / poids.groupby(donnees[colonne_groupe], observed=True).sum() * 100


def correlation_ponderee(x, y, poids):
    """Corrélation de Pearson pondérée (sur les rangs, elle approche celle de Spearman)."""
    mx, my = np.average(x, weights=poids), np.average(y, weights=poids)
    covariance = np.average((x - mx) * (y - my), weights=poids)
    return covariance / np.sqrt(np.average((x - mx) ** 2, weights=poids) * np.average((y - my) ** 2, weights=poids))


def spearman_pondere(table, x, y):
    donnees = table[[x, y, "poids"]].dropna()
    return correlation_ponderee(donnees[x].rank().to_numpy(), donnees[y].rank().to_numpy(),
                                donnees["poids"].to_numpy())


# --------------------------------------------------------------------------- Population de référence

def preparer_table(vols, poids=None):
    """Garde les colonnes utiles, ajoute les poids, repart d'un index 0..n-1 (le bootstrap a des doublons)."""
    colonnes = ["compagnie", "type_compagnie", "Cancelled", "DepDel15", "DepDelayMinutes", "DepDelay",
                "ArrDelay", "TaxiOut", "minutes_rattrapees", "heure_depart_prevue", "tranche_horaire",
                "saison", "annee_mois", "FlightDate", "classe_retard", "vol_precedent_en_retard",
                "retard_ou_annule"]
    table = vols[colonnes].reset_index(drop=True)
    table["poids"] = 1.0 if poids is None else np.asarray(poids, dtype=float)
    return table


def mesures_brutes(table):
    """Les statistiques qu'on compare entre échantillon et population (sur une table avec poids)."""
    partis = table[~table["Cancelled"]]
    w = partis["poids"].to_numpy()
    stats = {}
    for colonne, nom in VARIABLES_DISTRIBUTION.items():
        valeurs = partis[colonne].to_numpy(dtype=float)
        for q in (50, 90, 99):
            stats[f"p{q}_{nom}"] = centile_pondere(valeurs, w, q / 100)

    parts = table.groupby("compagnie")["poids"].sum() / table["poids"].sum() * 100
    classes = table.groupby("classe_retard", observed=True)["poids"].sum() / table["poids"].sum() * 100
    stats["parts_compagnies"] = parts
    stats["parts_classes"] = classes
    stats["pct_grand_retard"] = moyenne_ponderee((partis["DepDelayMinutes"] > 45).to_numpy(float), w) * 100
    stats["pct_retard"] = moyenne_ponderee(partis["DepDel15"].to_numpy(float), w) * 100
    stats["profil_horaire"] = taux_par_groupe(partis, "heure_depart_prevue")
    stats["par_tranche"] = taux_par_groupe(partis, "tranche_horaire")
    domino = taux_par_groupe(partis[partis["vol_precedent_en_retard"].notna()], "vol_precedent_en_retard")
    stats["ecart_domino"] = domino.get(1.0, np.nan) - domino.get(0.0, np.nan)
    depart_arrivee = partis[["DepDelay", "ArrDelay", "poids"]].dropna()
    stats["corr_depart_arrivee"] = correlation_ponderee(depart_arrivee["DepDelay"].to_numpy(),
                                                        depart_arrivee["ArrDelay"].to_numpy(),
                                                        depart_arrivee["poids"].to_numpy())
    stats["corr_globale_roulage"] = spearman_pondere(partis, "TaxiOut", "DepDelayMinutes")
    stats["serie_mensuelle"] = taux_par_groupe(partis, "annee_mois")
    stats["par_type"] = taux_par_groupe(partis, "type_compagnie")

    # Mesures PAR COMPAGNIE : à l'intérieur d'une strate, les poids sont égaux, inutile de pondérer
    effectifs = partis["compagnie"].value_counts()
    stats["effectifs_compagnies"] = table["compagnie"].value_counts()
    stats["par_compagnie"] = taux_par_groupe(partis, "compagnie", ponderer=False)
    stats["par_compagnie_retard_ou_annule"] = taux_par_groupe(table, "compagnie", "retard_ou_annule",
                                                              ponderer=False)
    cellules = partis.groupby(["compagnie", "saison"])["DepDel15"]
    stats["compagnie_saison"] = cellules.mean() * 100
    stats["compagnie_saison_effectifs"] = cellules.size()
    assez = effectifs[effectifs >= N_MIN_GROUPE].index
    stats["corr_par_compagnie"] = pd.Series({
        nom: groupe["TaxiOut"].corr(groupe["DepDelayMinutes"], method="spearman")
        for nom, groupe in partis[partis["compagnie"].isin(assez)].groupby("compagnie")
    })
    stats["compagnies_fiables"] = assez
    stats["jours"] = table["FlightDate"].nunique()
    return stats


def references_population(vols):
    """Tout ce qu'il faut savoir de la population pour juger un échantillon (calculé une seule fois)."""
    table = preparer_table(vols)
    reference = mesures_brutes(table)
    partis = table[~table["Cancelled"]]
    reference["grilles"] = {}
    for colonne, nom in VARIABLES_DISTRIBUTION.items():
        valeurs = np.sort(partis[colonne].dropna().to_numpy(dtype=float))
        grille = np.unique(valeurs)
        reference["grilles"][nom] = (grille, np.searchsorted(valeurs, grille, side="right") / len(valeurs))
    reference["standardise_tranche"] = taux_standardise(vols)["taux_standardise_pct"]
    reference["poids_tranches"] = partis["tranche_horaire"].value_counts(normalize=True)
    reference["cases_compagnie_saison"] = len(reference["compagnie_saison"])
    return reference


# --------------------------------------------------------------------------- Mesure d'un échantillon

def conclusions_vraies(stats, reference):
    """Pour chaque conclusion C1-C8 : est-elle encore vraie dans l'échantillon ?"""
    taux = stats["par_compagnie"].reindex(stats["compagnies_fiables"])
    type_ = stats["par_type"]
    tranches = stats["par_tranche"]

    def standardise(compagnie, table_tranches):
        if compagnie not in table_tranches.index:
            return np.nan
        ligne = table_tranches.loc[compagnie].dropna()
        poids = reference["poids_tranches"].reindex(ligne.index)
        return (ligne * poids).sum() / poids.sum()

    tranches_compagnie = stats["compagnie_tranche"]
    aa, dl = "American", "Delta"
    disponibles = {aa, dl} <= set(stats["compagnies_fiables"])
    taux_ou_annule = stats["par_compagnie_retard_ou_annule"]
    positives = (stats["corr_par_compagnie"] > 0).mean() if len(stats["corr_par_compagnie"]) else np.nan

    return {
        "C1": bool(type_.get("Bas coût", np.nan) > type_.get("Majeure", np.nan) > type_.get("Régionale", np.nan)),
        "C2": bool(taux.get("Southwest", np.nan) > taux.get("Delta", np.nan)),
        # « la plus ponctuelle DE TOUTES » : impossible à affirmer si certaines compagnies
        # ont trop peu de vols dans l'échantillon pour être comparées
        "C3": bool(len(stats["compagnies_fiables"]) == len(reference["par_compagnie"])
                   and taux.idxmin() == "Endeavor (Delta Connection)"),
        "C4": bool(disponibles and taux[aa] < taux[dl]
                   and standardise(aa, tranches_compagnie) > standardise(dl, tranches_compagnie)),
        "C5": bool(disponibles and taux[aa] < taux[dl] and taux_ou_annule[aa] > taux_ou_annule[dl]),
        "C6": bool(stats["ecart_domino"] > 15),
        "C7": bool(abs(stats["corr_globale_roulage"]) < 0.03 and positives >= 2 / 3),
        "C8": bool(tranches.get("Soir (après 18 h)", np.nan) > tranches.get("Matin (avant 10 h)", np.nan)),
    }


def mesurer_echantillon(echantillon, reference, poids=None):
    """Toutes les mesures E1-E6 d'un échantillon, comparé à la population de référence."""
    table = preparer_table(echantillon, poids)
    stats = mesures_brutes(table)
    partis = table[~table["Cancelled"]]
    stats["compagnie_tranche"] = partis.pivot_table(index="compagnie", columns="tranche_horaire",
                                                    values="DepDel15", aggfunc="mean") * 100
    resultat = {}

    # E1 : distributions
    w = partis["poids"].to_numpy()
    for colonne, nom in VARIABLES_DISTRIBUTION.items():
        grille, repartition = reference["grilles"][nom]
        resultat[f"E1_ks_{nom}"] = distance_ks(partis[colonne].to_numpy(dtype=float), w, grille, repartition)
        for q in (50, 90, 99):
            resultat[f"E1_erreur_p{q}_{nom}"] = abs(stats[f"p{q}_{nom}"] - reference[f"p{q}_{nom}"])

    # E2 : proportions et groupes rares
    parts = stats["parts_compagnies"].reindex(reference["parts_compagnies"].index, fill_value=0)
    resultat["E2_ecart_max_parts_compagnies"] = (parts - reference["parts_compagnies"]).abs().max()
    classes = stats["parts_classes"].reindex(reference["parts_classes"].index, fill_value=0)
    resultat["E2_ecart_max_classes_retard"] = (classes - reference["parts_classes"]).abs().max()
    resultat["E2_erreur_pct_grand_retard"] = abs(stats["pct_grand_retard"] - reference["pct_grand_retard"])
    effectifs = stats["effectifs_compagnies"].reindex(reference["effectifs_compagnies"].index, fill_value=0)
    resultat["E2_nb_compagnies_moins_30_vols"] = int((effectifs < N_MIN_GROUPE).sum())

    # E3 : relations entre deux variables
    profil = stats["profil_horaire"].reindex(reference["profil_horaire"].index)
    resultat["E3_ecart_moyen_profil_horaire"] = (profil - reference["profil_horaire"]).abs().mean()
    resultat["E3_erreur_ecart_domino"] = abs(stats["ecart_domino"] - reference["ecart_domino"])
    resultat["E3_erreur_corr_depart_arrivee"] = abs(stats["corr_depart_arrivee"] - reference["corr_depart_arrivee"])

    # E4 : structures à plusieurs variables et composition
    cellules = stats["compagnie_saison"][stats["compagnie_saison_effectifs"] >= N_MIN_GROUPE]
    vraies = reference["compagnie_saison"].reindex(cellules.index)
    resultat["E4_erreur_moyenne_compagnie_saison"] = (cellules - vraies).abs().mean()
    resultat["E4_pct_cases_moins_30_vols"] = (1 - len(cellules) / reference["cases_compagnie_saison"]) * 100
    resultat["E4_corr_globale_roulage_retard"] = stats["corr_globale_roulage"]
    par_compagnie = stats["corr_par_compagnie"]
    resultat["E4_erreur_corr_par_compagnie"] = (
        par_compagnie - reference["corr_par_compagnie"].reindex(par_compagnie.index)).abs().mean()

    # E5 : temps
    serie = stats["serie_mensuelle"].reindex(reference["serie_mensuelle"].index)
    resultat["E5_ecart_moyen_serie_mensuelle"] = (serie - reference["serie_mensuelle"]).abs().mean()
    resultat["E5_nb_jours_couverts"] = stats["jours"]

    # E6 : conclusions de la question principale
    fiables = stats["compagnies_fiables"]
    taux = stats["par_compagnie"].reindex(fiables)
    vrais = reference["par_compagnie"].reindex(fiables)
    # Attention : la corrélation de rang ne porte que sur les compagnies comparables ; avec un petit
    # échantillon il en reste peu, et classer 3 compagnies est facile. Toujours la lire avec le nombre
    # de compagnies comparables, sinon on récompense la perte d'information.
    resultat["E6_nb_compagnies_comparables"] = len(fiables)
    resultat["E6_spearman_classement_compagnies"] = taux.corr(vrais, method="spearman") if len(fiables) > 2 else np.nan
    resultat["E6_erreur_max_compagnie"] = (stats["par_compagnie"] - reference["par_compagnie"]).abs().max()
    resultat["E6_erreur_pct_retard_global"] = abs(stats["pct_retard"] - reference["pct_retard"])
    resultat.update(conclusions_vraies(stats, reference))
    return resultat


# --------------------------------------------------------------------------- Expériences répétées

def tirer_et_mesurer(vols, reference, methode, taille, graine):
    """Tire un échantillon et renvoie ses mesures naïves (et repondérées si la méthode l'exige)."""
    echantillon = METHODES[methode](vols, taille=taille, graine=graine)
    lignes = [{"methode": methode, "estimateur": "naïf", "taille": taille, "tirage": graine,
               "taille_obtenue": len(echantillon), **mesurer_echantillon(echantillon, reference)}]
    if methode in METHODES_A_REPONDERER:
        poids = poids_de_sondage(echantillon, vols)
        lignes.append({"methode": methode, "estimateur": "repondéré", "taille": taille, "tirage": graine,
                       "taille_obtenue": len(echantillon),
                       **mesurer_echantillon(echantillon, reference, poids)})
    return lignes


def etude_methodes(vols, reference=None, taille=5000, nb_tirages=NB_TIRAGES, methodes=None):
    """E1-E6 : toutes les méthodes, une taille, nb_tirages tirages chacune."""
    reference = reference if reference is not None else references_population(vols)
    lignes = []
    for methode in methodes or METHODES:
        for graine in range(nb_tirages):
            lignes += tirer_et_mesurer(vols, reference, methode, taille, graine)
    return pd.DataFrame(lignes)


def etude_tailles(vols, reference=None, tailles=TAILLES_ETUDIEES, nb_tirages=NB_TIRAGES,
                  methodes=METHODES_ETUDE_TAILLE):
    """E7 : quelques méthodes, plusieurs tailles d'échantillon."""
    reference = reference if reference is not None else references_population(vols)
    return pd.concat([etude_methodes(vols, reference, taille, nb_tirages, methodes) for taille in tailles],
                     ignore_index=True)


def synthese(mesures):
    """Moyenne de chaque mesure et % de conclusions conservées, par méthode, estimateur et taille."""
    colonnes_mesures = [c for c in mesures.columns if c[:2] in {"E1", "E2", "E3", "E4", "E5", "E6"}]
    groupes = mesures.groupby(["methode", "estimateur", "taille"])
    resume = groupes[colonnes_mesures].mean()
    conclusions = groupes[list(CONCLUSIONS)].mean() * 100
    conclusions.columns = [f"pct_tirages_{c}" for c in conclusions.columns]
    resume = resume.join(conclusions)
    resume["pct_conclusions_conservees"] = groupes[list(CONCLUSIONS)].mean().mean(axis=1) * 100
    return resume.round(3)


if __name__ == "__main__":
    import time

    from preprocessing import charger_vols_propres

    DOSSIER_ECHANTILLONNAGE.mkdir(parents=True, exist_ok=True)
    vols = charger_vols_propres()
    debut = time.time()
    reference = references_population(vols)
    verite = conclusions_vraies({**reference, "compagnie_tranche": vols[~vols["Cancelled"]].pivot_table(
        index="compagnie", columns="tranche_horaire", values="DepDel15", aggfunc="mean") * 100}, reference)
    print("Conclusions vraies dans la population :", verite)

    mesures = etude_methodes(vols, reference)
    mesures.to_parquet(DOSSIER_ECHANTILLONNAGE / "mesures_methodes.parquet", index=False)
    print(f"E1-E6 : {len(mesures)} échantillons mesurés ({time.time() - debut:.0f} s)")

    tailles = etude_tailles(vols, reference)
    tailles.to_parquet(DOSSIER_ECHANTILLONNAGE / "mesures_tailles.parquet", index=False)
    print(f"E7 : {len(tailles)} échantillons mesurés ({time.time() - debut:.0f} s)")

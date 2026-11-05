"""Approfondissement : ce que les premières analyses ne montraient pas (deuxième itération de l'étude).

Chaque fonction répond à une critique précise du projet :
1. le temps vu par mois cache les jours extrêmes          → serie_journaliere, resume_journalier
2. le jour de la semaine n'était pas étudié                → par_jour_semaine
3. le bivarié n'avait pas de mesure de la force du lien    → association_categorielle, liens_bivaries
4. le classement 1 à 14 n'était pas testé                  → paires_non_distinctes
5. le rang du départ dans la journée semblait contredire le domino → domino_par_tranche_et_rang
6. le type « bas coût » est un choix                       → sensibilite_types
7. les valeurs extrêmes n'étaient pas examinées une à une  → valeurs_extremes

La problématique ne change pas : les 14 compagnies et les conclusions restent les mêmes ;
ces analyses les précisent, les vérifient ou en montrent les limites.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from analysis import vols_partis

ORDRE_JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
SEUIL_MAUVAIS_JOUR = 40  # % de vols en retard : plus du double de la médiane des jours (16 %)


# --------------------------------------------------------------------------- 1. Le temps, jour par jour
def serie_journaliere(vols):
    """Une ligne par jour : nombre de vols, % en retard, % annulés, moyenne mobile sur 7 jours."""
    partis = vols_partis(vols)
    jour = pd.DataFrame({
        "nb_vols": vols.groupby("FlightDate").size(),
        "pct_retard": partis.groupby("FlightDate")["DepDel15"].mean() * 100,
        "pct_annules": vols.groupby("FlightDate")["Cancelled"].mean() * 100,
    })
    jour["moyenne_7_jours"] = jour["pct_retard"].rolling(7, center=True, min_periods=4).mean()
    return jour


def resume_journalier(vols):
    """Ce que la vue par jour apporte par rapport à la vue par mois."""
    jour = serie_journaliere(vols)
    mois = vols_partis(vols).groupby("annee_mois")["DepDel15"].mean() * 100
    partis = vols_partis(vols)
    pires = jour["pct_retard"] >= jour["pct_retard"].quantile(0.9)
    retards_pires_jours = partis[partis["FlightDate"].isin(jour.index[pires])]["DepDel15"].sum()
    return {
        "nb_jours": len(jour),
        "jour_min": {"date": f"{jour['pct_retard'].idxmin():%Y-%m-%d}", "pct": round(jour["pct_retard"].min(), 1)},
        "jour_max": {"date": f"{jour['pct_retard'].idxmax():%Y-%m-%d}", "pct": round(jour["pct_retard"].max(), 1)},
        "mediane_jours": round(jour["pct_retard"].median(), 1),
        "ecart_type_jours": round(jour["pct_retard"].std(), 1),
        "ecart_type_mois": round(mois.std(), 1),
        "nb_mauvais_jours": int((jour["pct_retard"] >= SEUIL_MAUVAIS_JOUR).sum()),
        "pct_retards_dans_10pct_pires_jours": round(retards_pires_jours / partis["DepDel15"].sum() * 100, 1),
        # Un mauvais jour est-il suivi d'un mauvais jour ? (corrélation d'un jour avec le lendemain)
        "autocorrelation_1_jour": round(jour["pct_retard"].autocorr(1), 2),
        "jour_max_annulations": {"date": f"{jour['pct_annules'].idxmax():%Y-%m-%d}",
                                 "pct": round(jour["pct_annules"].max(), 1)},
        "nb_jours_plus_5pct_annules": int((jour["pct_annules"] > 5).sum()),
        "pires_jours": {f"{d:%Y-%m-%d}": round(t, 1) for d, t in jour["pct_retard"].nlargest(8).items()},
    }


# --------------------------------------------------------------------------- 2. Le jour de la semaine
def association_categorielle(table):
    """Test du khi-deux et V de Cramér d'un tableau croisé (force du lien : 0 = aucun, 1 = total)."""
    chi2, p, ddl, attendus = stats.chi2_contingency(table)
    v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
    return {"chi2": round(float(chi2), 0), "ddl": int(ddl), "p_valeur": float(p), "v_cramer": round(float(v), 3),
            "effectif_attendu_min": round(float(attendus.min()), 1)}


def par_jour_semaine(vols):
    """% de retard et nombre moyen de vols par jour de la semaine, et force du lien jour × retard."""
    partis = vols_partis(vols)
    tableau = pd.DataFrame({
        "pct_retard": partis.groupby("jour_semaine")["DepDel15"].mean() * 100,
        "vols_par_jour": vols.groupby("jour_semaine").size() / vols.groupby("jour_semaine")["FlightDate"].nunique(),
    }).reindex(ORDRE_JOURS)
    saisons = (partis.pivot_table(index="saison", columns="jour_semaine", values="DepDel15") * 100).reindex(
        columns=ORDRE_JOURS)
    meilleur_partout = (saisons.idxmin(axis=1).isin(["Mardi", "Mercredi"])).all()
    return {
        "tableau": tableau.round(1),
        "par_saison": saisons.round(1),
        "association": association_categorielle(pd.crosstab(partis["jour_semaine"], partis["DepDel15"])),
        "mardi_ou_mercredi_meilleur_chaque_saison": bool(meilleur_partout),
    }


# --------------------------------------------------------------------------- 3. Le bivarié et la force des liens
def quantiles_retard_par_compagnie(vols):
    """Catégorielle × numérique : médiane, 3e quartile et 90e centile du retard (min) par compagnie."""
    partis = vols_partis(vols)
    tableau = partis.groupby("compagnie")["DepDelayMinutes"].quantile([0.5, 0.75, 0.9]).unstack()
    tableau.columns = ["mediane", "q3", "p90"]
    tableau["type"] = vols.groupby("compagnie")["type_compagnie"].first()
    return tableau.sort_values("p90")


def liens_bivaries(vols):
    """La force de chaque lien entre deux variables, avec la mesure adaptée au type des variables."""
    partis = vols_partis(vols)
    groupes = [g.dropna().to_numpy() for _, g in partis.groupby("compagnie")["DepDelayMinutes"]]
    h, p_kw = stats.kruskal(*groupes)
    n, k = sum(len(g) for g in groupes), len(groupes)
    liens = {
        "compagnie × retard (min)": {"type": "catégorielle × numérique", "mesure": "epsilon² (Kruskal-Wallis)",
                                     "valeur": round((h - k + 1) / (n - k), 3), "p_valeur": float(p_kw)},
    }
    for nom, x, y, table in [
        ("type × retardé ou annulé", "type_compagnie", "retard_ou_annule", vols),
        ("tranche horaire × en retard", "tranche_horaire", "DepDel15", partis),
        ("jour de la semaine × en retard", "jour_semaine", "DepDel15", partis),
        ("saison × en retard", "saison", "DepDel15", partis),
        ("compagnie × classe de retard", "compagnie", "classe_retard", vols),
    ]:
        resultat = association_categorielle(pd.crosstab(table[x], table[y]))
        liens[nom] = {"type": "catégorielle × catégorielle", "mesure": "V de Cramér",
                      "valeur": resultat["v_cramer"], "p_valeur": resultat["p_valeur"]}
    for nom, x, y in [("retard départ × retard arrivée", "DepDelay", "ArrDelay"),
                      ("heure de départ × retard (min)", "heure_depart_prevue", "DepDelayMinutes"),
                      ("distance × retard (min)", "Distance", "DepDelayMinutes"),
                      ("distance × minutes rattrapées", "Distance", "minutes_rattrapees")]:
        couple = partis[[x, y]].dropna()
        rho, p = stats.spearmanr(couple[x], couple[y])
        liens[nom] = {"type": "numérique × numérique", "mesure": "rho de Spearman",
                      "valeur": round(float(rho), 3), "p_valeur": float(p)}
    return liens


def classes_par_type(vols):
    """Catégorielle × catégorielle : répartition des classes de retard dans chaque type (en %)."""
    return (pd.crosstab(vols["type_compagnie"], vols["classe_retard"], normalize="index") * 100).round(1)


# --------------------------------------------------------------------------- 4. Le classement est-il testé ?
def paires_non_distinctes(vols, indicateur="retard_ou_annule", alpha=0.05):
    """Test de deux proportions pour chaque paire de compagnies, avec correction de Bonferroni.

    Avec 14 compagnies, on fait 91 tests : sans correction, environ 5 sortiraient « significatifs »
    par hasard. Bonferroni divise le seuil par 91. Renvoie les paires qu'on ne peut PAS départager.
    """
    taux = vols.groupby("compagnie")[indicateur].agg(["mean", "size"])
    paires = list(combinations(taux.index, 2))
    seuil = alpha / len(paires)
    lignes = []
    for a, b in paires:
        na, nb = taux.loc[a, "size"], taux.loc[b, "size"]
        pa, pb = taux.loc[a, "mean"], taux.loc[b, "mean"]
        commun = (pa * na + pb * nb) / (na + nb)
        z = (pa - pb) / np.sqrt(commun * (1 - commun) * (1 / na + 1 / nb))
        p = 2 * stats.norm.sf(abs(z))
        lignes.append({"compagnie_a": a, "compagnie_b": b, "pct_a": round(pa * 100, 1), "pct_b": round(pb * 100, 1),
                       "ecart_points": round(abs(pa - pb) * 100, 1), "p_valeur": p, "distinctes": bool(p < seuil)})
    return pd.DataFrame(lignes), seuil


# --------------------------------------------------------------------------- 5. Domino : le rang dans la journée
def domino_par_tranche_et_rang(vols):
    """Le domino tient-il à chaque heure ? Et pourquoi le rang du départ semble ne rien changer ?

    rang_rotation compte seulement les départs d'ATLANTA : un « 1er départ d'Atlanta » à 19 h vient
    souvent d'un avion qui a déjà volé ailleurs (vols absents de nos données).
    """
    partis = vols_partis(vols).dropna(subset=["rang_rotation"])
    premier = partis[partis["rang_rotation"] == 1]
    return {
        "pct_retard_par_rang": (partis.groupby(partis["rang_rotation"].clip(upper=3))["DepDel15"].mean() * 100).round(1),
        "pct_premiers_departs_apres_14h": round((premier["heure_depart_prevue"] >= 14).mean() * 100, 1),
        "pct_retard_premier_depart_soir": round(
            premier.loc[premier["tranche_horaire"] == "Soir (après 18 h)", "DepDel15"].mean() * 100, 1),
        "domino_par_tranche": (partis.dropna(subset=["vol_precedent_en_retard"]).pivot_table(
            index="tranche_horaire", columns="vol_precedent_en_retard", values="DepDel15", aggfunc=["mean", "size"])),
    }


# --------------------------------------------------------------------------- 6. Plusieurs versions des types
def sensibilite_types(vols):
    """La conclusion « bas coût > majeures > régionales » dépend-elle de nos choix ? (plan « multi-versions »)"""
    type_ = vols["type_compagnie"]
    versions = {
        "A : classification retenue": type_,
        "B : JetBlue classée majeure (« hybride »)": type_.where(vols["compagnie"] != "JetBlue", "Majeure"),
        "C : sans la plus grosse compagnie de chaque type": type_.where(
            ~vols["compagnie"].isin(["Delta", "Endeavor (Delta Connection)", "Southwest"])),
    }
    lignes = {}
    for nom, colonne in versions.items():
        taux = vols.groupby(colonne)["retard_ou_annule"].mean() * 100
        lignes[nom] = {**taux.round(1).to_dict(),
                       "ordre_conserve": bool(taux["Bas coût"] > taux["Majeure"] > taux["Régionale"])}
    return pd.DataFrame(lignes).T


# --------------------------------------------------------------------------- 7. Les valeurs extrêmes, une à une
def valeurs_extremes(vols):
    """Classe les retards extrêmes selon la grille du plan de travail (erreur / corrigible / suspecte / indéterminée).

    Contrôle de cohérence : le retard annoncé doit correspondre à l'heure réelle moins l'heure prévue
    (à 24 h près, car un vol peut partir le lendemain).
    """
    partis = vols_partis(vols)
    minutes = lambda heure: (heure // 100) * 60 + heure % 100  # noqa: E731  (1435 → 875 min)
    ecart = ((minutes(partis["DepTime"]) - minutes(partis["CRSDepTime"])) % 1440 - partis["DepDelay"] % 1440).abs()
    incoherents = int((ecart.between(1.5, 1438.5)).sum())
    return {
        "nb_retards_plus_6h": int((partis["DepDelay"] > 360).sum()),
        "nb_retards_plus_24h": int((partis["DepDelay"] > 1440).sum()),
        "retard_max_min": float(partis["DepDelay"].max()),
        "nb_heures_incoherentes": incoherents,
        "nb_arrivee_incoherente": int(((partis["ArrDelay"] - partis["DepDelay"]).abs() > 300).sum()),
        "decision": "valeurs suspectes : conservées et signalées (pas de preuve d'erreur ; un vol reporté au "
                    "lendemain garde son numéro). Erreurs vérifiées : aucune.",
        "plus_grands": partis.nlargest(5, "DepDelay")[["FlightDate", "compagnie", "Dest", "DepDelay", "ArrDelay"]],
    }


def decomposition_variance(vols, groupes=("FlightDate", "compagnie", "tranche_horaire", "saison"), taille_echantillon=5000):
    """Variance du retard (DepDel15) décomposée en intra-groupe et inter-groupe, pour chaque façon de grouper.

    Pour chaque groupe : part de la variance qui vient des différences ENTRE groupes (eta²) et corrélation
    intra-classe (ICC, méthode ANOVA) : à quel point les vols d'un même groupe se ressemblent.
    Pour les jours (les grappes de l'échantillonnage) : effet de plan deff = 1 + (m - 1) · ICC, taille
    efficace n / deff et marge d'erreur attendue avec cette taille efficace.
    """
    partis = vols_partis(vols)
    y = partis["DepDel15"].astype(float)
    total = ((y - y.mean()) ** 2).sum()
    resultat = {}
    for colonne in groupes:
        g = partis.groupby(colonne, observed=True)["DepDel15"]
        n_i, moyennes = g.size(), g.mean()
        k, n = len(n_i), len(y)
        inter = (n_i * (moyennes - y.mean()) ** 2).sum()
        intra = total - inter
        msb, msw = inter / (k - 1), intra / (n - k)
        m0 = (n - (n_i ** 2).sum() / n) / (k - 1)
        icc = (msb - msw) / (msb + (m0 - 1) * msw)
        resultat[colonne] = {"nb_groupes": int(k), "taille_moyenne": round(n / k, 1),
                             "part_inter_pct": round(100 * inter / total, 2), "icc": round(float(icc), 4)}
    jours = resultat["FlightDate"]
    deff = 1 + (jours["taille_moyenne"] - 1) * jours["icc"]
    n_efficace = taille_echantillon / deff
    p = y.mean()
    resultat["grappes_jours"] = {"effet_de_plan": round(deff, 1), "taille_efficace": round(n_efficace),
                                 "marge_attendue_points": round(1.96 * np.sqrt(p * (1 - p) / n_efficace) * 100, 2)}
    return resultat


def reponses_des_echantillons(vols, nb=30, taille=5000):
    """Ce que les échantillons répondent SEULS (sans regarder la population) et la stabilité des statistiques.

    - 30 échantillons stratifiés par compagnie : % retardés ou annulés de chaque compagnie (médiane et plage 5 %-95 %).
    - 30 échantillons au hasard : moyenne, écart-type et coefficient de variation de six statistiques.
    """
    from sampling import METHODES
    lignes, stabilite = [], []
    for graine in range(nb):
        e = METHODES["Stratifié par compagnie"](vols, taille=taille, graine=graine)
        lignes.append((e.groupby("compagnie", observed=True)["retard_ou_annule"].mean() * 100).rename(graine))
        a = METHODES["Aléatoire simple"](vols, taille=taille, graine=graine)
        pa = vols_partis(a)
        tranche = pa.groupby("tranche_horaire", observed=True)["DepDel15"].mean() * 100
        domino = pa.groupby("vol_precedent_en_retard")["DepDel15"].mean() * 100
        stabilite.append({"% de vols en retard": pa["DepDel15"].mean() * 100, "% annulés": a["Cancelled"].mean() * 100,
                          "Retard médian (min)": pa["DepDelay"].median(), "Retard moyen (min)": pa["DepDelay"].mean(),
                          "Écart soir − matin (pts)": tranche["Soir (après 18 h)"] - tranche["Matin (avant 10 h)"],
                          "Écart domino (pts)": domino.get(1.0, np.nan) - domino.get(0.0, np.nan)})
    par_cie = pd.concat(lignes, axis=1)
    stabilite = pd.DataFrame(stabilite)
    return {
        "nb_echantillons": nb, "taille": taille,
        "par_compagnie": {c: {"mediane": round(float(par_cie.loc[c].median()), 2), "bas": round(float(par_cie.loc[c].quantile(0.05)), 2),
                              "haut": round(float(par_cie.loc[c].quantile(0.95)), 2)} for c in par_cie.index},
        "stabilite": {s: {"moyenne": round(float(stabilite[s].mean()), 3), "ecart_type": round(float(stabilite[s].std()), 3),
                          "cv_pct": round(float(stabilite[s].std() / abs(stabilite[s].mean()) * 100), 2) if stabilite[s].mean() else 0.0}
                      for s in stabilite.columns},
    }

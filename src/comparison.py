"""Comparaison des compagnies et effets de composition (questions Q1 à Q3).

Ce fichier ne dessine rien : il calcule les tableaux de la comparaison.
- profil_compagnies : un tableau « carte d'identité » de chaque compagnie (Q1) ;
- comparaison_brut_standardise : la même comparaison « à heure égale » (Q2, effet de composition) ;
- correlation_globale_et_par_groupe : une relation peut disparaître quand on mélange les groupes (Q2) ;
- taux_par_compagnie_et_periode : le classement change-t-il avec la saison ? (Q3).
"""

import pandas as pd

from analysis import marge_erreur_95, vols_partis


def profil_compagnies(vols):
    """Profil de chaque compagnie : volume, ponctualité, gravité, fiabilité et exploitation.

    Les pourcentages de retard sont calculés sur les vols partis ; le taux « retardé ou annulé »
    et le taux d'annulation sur tous les vols programmés. La marge d'erreur à 95 % rappelle
    qu'un pourcentage mesuré sur peu de vols est fragile.
    """
    partis = vols_partis(vols)
    groupes_partis = partis.groupby("compagnie")
    groupes_tous = vols.groupby("compagnie")

    profil = pd.DataFrame({
        "type": groupes_tous["type_compagnie"].first(),
        "nb_vols": groupes_tous.size(),
        "part_vols_pct": groupes_tous.size() / len(vols) * 100,
        "pct_retard": groupes_partis["DepDel15"].mean() * 100,
        "pct_retard_ou_annule": groupes_tous["retard_ou_annule"].mean() * 100,
        "pct_annules": groupes_tous["Cancelled"].mean() * 100,
        "pct_grand_retard": groupes_partis["DepDelayMinutes"].apply(lambda s: (s > 45).mean() * 100),
        "retard_p90_min": groupes_partis["DepDelayMinutes"].quantile(0.90),
        "roulage_median_min": groupes_partis["TaxiOut"].median(),
        "rattrapage_median_min": groupes_partis["minutes_rattrapees"].median(),
        "distance_mediane": groupes_partis["Distance"].median(),
        "nb_destinations": groupes_tous["Dest"].nunique(),
        "part_vols_soir_pct": groupes_tous["heure_depart_prevue"].apply(lambda h: (h >= 18).mean() * 100),
    })
    profil["marge_95_pct_retard"] = marge_erreur_95(profil["pct_retard"] / 100, groupes_partis.size())
    return profil.sort_values("pct_retard_ou_annule").round(1)


def taux_standardise(vols, groupe="compagnie", strate="tranche_horaire", indicateur="DepDel15"):
    """Taux de chaque groupe s'il avait la même répartition par strate que toute la population.

    Méthode de standardisation directe : taux_standardisé = Σ poids_strate × taux_du_groupe_dans_la_strate,
    avec les poids de la population entière (même formule que la moyenne globale du guide :
    x̄ = Σ w_h × μ_h). Si un groupe n'a aucun vol dans une strate, on répartit les poids
    sur les strates où il vole (et la colonne « couverture_pct » l'indique).

    Strates conseillées : les 4 tranches horaires ou les 12 mois. Des strates trop fines
    (heure par heure) donnent des taux instables pour les petites compagnies, qui n'ont
    que quelques vols dans certaines strates.
    """
    partis = vols_partis(vols)
    poids = partis[strate].value_counts(normalize=True)
    taux = partis.pivot_table(index=groupe, columns=strate, values=indicateur, aggfunc="mean")
    poids_presents = taux.notna().mul(poids, axis=1)
    standardise = taux.mul(poids, axis=1).sum(axis=1) / poids_presents.sum(axis=1)
    return pd.DataFrame({
        "taux_standardise_pct": standardise * 100,
        "couverture_pct": poids_presents.sum(axis=1) * 100,
    })


def comparaison_brut_standardise(vols, strate="tranche_horaire"):
    """Compare le % de retard brut et le % « à strate égale » de chaque compagnie, avec leurs rangs.

    Si l'écart est grand ou si le rang change, la composition (quand / où la compagnie vole)
    contribue à sa différence apparente : c'est un effet de composition.
    """
    brut = vols_partis(vols).groupby("compagnie")["DepDel15"].mean() * 100
    tableau = taux_standardise(vols, strate=strate).assign(pct_retard_brut=brut)
    tableau["ecart_points"] = tableau["taux_standardise_pct"] - tableau["pct_retard_brut"]
    tableau["rang_brut"] = tableau["pct_retard_brut"].rank(ascending=False).astype(int)
    tableau["rang_standardise"] = tableau["taux_standardise_pct"].rank(ascending=False).astype(int)
    colonnes = ["pct_retard_brut", "taux_standardise_pct", "ecart_points", "rang_brut",
                "rang_standardise", "couverture_pct"]
    return tableau[colonnes].sort_values("pct_retard_brut", ascending=False).round(1)


def correlation_globale_et_par_groupe(vols, x, y, groupe="compagnie", methode="spearman"):
    """Corrélation entre x et y sur tous les vols mélangés, puis à l'intérieur de chaque groupe.

    Si la corrélation globale est très différente de celles des groupes (voire de signe opposé),
    le mélange des groupes masque ou crée la relation : effet de composition « de type Simpson ».
    Spearman (corrélation des rangs) est utilisé par défaut : il résiste mieux que Pearson
    aux valeurs extrêmes, nombreuses dans les retards.
    """
    donnees = vols_partis(vols)[[x, y, groupe]].dropna()
    par_groupe = donnees.groupby(groupe).apply(
        lambda t: pd.Series({"correlation": t[x].corr(t[y], method=methode), "n": len(t)}),
        include_groups=False,
    )
    globale = donnees[x].corr(donnees[y], method=methode)
    return globale, par_groupe.round(3)


def taux_par_compagnie_et_periode(vols, periode="saison", indicateur="DepDel15"):
    """% de retard de chaque compagnie pour chaque période (saison, mois...) et rang dans la période."""
    taux = vols_partis(vols).pivot_table(index="compagnie", columns=periode, values=indicateur,
                                         aggfunc="mean") * 100
    rangs = taux.rank(ascending=False)
    return taux.round(1), rangs


# En dessous de 30 cas, la marge d'erreur d'un pourcentage dépasse ± 18 points
# (1,96 × √(0,25 / 30) ≈ 0,18) : le pourcentage n'est pas interprétable.
N_MIN_CAS_DOMINO = 30


def effet_domino_par_compagnie(vols, n_min=N_MIN_CAS_DOMINO):
    """Écart domino par compagnie : % de retard si le vol précédent du même avion était en retard ou non.

    n_vol_precedent_en_retard indique sur combien de cas repose le second pourcentage :
    l'effet domino mesuré ici n'existe que pour les compagnies dont les avions partent
    plusieurs fois par jour d'Atlanta.
    """
    partis = vols_partis(vols)
    partis = partis[partis["vol_precedent_en_retard"].notna()]
    tableau = partis.pivot_table(index="compagnie", columns="vol_precedent_en_retard",
                                 values="DepDel15", aggfunc=["mean", "size"])
    resultat = pd.DataFrame({
        "pct_si_precedent_a_l_heure": tableau[("mean", 0.0)] * 100,
        "pct_si_precedent_en_retard": tableau[("mean", 1.0)] * 100,
        "n_vol_precedent_en_retard": tableau[("size", 1.0)],
    })
    resultat["ecart_points"] = resultat["pct_si_precedent_en_retard"] - resultat["pct_si_precedent_a_l_heure"]
    resultat = resultat[resultat["n_vol_precedent_en_retard"] >= n_min]
    return resultat.sort_values("n_vol_precedent_en_retard", ascending=False).round(1)

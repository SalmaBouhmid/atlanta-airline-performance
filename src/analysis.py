"""Calculs d'analyse : tableaux et indicateurs utilisés par le notebook et le dashboard.

Ce fichier ne dessine rien : il prépare les chiffres. Les graphiques sont dans visualizations.py.
"""

import pandas as pd

# Colonnes au cœur de la problématique (utilisées pour les statistiques et corrélations)
VARIABLES_NUMERIQUES_CLES = ["DepDelay", "ArrDelay", "TaxiOut", "TaxiIn", "Distance", "AirTime"]

# Variables numériques décrites dans la partie « statistiques descriptives »
VARIABLES_DESCRIPTIVES = {
    "DepDelay": "Retard au départ (min)",
    "ArrDelay": "Retard à l'arrivée (min)",
    "TaxiOut": "Roulage au départ (min)",
    "minutes_rattrapees": "Minutes rattrapées en vol",
    "Distance": "Distance (miles)",
}


def statistiques_descriptives(vols, variables=VARIABLES_DESCRIPTIVES):
    """Tableau descriptif d'une ou plusieurs variables numériques (une ligne par variable).

    Il suit une grille de lecture classique : centre (moyenne, médiane), dispersion
    (écart-type, IQR), forme (moyenne - médiane > 0 : asymétrie à droite), extrêmes
    (centiles 90 et 99, valeurs atypiques selon la règle IQR : hors de [Q1 - 1,5 IQR ; Q3 + 1,5 IQR]).
    """
    lignes = {}
    for colonne, nom in variables.items():
        x = vols[colonne]
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lignes[nom] = {
            "n": int(x.notna().sum()),
            "manquants": int(x.isna().sum()),
            "moyenne": x.mean(),
            "médiane": x.median(),
            "écart-type": x.std(),
            "min": x.min(),
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "P90": x.quantile(0.90),
            "P99": x.quantile(0.99),
            "max": x.max(),
            "moyenne - médiane": x.mean() - x.median(),
            "atypiques (%)": ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() * 100,
        }
    return pd.DataFrame(lignes).T.round(1)


def frequences(vols, colonne):
    """Effectifs et pourcentages d'une variable catégorielle, de la plus fréquente à la plus rare."""
    effectifs = vols[colonne].value_counts()
    return pd.DataFrame({"effectif": effectifs, "pourcentage": (effectifs / len(vols) * 100).round(1)})


def marge_erreur_95(proportion, n):
    """Marge d'erreur à 95 % d'un pourcentage mesuré sur n vols (en points de pourcentage).

    Formule de l'erreur type d'une proportion : 1,96 × √(p(1 - p) / n).
    Elle dit de combien le pourcentage pourrait varier si on observait d'autres vols
    comparables : plus n est petit, plus la marge est grande.
    """
    return 1.96 * (proportion * (1 - proportion) / n) ** 0.5 * 100


def resume_qualite(vols):
    """Tableau résumé de la qualité des données (nombre de lignes, colonnes, types, manquants, doublons)."""
    types = vols.dtypes
    return pd.Series({
        "Nombre de lignes": len(vols),
        "Nombre de colonnes": vols.shape[1],
        "Variables numériques": int(types.apply(pd.api.types.is_numeric_dtype).sum() - (types == bool).sum()),  # noqa: E721 (comparaison élément par élément des dtypes)
        "Variables catégorielles (texte)": int((types == object).sum()),  # noqa: E721
        "Variables vrai/faux": int((types == bool).sum()),  # noqa: E721
        "Variables temporelles (dates)": int(types.apply(pd.api.types.is_datetime64_any_dtype).sum()),
        "Colonnes avec valeurs manquantes": int((vols.isna().sum() > 0).sum()),
        "Cellules manquantes (%)": round(vols.isna().mean().mean() * 100, 2),
        "Lignes en double": int(vols.duplicated().sum()),
    }, name="Résultat")


def vols_partis(vols):
    """Garde seulement les vols qui ont décollé (sans les annulés).

    Pour mesurer un retard au départ, il faut que le vol soit parti.
    """
    return vols[~vols["Cancelled"]]


def taux_de_retard_par(vols, colonne):
    """Pour chaque valeur de `colonne` : nombre de vols, % en retard (> 15 min) et retard médian.

    Le % de vols en retard est l'indicateur principal du projet : il est plus parlant
    et moins sensible aux retards extrêmes que la moyenne.
    """
    partis = vols_partis(vols)
    return partis.groupby(colonne).agg(
        nb_vols=("DepDel15", "size"),
        pct_retard=("DepDel15", lambda s: s.mean() * 100),
        retard_median_min=("DepDelay", "median"),
    ).round(1)


def effet_domino(vols):
    """% de vols en retard selon que le vol précédent du même avion (même jour) était en retard ou non."""
    partis = vols_partis(vols)
    partis = partis[partis["vol_precedent_en_retard"].notna()]
    tableau = taux_de_retard_par(partis, "vol_precedent_en_retard")
    tableau.index = tableau.index.map({0.0: "Vol précédent à l'heure", 1.0: "Vol précédent en retard"})
    return tableau


def tableau_croise_retard(vols, lignes, colonnes):
    """Tableau croisé du % de vols en retard (ex. : lignes = jour, colonnes = heure)."""
    partis = vols_partis(vols)
    return partis.pivot_table(index=lignes, columns=colonnes, values="DepDel15", aggfunc="mean").mul(100).round(1)


def indicateurs_cles(vols):
    """KPI principaux pour le dashboard et la présentation."""
    partis = vols_partis(vols)
    return {
        "nb_vols": len(vols),
        "pct_retard": partis["DepDel15"].mean() * 100,
        "retard_median_min": partis["DepDelay"].median(),
        "retard_moyen_des_vols_en_retard": partis.loc[partis["DepDel15"] == 1, "DepDelay"].mean(),
        "pct_annules": vols["Cancelled"].mean() * 100,
    }

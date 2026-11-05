"""Toutes les figures du projet, produites par une fonction chacune.

Le notebook, le rapport et le guide utilisent les MÊMES images (reports/figures/),
générées ici : une figure ne peut donc pas exister en deux versions différentes.
Chaque fonction reçoit les données, applique la charte (visualizations.py) et renvoie la figure.

    python src/figures.py     # régénère toutes les figures
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import effet_domino, frequences, tableau_croise_retard, vols_partis
from comparison import (comparaison_brut_standardise, correlation_globale_et_par_groupe,
                        effet_domino_par_compagnie, profil_compagnies, taux_par_compagnie_et_periode)
from matplotlib.patches import Patch

from preprocessing import CLASSES_RETARD
from visualizations import (BLEU, COULEURS_TYPES, ORANGE, ENCRE, ENCRE_SECONDAIRE, GRIS_CONTEXTE, GRIS_FONCE,
                            PALETTE_DIVERGENTE, PALETTE_SEQUENTIELLE, ROUGE_RETARD, appliquer_style,
                            enregistrer_figure, format_pourcentage, nombre_fr, titrer)

ORDRE_SAISONS = ["Automne", "Hiver", "Printemps", "Été"]  # dans l'ordre de la période (août → juillet)


def nom_court(compagnie):
    return "Endeavor" if compagnie.startswith("Endeavor") else compagnie


MOIS_ABREGES = {1: "janv.", 2: "févr.", 3: "mars", 4: "avr.", 5: "mai", 6: "juin", 7: "juil.", 8: "août",
                9: "sept.", 10: "oct.", 11: "nov.", 12: "déc."}


def etiquette_mois(annee_mois):
    """« 2021-08 » → « août 21 » (abréviations françaises usuelles)."""
    annee, mois = annee_mois.split("-")
    return f"{MOIS_ABREGES[int(mois)]} {annee[2:]}"


# --------------------------------------------------------------------------- Univarié (Q1)

def f01_classes_retard(vols):
    """Gravité des retards : 5 classes ordonnées (couleur = valeur, de clair à foncé)."""
    parts = frequences(vols, "classe_retard")["pourcentage"].reindex(CLASSES_RETARD)
    couleurs = ["#d9d8d0", "#f4c7a1", "#e8834f", ROUGE_RETARD, GRIS_FONCE]
    fig, axe = plt.subplots(figsize=(9, 3.6))
    barres = axe.barh(parts.index[::-1], parts.values[::-1], color=couleurs[::-1], height=0.65)
    axe.bar_label(barres, labels=[f"{nombre_fr(v, 1)} %" for v in parts.values[::-1]], padding=4,
                  color=ENCRE_SECONDAIRE, fontsize=9)
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    format_pourcentage(axe, "x")
    axe.set_xlim(0, parts.max() * 1.15)
    titrer(axe, f"{nombre_fr(parts.iloc[0], 0)} % des vols partent à l'heure ; "
           f"{nombre_fr(parts.iloc[3], 0)} % ont plus de 45 minutes de retard", n=len(vols),
           precision="classes de l'auteur du dataset Kaggle")
    return fig


def f02_distribution_retard(vols):
    """Histogramme du retard au départ (coupé à 180 min, signalé), médiane et moyenne."""
    retards = vols_partis(vols)["DepDelay"].dropna()
    fig, axe = plt.subplots(figsize=(9, 4.2))
    axe.hist(retards.clip(upper=180), bins=np.arange(-30, 185, 5), color=GRIS_CONTEXTE, edgecolor="white")
    hauteur = axe.get_ylim()[1]
    for valeur, couleur, texte, position in [(retards.median(), ENCRE, "médiane", 0.92),
                                             (retards.mean(), ROUGE_RETARD, "moyenne", 0.78)]:
        axe.axvline(valeur, color=couleur, linewidth=2)
        axe.text(valeur + 3, hauteur * position, f"{texte} = {nombre_fr(valeur, 1)} min", color=couleur)
    axe.set_xlabel("Retard au départ (minutes ; négatif = parti en avance ; "
                   "les retards de plus de 180 min sont regroupés à droite)")
    axe.set_ylabel("Nombre de vols")
    axe.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: nombre_fr(v)))
    titrer(axe, "Distribution très asymétrique : la moyenne est tirée par quelques très gros retards",
           n=len(retards), unite="vols partis")
    return fig


# --------------------------------------------------------------------------- Temps (Q3, Q4)

def f03_retard_par_jour_saisons(vols):
    """Les 365 jours, avec les 4 saisons en fond et la moyenne de chaque saison (remplace la vue par mois)."""
    from approfondissement import serie_journaliere
    jour = serie_journaliere(vols)
    partis = vols_partis(vols)
    par_saison = partis.groupby("saison")["DepDel15"].mean() * 100
    saison_du_jour = partis.groupby("FlightDate")["saison"].first()
    fig, axe = plt.subplots(figsize=(11, 4.6))
    # Fond : une bande par saison (l'été est coupé en deux : août 2021, puis juin-juillet 2022)
    fonds = {"Automne": "#f6efe3", "Hiver": "#e9f0f7", "Printemps": "#eaf4ec", "Été": "#fbe9e9"}
    blocs = (saison_du_jour != saison_du_jour.shift()).cumsum()
    for _, bloc in saison_du_jour.groupby(blocs):
        saison, debut, fin = bloc.iloc[0], bloc.index[0], bloc.index[-1] + pd.Timedelta(days=1)
        axe.axvspan(debut, fin, color=fonds[saison], linewidth=0, zorder=0)
        if len(bloc) > 40:
            axe.text(debut + (fin - debut) / 2, 69, f"{saison}\n{nombre_fr(par_saison[saison], 1)} %",
                     ha="center", va="top", fontsize=9.5, fontweight="bold", color=ENCRE)
    axe.plot(jour.index, jour["pct_retard"], color=GRIS_CONTEXTE, linewidth=0.9, label="chaque jour", zorder=2)
    axe.plot(jour.index, jour["moyenne_7_jours"], color=ROUGE_RETARD, linewidth=2.2, label="moyenne sur 7 jours", zorder=3)
    axe.axhline(partis["DepDel15"].mean() * 100, color=ENCRE_SECONDAIRE, linestyle="--", linewidth=1,
                label=f"moyenne de l'année ({nombre_fr(partis['DepDel15'].mean() * 100, 1)} %)", zorder=2)
    axe.set_ylim(0, 72)
    format_pourcentage(axe)
    axe.set_ylabel("% de vols partis en retard")
    axe.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%y"))
    axe.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncols=3, fontsize=9, frameon=False)
    titrer(axe, f"Le retard change avec la saison (de {nombre_fr(par_saison.min(), 1)} % à {nombre_fr(par_saison.max(), 1)} %) "
           "et, jour par jour, arrive par crises",
           n=len(partis), unite="vols partis", precision="365 jours ; fond coloré = saison, chiffre = moyenne de la saison")
    return fig


def f04_retard_par_heure(vols):
    """% de vols en retard selon l'heure de départ prévue."""
    tableau = vols_partis(vols).groupby("heure_depart_prevue")["DepDel15"].agg(["mean", "size"])
    tableau = tableau[tableau["size"] >= 30]
    fig, axe = plt.subplots(figsize=(9, 4))
    axe.plot(tableau.index, tableau["mean"] * 100, color=ROUGE_RETARD, linewidth=2.2, marker="o")
    axe.set_xticks(tableau.index)
    axe.set_ylim(0, tableau["mean"].max() * 100 * 1.15)
    format_pourcentage(axe)
    axe.set_xlabel("Heure de départ prévue")
    axe.set_ylabel("% de vols partis en retard")
    titrer(axe, "Le retard s'accumule au fil de la journée", n=int(tableau["size"].sum()), unite="vols partis",
           precision="heures avec au moins 30 vols")
    return fig


# --------------------------------------------------------------------------- Compagnies (Q1)

def f05_profil_compagnies(vols):
    """Carte des profils : compagnies × indicateurs. Couleur = écart à la moyenne des compagnies
    (en écarts-types), texte = valeur réelle. Rouge = moins bon que la moyenne."""
    profil = profil_compagnies(vols)
    indicateurs = {
        "pct_retard_ou_annule": ("Retardés ou\nannulés (%)", 1),
        "pct_annules": ("Annulés (%)", 1),
        "pct_grand_retard": ("Grands retards\n> 45 min (%)", 1),
        "retard_p90_min": ("Retard du 90e\ncentile (min)", 1),
        "roulage_median_min": ("Roulage\nmédian (min)", 1),
        "rattrapage_median_min": ("Rattrapé en\nvol (min)", -1),  # rattraper plus = mieux
        "part_vols_soir_pct": ("Vols après\n18 h (%)", 0),  # composition, pas performance : non coloré
    }
    valeurs = profil[list(indicateurs)]
    ecarts = (valeurs - valeurs.mean()) / valeurs.std()
    ecarts = ecarts * pd.Series({k: s for k, (_, s) in indicateurs.items()})
    lignes = [f"{nom_court(c)}  ({profil.loc[c, 'type']}, n = {nombre_fr(profil.loc[c, 'nb_vols'])})"
              for c in profil.index]

    fig, axe = plt.subplots(figsize=(11, 7))
    image = axe.imshow(ecarts.to_numpy(), cmap=PALETTE_DIVERGENTE, vmin=-2.5, vmax=2.5, aspect="auto")
    for i in range(len(valeurs)):
        for j in range(len(indicateurs)):
            couleur = "white" if abs(ecarts.iloc[i, j]) > 1.6 else ENCRE
            axe.text(j, i, nombre_fr(valeurs.iloc[i, j], 1), ha="center", va="center", fontsize=8.5, color=couleur)
    axe.set_xticks(range(len(indicateurs)), [nom for nom, _ in indicateurs.values()], fontsize=8.5)
    axe.set_yticks(range(len(lignes)), lignes, fontsize=8.5)
    axe.grid(False)
    barre = fig.colorbar(image, ax=axe, shrink=0.6)
    barre.set_label("Écart à la moyenne des compagnies (écarts-types)\nrouge = moins bon, bleu = meilleur\n"
                    "(« Vols après 18 h » : contexte, non coloré)")
    titrer(axe, "Les compagnies à bas coût cumulent plus de retards, et des retards plus graves",
           n=len(vols), precision="compagnies triées du moins au plus perturbé")
    return fig


def f06_compagnies_retard_ou_annule(vols):
    """% de vols retardés OU annulés par compagnie, couleur = type, barre d'erreur = marge à 95 %."""
    profil = profil_compagnies(vols).sort_values("pct_retard_ou_annule")
    p = profil["pct_retard_ou_annule"] / 100
    marge = 1.96 * np.sqrt(p * (1 - p) / profil["nb_vols"]) * 100
    fig, axe = plt.subplots(figsize=(9, 6))
    axe.barh([f"{nom_court(c)} (n = {nombre_fr(n)})" for c, n in profil["nb_vols"].items()],
             profil["pct_retard_ou_annule"], xerr=marge, color=[COULEURS_TYPES[t] for t in profil["type"]],
             height=0.65, error_kw=dict(ecolor=ENCRE_SECONDAIRE, capsize=2, linewidth=1))
    axe.legend(handles=[Patch(color=c, label=t) for t, c in COULEURS_TYPES.items()], loc="lower right",
               title="Type de compagnie")
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    format_pourcentage(axe, "x")
    axe.set_xlabel("% de vols programmés retardés (> 15 min) ou annulés — trait : marge d'erreur à 95 %")
    meilleure, pire = profil.index[0], profil.index[-1]
    titrer(axe, f"De {nombre_fr(profil.iloc[0]['pct_retard_ou_annule'], 0)} % ({nom_court(meilleure)}) à "
           f"{nombre_fr(profil.iloc[-1]['pct_retard_ou_annule'], 0)} % ({nom_court(pire)}) de vols perturbés "
           "selon la compagnie", n=len(vols),
           unite="vols programmés")
    return fig


def f07_composition(vols):
    """Brut contre standardisé : à tranche horaire égale (gauche), à mois égal (droite)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    ordre = comparaison_brut_standardise(vols, "tranche_horaire").sort_values("pct_retard_brut").index
    for axe, strate, titre in [(axes[0], "tranche_horaire", "À tranche horaire égale"),
                               (axes[1], "annee_mois", "À mois égal")]:
        tableau = comparaison_brut_standardise(vols, strate).reindex(ordre)
        y = np.arange(len(tableau))
        axe.hlines(y, tableau["pct_retard_brut"], tableau["taux_standardise_pct"], color=GRIS_CONTEXTE, linewidth=2)
        axe.scatter(tableau["pct_retard_brut"], y, color=GRIS_FONCE, zorder=3, label="Brut (tel qu'observé)")
        axe.scatter(tableau["taux_standardise_pct"], y, color=ROUGE_RETARD, zorder=3,
                    label="Standardisé (même composition pour tous)")
        for i, (compagnie, ligne) in enumerate(tableau.iterrows()):
            if abs(ligne["ecart_points"]) >= 2:
                axe.text(max(ligne["pct_retard_brut"], ligne["taux_standardise_pct"]) + 0.8, i,
                         f"{ligne['ecart_points']:+.1f} pts".replace(".", ","), va="center", fontsize=8,
                         color=ROUGE_RETARD)
        axe.set_yticks(y, [nom_court(c) for c in tableau.index])
        axe.set_title(titre, loc="left", fontsize=11)
        format_pourcentage(axe, "x")
        axe.grid(axis="y", visible=False)
        axe.grid(axis="x", visible=True)
        axe.set_xlabel("% de vols partis en retard")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Effet de composition : une partie de l'écart vient de QUAND les compagnies volent",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.92, "Gauche : American et United volent surtout le matin. Droite : SkyWest vole surtout "
             "à l'été 2022. Standardisation directe (poids de la population).", fontsize=8.5,
             color=ENCRE_SECONDAIRE)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def f08_compagnies_par_saison(vols):
    """% de retard compagnie × saison (heatmap séquentielle), n minimum signalé."""
    taux, _ = taux_par_compagnie_et_periode(vols, "saison")
    taux = taux[ORDRE_SAISONS]
    taux = taux.loc[taux.mean(axis=1).sort_values().index]
    fig, axe = plt.subplots(figsize=(8, 6.5))
    image = axe.imshow(taux.to_numpy(), cmap=PALETTE_SEQUENTIELLE, aspect="auto", vmin=0, vmax=40)
    for i in range(taux.shape[0]):
        for j in range(taux.shape[1]):
            valeur = taux.iloc[i, j]
            axe.text(j, i, nombre_fr(valeur, 1), ha="center", va="center", fontsize=8.5,
                     color="white" if valeur > 28 else ENCRE)
    axe.set_xticks(range(4), ["Automne 21", "Hiver 21-22", "Printemps 22", "Été 22"])
    axe.set_yticks(range(len(taux)), [nom_court(c) for c in taux.index])
    axe.grid(False)
    fig.colorbar(image, ax=axe, shrink=0.7, label="% de vols partis en retard")
    effectif_min = vols_partis(vols).groupby(["compagnie", "saison"]).size().min()
    titrer(axe, "Le classement des compagnies change selon la saison", n=len(vols_partis(vols)),
           unite="vols partis", precision=f"au moins {nombre_fr(effectif_min)} vols par case")
    return fig


def f09_roulage_retard_simpson(vols):
    """Corrélation roulage-retard : globale (≈ 0) contre à l'intérieur de chaque compagnie."""
    globale, par_compagnie = correlation_globale_et_par_groupe(vols, "TaxiOut", "DepDelayMinutes")
    par_compagnie = par_compagnie.sort_values("correlation")
    fig, axe = plt.subplots(figsize=(9, 5.5))
    couleurs = [ROUGE_RETARD if r > 0 else GRIS_FONCE for r in par_compagnie["correlation"]]
    axe.scatter(par_compagnie["correlation"], range(len(par_compagnie)), color=couleurs, s=40, zorder=3)
    axe.axvline(0, color=ENCRE_SECONDAIRE, linewidth=0.8)
    axe.axvline(globale, color=ENCRE, linestyle="--", linewidth=1.5)
    axe.text(globale, -0.9, f"  toutes compagnies mélangées : ρ = {nombre_fr(globale, 3)}", fontsize=8.5,
             va="top")
    axe.set_yticks(range(len(par_compagnie)),
                   [f"{nom_court(c)} (n = {nombre_fr(n)})" for c, n in par_compagnie["n"].items()])
    axe.set_xlabel("Corrélation de Spearman entre roulage au départ et retard au départ")
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    titrer(axe, "Mélanger les compagnies masque une relation présente dans presque chacune",
           n=int(par_compagnie["n"].sum()), unite="vols partis", precision="effet de composition de type Simpson")
    return fig


def f10_domino(vols):
    """Effet domino : global et par compagnie (au moins 30 cas), avec n."""
    global_ = effet_domino(vols)["pct_retard"]
    tableau = effet_domino_par_compagnie(vols).sort_values("ecart_points")
    fig, axe = plt.subplots(figsize=(9, 5))
    y = np.arange(len(tableau))
    axe.hlines(y, tableau["pct_si_precedent_a_l_heure"], tableau["pct_si_precedent_en_retard"],
               color=GRIS_CONTEXTE, linewidth=3)
    axe.scatter(tableau["pct_si_precedent_a_l_heure"], y, color=GRIS_FONCE, zorder=3, label="Vol précédent à l'heure")
    axe.scatter(tableau["pct_si_precedent_en_retard"], y, color=ROUGE_RETARD, zorder=3,
                label="Vol précédent en retard")
    axe.set_yticks(y, [f"{nom_court(c)} ({nombre_fr(n)} cas)" for c, n in tableau["n_vol_precedent_en_retard"].items()])
    format_pourcentage(axe, "x")
    axe.set_xlim(0, 85)
    axe.legend(loc="lower right")
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    axe.set_xlabel("% de vols partis en retard ; « cas » = vols dont le vol précédent du même avion était en retard")
    titrer(axe, f"Effet domino : {nombre_fr(global_.iloc[0], 1)} % → {nombre_fr(global_.iloc[1], 1)} % de retard "
           "selon le vol précédent du même avion", n=int(tableau["n_vol_precedent_en_retard"].sum()), unite="cas",
           precision="compagnies avec au moins 30 cas")
    return fig


def f11_jour_heure(vols):
    """% de retard jour de la semaine × heure (heatmap séquentielle)."""
    from preprocessing import NOMS_JOURS
    carte = tableau_croise_retard(vols, "jour_semaine", "heure_depart_prevue").reindex(list(NOMS_JOURS.values()))
    carte = carte.loc[:, [h for h in carte.columns if 6 <= h <= 23]]
    fig, axe = plt.subplots(figsize=(11, 4))
    image = axe.imshow(carte.to_numpy(), cmap=PALETTE_SEQUENTIELLE, aspect="auto", vmin=0, vmax=40)
    axe.set_xticks(range(carte.shape[1]), carte.columns)
    axe.set_yticks(range(7), carte.index)
    axe.grid(False)
    axe.set_xlabel("Heure de départ prévue")
    fig.colorbar(image, ax=axe, label="% de vols partis en retard")
    titrer(axe, "Le soir est plus à risque que le matin, tous les jours de la semaine",
           n=len(vols_partis(vols)), unite="vols partis")
    return fig


# --------------------------------------------------------------------------- Univarié catégoriel et bivarié numérique

TAILLE_NUAGE = 5000  # points affichés dans un nuage (sous-échantillonnage informatique, pour la lisibilité)


def f15_vols_par_compagnie(vols):
    """Répartition des vols par compagnie (univarié catégoriel), couleur = type."""
    parts = vols.groupby("compagnie").agg(n=("compagnie", "size"), type=("type_compagnie", "first"))
    parts["pct"] = parts["n"] / len(vols) * 100
    parts = parts.sort_values("pct")
    fig, axe = plt.subplots(figsize=(9, 5.5))
    barres = axe.barh([nom_court(c) for c in parts.index], parts["pct"],
                      color=[COULEURS_TYPES[t] for t in parts["type"]], height=0.65)
    axe.bar_label(barres, labels=[f"{nombre_fr(p, 1)} % ({nombre_fr(n)})" for p, n in zip(parts["pct"], parts["n"])],
                  padding=4, fontsize=8.5, color=ENCRE_SECONDAIRE)
    axe.legend(handles=[Patch(color=c, label=t) for t, c in COULEURS_TYPES.items()], loc="lower right",
               title="Type de compagnie")
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    format_pourcentage(axe, "x")
    axe.set_xlim(0, parts["pct"].max() * 1.25)
    axe.set_xlabel("% des vols programmés au départ d'Atlanta")
    delta = parts.loc["Delta", "pct"]
    titrer(axe, f"Delta opère {nombre_fr(delta, 0)} % des vols : tout chiffre global reflète surtout Delta",
           n=len(vols))
    return fig


def f12_depart_arrivee(vols):
    """Nuage retard au départ × retard à l'arrivée (5 000 points affichés), droite y = x, r sur tous les vols."""
    donnees = vols_partis(vols)[["DepDelay", "ArrDelay"]].dropna()
    pearson = donnees["DepDelay"].corr(donnees["ArrDelay"])
    spearman = donnees["DepDelay"].corr(donnees["ArrDelay"], method="spearman")
    points = donnees.sample(n=TAILLE_NUAGE, random_state=42)
    fig, axe = plt.subplots(figsize=(7.5, 6.5))
    axe.scatter(points["DepDelay"], points["ArrDelay"], s=6, alpha=0.3, color=GRIS_FONCE, linewidths=0)
    axe.plot([-40, 300], [-40, 300], color=ROUGE_RETARD, linewidth=1.5)
    axe.text(230, 250, "arrivée = départ", color=ROUGE_RETARD, rotation=40, fontsize=9)
    axe.set_xlim(-40, 300)
    axe.set_ylim(-70, 300)
    axe.set_xlabel("Retard au départ (min)")
    axe.set_ylabel("Retard à l'arrivée (min)")
    axe.grid(axis="x", visible=True)
    titrer(axe, "Le retard au départ se retrouve presque entièrement à l'arrivée", n=len(donnees),
           unite="vols partis", precision=f"Pearson r = {nombre_fr(pearson, 2)}, Spearman ρ = {nombre_fr(spearman, 2)} ; "
                                          f"{nombre_fr(TAILLE_NUAGE)} points affichés")
    return fig


def f13_distance_rattrapage(vols):
    """Nuage distance × minutes rattrapées en vol, avec la médiane par tranche de distance."""
    donnees = vols_partis(vols)[["Distance", "minutes_rattrapees"]].dropna()
    spearman = donnees["Distance"].corr(donnees["minutes_rattrapees"], method="spearman")
    points = donnees.sample(n=TAILLE_NUAGE, random_state=42)
    tranches = pd.cut(donnees["Distance"], bins=[0, 250, 500, 750, 1000, 1500, 2000, 5000])
    medianes = donnees.groupby(tranches, observed=True)["minutes_rattrapees"].median()
    centres = [(i.left + min(i.right, 2600)) / 2 for i in medianes.index]
    fig, axe = plt.subplots(figsize=(9, 5))
    axe.scatter(points["Distance"], points["minutes_rattrapees"], s=6, alpha=0.25, color=GRIS_FONCE, linewidths=0)
    axe.plot(centres, medianes.values, color=ROUGE_RETARD, linewidth=2.2, marker="o", label="médiane par tranche de distance")
    axe.axhline(0, color=ENCRE_SECONDAIRE, linewidth=0.8)
    axe.set_ylim(-60, 60)
    axe.set_xlim(0, 2600)
    axe.legend(loc="lower right")
    axe.set_xlabel("Distance du vol (miles)")
    axe.set_ylabel("Minutes rattrapées en vol\n(retard départ − retard arrivée)")
    axe.grid(axis="x", visible=True)
    titrer(axe, "Plus le vol est long, plus l'avion rattrape de minutes en vol (relation faible)", n=len(donnees),
           unite="vols partis", precision=f"Spearman ρ = {nombre_fr(spearman, 2)} ; {nombre_fr(TAILLE_NUAGE)} points affichés, "
                                          "axe limité à ± 60 min")
    return fig


VARIABLES_CORRELATION = {"DepDelay": "Retard départ", "ArrDelay": "Retard arrivée", "TaxiOut": "Roulage départ",
                         "TaxiIn": "Roulage arrivée", "Distance": "Distance", "minutes_rattrapees": "Rattrapé en vol"}


def f14_correlations(vols):
    """Matrice de corrélation de Spearman sur 6 variables choisies (pas toutes les colonnes)."""
    matrice = vols_partis(vols)[list(VARIABLES_CORRELATION)].corr(method="spearman")
    matrice.index = matrice.columns = list(VARIABLES_CORRELATION.values())
    fig, axe = plt.subplots(figsize=(7.5, 6))
    image = axe.imshow(matrice.to_numpy(), cmap=PALETTE_DIVERGENTE, vmin=-1, vmax=1)
    for i in range(len(matrice)):
        for j in range(len(matrice)):
            valeur = matrice.iloc[i, j]
            axe.text(j, i, nombre_fr(valeur, 2), ha="center", va="center", fontsize=9,
                     color="white" if abs(valeur) > 0.6 else ENCRE)
    axe.set_xticks(range(len(matrice)), matrice.columns, rotation=35, ha="right")
    axe.set_yticks(range(len(matrice)), matrice.index)
    axe.grid(False)
    fig.colorbar(image, ax=axe, shrink=0.75, label="ρ de Spearman")
    titrer(axe, "Les retards et le rattrapage sont liés entre eux ; la distance l'est peu",
           n=len(vols_partis(vols)), unite="vols partis", precision="corrélation de Spearman (rangs)")
    return fig


# --------------------------------------------------------------------------- Approfondissement (deuxième itération)

def evenements_numerotes():
    """Numéro et description de chaque événement réel (contexte.py), pour les annoter sur les figures."""
    from contexte import EVENEMENTS
    return [(i + 1, ev) for i, ev in enumerate(EVENEMENTS)]


def f16_retard_par_jour(vols):
    """Les 365 jours : % de retard (en haut) et % d'annulations (en bas), avec les événements réels numérotés."""
    from approfondissement import serie_journaliere
    jour = serie_journaliere(vols)
    fig, (haut, bas) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    haut.plot(jour.index, jour["pct_retard"], color=GRIS_CONTEXTE, linewidth=0.9, label="chaque jour")
    haut.plot(jour.index, jour["moyenne_7_jours"], color=ROUGE_RETARD, linewidth=2.2, label="moyenne sur 7 jours")
    partis = vols_partis(vols)
    mois = partis.groupby(partis["FlightDate"].dt.to_period("M"))["DepDel15"].mean() * 100
    debuts = [p.start_time for p in mois.index] + [jour.index.max() + pd.Timedelta(days=1)]
    haut.step(debuts, list(mois.values) + [mois.values[-1]], where="post", color=ENCRE_SECONDAIRE,
              linewidth=1.2, linestyle="--", label="moyenne du mois (la vue précédente)")
    haut.set_ylim(0, 74)
    format_pourcentage(haut)
    haut.set_ylabel("% de vols en retard")
    bas.bar(jour.index, jour["pct_annules"], width=1.0,
            color=[ROUGE_RETARD if a > 5 else GRIS_CONTEXTE for a in jour["pct_annules"]])
    bas.set_ylim(0, 36)
    format_pourcentage(bas)
    bas.set_ylabel("% annulés")
    for numero, ev in evenements_numerotes():
        debut, fin = pd.Timestamp(ev["debut"]), pd.Timestamp(ev["fin"])
        for axe in (haut, bas):
            axe.axvspan(debut - pd.Timedelta(hours=12), fin + pd.Timedelta(hours=12), color=ROUGE_RETARD,
                        alpha=0.08, linewidth=0)
        haut.annotate(str(numero), (debut + (fin - debut) / 2, 68), ha="center", fontsize=8, fontweight="bold",
                      color=ENCRE, bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=ROUGE_RETARD, lw=0.8))
    bas.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%y"))
    fig.legend(*haut.get_legend_handles_labels(), loc="lower center", ncols=3, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    titrer(haut, "Jour par jour, le retard va de 3 % à 64 % : des crises courtes que la moyenne du mois efface",
           n=len(vols), unite="vols programmés", precision="365 jours ; numéros = événements réels documentés")
    return fig


def f17_jour_semaine(vols):
    """% de retard selon le jour de la semaine (et nombre moyen de vols ce jour-là)."""
    from approfondissement import par_jour_semaine
    tableau = par_jour_semaine(vols)["tableau"]
    fig, axe = plt.subplots(figsize=(9, 4))
    couleurs = [BLEU if j in ("Mardi", "Mercredi") else ROUGE_RETARD if j == "Vendredi" else GRIS_CONTEXTE
                for j in tableau.index]
    barres = axe.bar(tableau.index, tableau["pct_retard"], color=couleurs, width=0.65)
    axe.bar_label(barres, labels=[f"{nombre_fr(t, 1)} %\n{nombre_fr(n)} vols/jour" for t, n in
                                  zip(tableau["pct_retard"], tableau["vols_par_jour"])],
                  padding=3, fontsize=8.5, color=ENCRE_SECONDAIRE)
    axe.set_ylim(0, tableau["pct_retard"].max() * 1.35)
    format_pourcentage(axe)
    axe.set_ylabel("% de vols partis en retard")
    titrer(axe, "Mardi et mercredi sont les jours les plus calmes, vendredi le plus difficile",
           n=len(vols_partis(vols)), unite="vols partis", precision="sur chaque barre : vols par jour en moyenne")
    return fig


def f18_queue_retard_compagnies(vols):
    """Catégorielle × numérique : médiane, 3e quartile et 90e centile du retard (minutes) par compagnie."""
    from approfondissement import quantiles_retard_par_compagnie
    tableau = quantiles_retard_par_compagnie(vols)
    fig, axe = plt.subplots(figsize=(9, 6))
    y = np.arange(len(tableau))
    couleurs = [COULEURS_TYPES[t] for t in tableau["type"]]
    axe.hlines(y, tableau["mediane"], tableau["p90"], color=couleurs, linewidth=3)
    axe.scatter(tableau["mediane"], y, color=ENCRE, s=22, zorder=3, label="médiane : 50 % des vols en dessous")
    axe.scatter(tableau["q3"], y, color="white", edgecolor=ENCRE, s=28, zorder=3, label="3e quartile : 75 %")
    axe.scatter(tableau["p90"], y, color=couleurs, edgecolor=ENCRE, s=55, zorder=3, label="90e centile : 90 %")
    for yi, p90 in zip(y, tableau["p90"]):
        axe.text(p90 + 1.5, yi, f"{nombre_fr(p90)} min", va="center", fontsize=8.5, color=ENCRE_SECONDAIRE)
    axe.set_yticks(y, [nom_court(c) for c in tableau.index])
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    axe.set_xlim(-2, tableau["p90"].max() * 1.15)
    axe.set_xlabel("Retard au départ (minutes ; 0 = à l'heure ou en avance)")
    axe.legend(loc="lower right", fontsize=8.5)
    titrer(axe, "Le vol typique part à l'heure partout ; les compagnies diffèrent par leurs pires retards",
           n=len(vols_partis(vols)), unite="vols partis", precision="couleur = type de compagnie")
    return fig


def f19_classes_par_type(vols):
    """Catégorielle × catégorielle : répartition des classes de retard dans chaque type (barres à 100 %)."""
    from approfondissement import association_categorielle, classes_par_type
    tableau = classes_par_type(vols).reindex(["Régionale", "Majeure", "Bas coût"])[CLASSES_RETARD]
    couleurs = ["#e9e8e2", "#f6c9c4", "#e07b73", ROUGE_RETARD, GRIS_FONCE]  # gravité croissante ; annulé à part
    texte_clair = {ROUGE_RETARD, GRIS_FONCE, "#e07b73"}
    fig, axe = plt.subplots(figsize=(10, 3.4))
    gauche = np.zeros(len(tableau))
    for classe, couleur in zip(CLASSES_RETARD, couleurs):
        valeurs = tableau[classe].to_numpy()
        axe.barh(tableau.index, valeurs, left=gauche, color=couleur, edgecolor="white", linewidth=1.5,
                 height=0.6, label=classe)
        for yi, (g, v) in enumerate(zip(gauche, valeurs)):
            if v >= 4:
                axe.text(g + v / 2, yi, f"{nombre_fr(v, 0)} %", ha="center", va="center", fontsize=9,
                         color="white" if couleur in texte_clair else ENCRE)
        gauche += valeurs
    axe.set_xlim(0, 100)
    format_pourcentage(axe, "x")
    axe.invert_yaxis()
    axe.grid(False)
    axe.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    v = association_categorielle(pd.crosstab(vols["type_compagnie"], vols["classe_retard"]))["v_cramer"]
    a_l_heure = tableau.iloc[:, 0]
    titrer(axe, f"Régionales : {nombre_fr(a_l_heure['Régionale'], 0)} % de vols à l'heure ; bas coût : "
           f"{nombre_fr(a_l_heure['Bas coût'], 0)} % (V de Cramér = {nombre_fr(v, 2)})",
           n=len(vols), unite="vols programmés")
    return fig


def f20_annulations_compagnie_semaine(vols):
    """% de vols annulés par compagnie et par semaine : les crises propres à chaque compagnie."""
    semaine = vols["FlightDate"].dt.to_period("W-SUN").dt.start_time
    carte = vols.pivot_table(index="compagnie", columns=semaine, values="Cancelled", aggfunc="mean") * 100
    effectifs = vols.pivot_table(index="compagnie", columns=semaine, values="Cancelled", aggfunc="size")
    carte = carte.where(effectifs >= 10)
    carte = carte.loc[carte.mean(axis=1).sort_values().index]
    fig, axe = plt.subplots(figsize=(12, 5.2))
    palette = plt.get_cmap("Reds").copy()
    palette.set_bad("#e1e0d9")  # case grise = moins de 10 vols cette semaine (pas de valeur)
    image = axe.imshow(carte.to_numpy(dtype=float), cmap=palette, aspect="auto", vmin=0, vmax=40,
                       interpolation="nearest")
    axe.set_yticks(range(len(carte)), [nom_court(c) for c in carte.index])
    positions = [i for i, d in enumerate(carte.columns) if d.day <= 7]
    axe.set_xticks(positions, [carte.columns[i].strftime("%m/%y") for i in positions])
    axe.grid(False)
    fig.colorbar(image, ax=axe, label="% de vols annulés dans la semaine", shrink=0.85)
    titrer(axe, "Chaque compagnie a ses propres crises : Spirit en août, Southwest en octobre, JetBlue en janvier",
           n=len(vols), unite="vols programmés", precision="gris = moins de 10 vols dans la semaine")
    return fig


def f21_force_des_liens(vols):
    """Bivarié : quelle variable est la plus liée au retard ? (mesure adaptée au type des deux variables)"""
    from approfondissement import liens_bivaries
    # seulement les liens AVEC le retard (départ × arrivée est trivial ; distance × rattrapage ne parle pas du retard)
    liens = {k: v for k, v in liens_bivaries(vols).items()
             if k not in ("retard départ × retard arrivée", "distance × minutes rattrapées")}
    tableau = pd.Series({k: abs(v["valeur"]) for k, v in liens.items()}).sort_values()
    mesures = {k: liens[k]["mesure"] for k in tableau.index}
    couleurs = {"V de Cramér": BLEU, "rho de Spearman": GRIS_FONCE, "epsilon² (Kruskal-Wallis)": ORANGE}
    fig, axe = plt.subplots(figsize=(9, 4.3))
    barres = axe.barh(tableau.index, tableau.values, color=[couleurs[mesures[k]] for k in tableau.index], height=0.6)
    axe.bar_label(barres, labels=[nombre_fr(v, 3) for v in tableau.values], padding=3, fontsize=8.5,
                  color=ENCRE_SECONDAIRE)
    axe.legend(handles=[Patch(color=c, label=m) for m, c in couleurs.items()], loc="lower right", fontsize=8.5)
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    axe.set_xlim(0, tableau.max() * 1.25)
    axe.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: nombre_fr(x, 2)))
    axe.set_xlabel("Force du lien (0 = aucun ; toutes les p-valeurs < 0,001 avec environ 300 000 vols)")
    titrer(axe, "L'heure de départ est aussi liée au retard que le type de compagnie",
           n=len(vols), unite="vols", precision="mesure adaptée au type des deux variables ; ordres de grandeur comparables, pas égaux")
    return fig


# --------------------------------------------------------------------------- L'équipe : météo, modèle, destinations, histoire

def f22_meteo_retard(vols):
    """Fusion vols × météo : % de retard du jour selon les heures de pluie à Atlanta (365 jours)."""
    from base_de_donnees import lire
    j = lire("jour")
    fig, axe = plt.subplots(figsize=(10, 5))
    axe.scatter(j["heures_de_pluie"], j["pct_retard"], s=18, color=GRIS_FONCE, alpha=0.55, linewidth=0)
    moyennes = j.groupby(pd.cut(j["heures_de_pluie"], [-0.1, 0, 3, 6, 12, 24]), observed=True)["pct_retard"].mean()
    centres = [0, 1.5, 4.5, 9, 18]
    axe.plot(centres[:len(moyennes)], moyennes.values, color=ROUGE_RETARD, linewidth=2.4, marker="o",
             label="moyenne par classe d'heures de pluie")
    for date, texte in [("2022-01-16", "16 janv. : neige (Izzy)"), ("2022-03-12", "12 mars : rafales 77 km/h"),
                        ("2022-07-21", "21 juil. : 11 h de pluie"), ("2022-03-18", "18 mars : orages")]:
        ligne = j[j["date_vol"] == date].iloc[0]
        axe.annotate(texte, (ligne["heures_de_pluie"], ligne["pct_retard"]), xytext=(8, 4), textcoords="offset points",
                     fontsize=8.5, color=ENCRE)
        axe.scatter([ligne["heures_de_pluie"]], [ligne["pct_retard"]], s=40, color=ROUGE_RETARD, zorder=3)
    format_pourcentage(axe)
    axe.set_xlabel("Heures de pluie dans la journée à Atlanta (Open-Meteo)")
    axe.set_ylabel("% de vols partis en retard ce jour-là")
    axe.legend(loc="lower right", fontsize=9)
    rho = j["heures_de_pluie"].corr(j["pct_retard"], method="spearman")
    titrer(axe, f"Plus il pleut à Atlanta, plus les vols partent en retard (Spearman ρ = {nombre_fr(rho, 2)})",
           n=len(j), unite="jours", precision="fusion vols × météo journalière")
    return fig


def f23_modele_rapports(vols):
    """Data scientist : rapports de cotes (odds ratios) de la régression logistique, à conditions égales."""
    from results import charger_resultats
    rapports = pd.DataFrame(charger_resultats()["equipe"]["modele"]["rapports_de_cotes"]).T
    rapports = rapports[~rapports.index.str.startswith("Jour :")].sort_values("rapport_de_cotes")
    fig, axe = plt.subplots(figsize=(9.5, 7))
    y = np.arange(len(rapports))
    # gris = intervalle de confiance qui contient 1 : effet non établi
    couleurs = [GRIS_CONTEXTE if b <= 1 <= h else ROUGE_RETARD if v > 1 else BLEU
                for v, b, h in zip(rapports["rapport_de_cotes"], rapports["ic_bas"], rapports["ic_haut"])]
    axe.hlines(y, rapports["ic_bas"], rapports["ic_haut"], color=GRIS_CONTEXTE, linewidth=3)
    axe.scatter(rapports["rapport_de_cotes"], y, color=couleurs, s=40, zorder=3)
    axe.axvline(1, color=ENCRE_SECONDAIRE, linewidth=1)
    for yi, v in zip(y, rapports["rapport_de_cotes"]):
        axe.text(rapports["ic_haut"].max() * 1.08, yi, f"× {nombre_fr(v, 2)}", va="center", fontsize=8.5, color=ENCRE_SECONDAIRE)
    axe.set_xscale("log")
    axe.set_xticks([0.5, 1, 2, 3], ["0,5", "1", "2", "3"])
    axe.xaxis.set_minor_formatter(plt.NullFormatter())
    axe.set_yticks(y, rapports.index)
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    axe.set_xlabel("Rapport de cotes (échelle log) : > 1 = plus de retard que la référence, à conditions égales")
    titrer(axe, "À conditions égales : le soir et un avion déjà en retard triplent la cote du retard",
           n=314844, unite="vols partis",
           precision="régression logistique ; références : Delta, matin, automne, mardi, vol précédent à l'heure ; trait = IC 95 %")
    return fig


def f24_importance_facteurs(vols):
    """Data scientist : ce que le modèle perd quand on retire chaque bloc de facteurs."""
    from results import charger_resultats
    importance = pd.Series(charger_resultats()["equipe"]["modele"]["importance_pseudo_r2"]).sort_values()
    fig, axe = plt.subplots(figsize=(9, 3.8))
    barres = axe.barh(importance.index, importance.values, color=[ROUGE_RETARD if v == importance.max() else GRIS_FONCE
                                                                  for v in importance.values], height=0.6)
    axe.bar_label(barres, labels=[nombre_fr(v, 3) for v in importance.values], padding=3, fontsize=8.5,
                  color=ENCRE_SECONDAIRE)
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    axe.set_xticks(np.arange(0, importance.max() * 1.15, 0.005))
    axe.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: nombre_fr(x, 3)))
    axe.set_xlabel("Perte de pseudo-R² si l'on retire ce facteur du modèle (plus grand = plus important)")
    titrer(axe, "L'heure de départ est le facteur le plus important, devant la compagnie",
           n=314844, unite="vols partis", precision="régression logistique, tous les facteurs ensemble")
    return fig


def f25_destinations(vols):
    """Data analyst : les destinations les plus et les moins en retard (au moins 300 vols)."""
    from base_de_donnees import lire
    d = lire("destination").sort_values("pct_retard")
    choix = pd.concat([d.head(6), d.tail(6)])
    moyenne = vols_partis(vols)["DepDel15"].mean() * 100
    fig, axe = plt.subplots(figsize=(9, 5.5))
    etiquettes = [f"{v} ({c}) — n = {nombre_fr(n)}" for v, c, n in zip(choix["ville"], choix["dest"], choix["nb_vols"])]
    couleurs = [BLEU if p < moyenne else ROUGE_RETARD for p in choix["pct_retard"]]
    barres = axe.barh(etiquettes, choix["pct_retard"], color=couleurs, height=0.6)
    axe.bar_label(barres, labels=[f"{nombre_fr(p, 1)} %" for p in choix["pct_retard"]], padding=3, fontsize=8.5,
                  color=ENCRE_SECONDAIRE)
    axe.axvline(moyenne, color=ENCRE_SECONDAIRE, linestyle="--", linewidth=1)
    axe.text(moyenne + 0.4, -0.9, f"moyenne {nombre_fr(moyenne, 1)} %", fontsize=8.5, color=ENCRE_SECONDAIRE)
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    format_pourcentage(axe, "x")
    axe.set_xlabel("% de vols partis en retard")
    titrer(axe, "Vers New York et la Floride, un vol sur quatre ou plus part en retard",
           n=len(vols), unite="vols", precision=f"6 meilleures et 6 pires destinations sur {len(d)} (au moins 300 vols)")
    return fig


FIGURES_ANALYSE = {
    "f12_depart_arrivee": f12_depart_arrivee,
    "f13_distance_rattrapage": f13_distance_rattrapage,
    "f14_correlations": f14_correlations,
    "f15_vols_par_compagnie": f15_vols_par_compagnie,
    "f01_classes_retard": f01_classes_retard,
    "f02_distribution_retard": f02_distribution_retard,
    "f03_retard_par_jour_saisons": f03_retard_par_jour_saisons,
    "f04_retard_par_heure": f04_retard_par_heure,
    "f05_profil_compagnies": f05_profil_compagnies,
    "f06_compagnies_retard_ou_annule": f06_compagnies_retard_ou_annule,
    "f07_composition": f07_composition,
    "f08_compagnies_par_saison": f08_compagnies_par_saison,
    "f09_roulage_retard_simpson": f09_roulage_retard_simpson,
    "f10_domino": f10_domino,
    "f11_jour_heure": f11_jour_heure,
    "f16_retard_par_jour": f16_retard_par_jour,
    "f17_jour_semaine": f17_jour_semaine,
    "f18_queue_retard_compagnies": f18_queue_retard_compagnies,
    "f19_classes_par_type": f19_classes_par_type,
    "f20_annulations_compagnie_semaine": f20_annulations_compagnie_semaine,
    "f21_force_des_liens": f21_force_des_liens,
    "f22_meteo_retard": f22_meteo_retard,
    "f23_modele_rapports": f23_modele_rapports,
    "f24_importance_facteurs": f24_importance_facteurs,
    "f25_destinations": f25_destinations,
}


# --------------------------------------------------------------------------- Échantillonnage (Q5)
# Les méthodes ne sont pas des compagnies : elles restent en gris / noir (styles de ligne
# et étiquettes directes), pour ne pas brouiller le code couleur du projet.

def nom_methode(methode, estimateur):
    return f"{methode} (repondéré)" if estimateur == "repondéré" else methode


def s1_erreur_pourcentage_global(mesures):
    """Erreur sur le % de retard global, pour les 100 tirages de chaque méthode (taille 5 000)."""
    from sampling_study import NB_TIRAGES  # noqa: F401  (documentation : 100 tirages)
    mesures = mesures.assign(nom=[nom_methode(m, e) for m, e in zip(mesures["methode"], mesures["estimateur"])])
    ordre = mesures.groupby("nom")["E6_erreur_pct_retard_global"].median().sort_values().index
    fig, axe = plt.subplots(figsize=(9, 5))
    generateur = np.random.default_rng(0)
    for i, nom in enumerate(ordre):
        valeurs = mesures.loc[mesures["nom"] == nom, "E6_erreur_pct_retard_global"]
        axe.scatter(valeurs, i + generateur.uniform(-0.18, 0.18, len(valeurs)), s=10, alpha=0.5,
                    color=ROUGE_RETARD if "grappes" in nom or "naïf" in nom else GRIS_FONCE)
        axe.plot([valeurs.median()] * 2, [i - 0.3, i + 0.3], color=ENCRE, linewidth=2)
    axe.set_yticks(range(len(ordre)), ordre)
    axe.set_xlabel("Erreur absolue sur le % de vols en retard (points) — trait noir : médiane des 100 tirages")
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    titrer(axe, "Pour un chiffre global, les grappes de jours sont de loin les moins précises",
           n=len(mesures), unite="échantillons de 5 000 vols", precision="100 tirages par méthode")
    return fig


def s2_conclusions_par_methode(mesures):
    """% des tirages où chaque conclusion de la question principale reste vraie (taille 5 000)."""
    from sampling_study import CONCLUSIONS, CONCLUSIONS_COURTES
    mesures = mesures.assign(nom=[nom_methode(m, e) for m, e in zip(mesures["methode"], mesures["estimateur"])])
    tableau = mesures.groupby("nom")[list(CONCLUSIONS)].mean() * 100
    tableau = tableau.loc[tableau.mean(axis=1).sort_values(ascending=False).index]
    fig, axe = plt.subplots(figsize=(12.5, 5.5))
    image = axe.imshow(tableau.to_numpy(), cmap="Greys", vmin=0, vmax=100, aspect="auto")
    for i in range(tableau.shape[0]):
        for j in range(tableau.shape[1]):
            valeur = tableau.iloc[i, j]
            axe.text(j, i, f"{valeur:.0f}", ha="center", va="center", fontsize=9,
                     color="white" if valeur > 60 else (ROUGE_RETARD if valeur < 50 else ENCRE),
                     fontweight="bold" if valeur < 50 else "normal")
    axe.set_xticks(range(len(CONCLUSIONS)), [f"{c}\n{CONCLUSIONS_COURTES[c]}" for c in CONCLUSIONS], fontsize=7.5)
    axe.set_yticks(range(len(tableau)), tableau.index, fontsize=9)
    axe.grid(False)
    fig.colorbar(image, ax=axe, shrink=0.7, label="% des 100 tirages où la conclusion reste vraie")
    titrer(axe, "Les grandes conclusions survivent à toutes les méthodes ; les écarts fins, non",
           precision="échantillons de 5 000 vols, 100 tirages par méthode ; en rouge : moins d'un tirage sur deux",
           n=None)
    return fig


def s3_tailles(mesures_tailles):
    """Courbes de perte d'information : quatre mesures en fonction de la taille de l'échantillon."""
    mesures = mesures_tailles.assign(nom=[nom_methode(m, e) for m, e in
                                          zip(mesures_tailles["methode"], mesures_tailles["estimateur"])])
    conclusions = [c for c in mesures.columns if len(c) == 2 and c.startswith("C")]
    mesures["pct_conclusions"] = mesures[conclusions].mean(axis=1) * 100
    panneaux = [("E6_erreur_pct_retard_global", "Erreur sur le % global (points)"),
                ("E6_erreur_max_compagnie", "Plus grande erreur sur une compagnie (points)"),
                ("E6_nb_compagnies_comparables", "Compagnies comparables (≥ 30 vols), sur 14"),
                ("pct_conclusions", "Conclusions C1-C8 conservées (%)")]
    moyennes = mesures.groupby(["nom", "taille"])[[colonne for colonne, _ in panneaux]].mean()
    styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (1, 1))]
    tailles = sorted(mesures["taille"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    for axe, (colonne, titre) in zip(axes, panneaux):
        for style, nom in zip(styles, moyennes.index.get_level_values("nom").unique()):
            serie = moyennes.loc[nom, colonne]
            couleur = ROUGE_RETARD if nom.startswith("Stratifié par compagnie") else ENCRE
            axe.plot(serie.index, serie.values, linestyle=style, marker="o", markersize=3, color=couleur,
                     label=nom, linewidth=1.6)
        axe.set_xscale("log")
        axe.set_xticks(tailles, [nombre_fr(t) for t in tailles], fontsize=8)
        axe.minorticks_off()
        axe.set_title(titre, loc="left", fontsize=10)
        axe.set_xlabel("Taille de l'échantillon (échelle log)")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Courbes de perte d'information : la taille utile dépend de la question posée",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.9, "Moyenne de 100 tirages par point. En rouge : stratifié à part égale par compagnie "
             "(naïf et repondéré) — le meilleur pour comparer les compagnies, pas pour le chiffre global.",
             fontsize=8.5, color=ENCRE_SECONDAIRE)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def s4_methode_par_objectif(mesures):
    """Quelle méthode pour quel objectif ? Rang de chaque méthode (1 = la meilleure) par mesure."""
    from sampling_study import DESCRIPTION_MESURES
    objectifs = {
        "E1_ks_retard": ("min", "E1 Forme de la\ndistribution"),
        "E2_nb_compagnies_moins_30_vols": ("min", "E2 Garder les\npetites compagnies"),
        "E3_erreur_ecart_domino": ("min", "E3 Relation\n(effet domino)"),
        "E4_erreur_moyenne_compagnie_saison": ("min", "E4 Compagnie\n× saison"),
        "E5_ecart_moyen_serie_mensuelle": ("min", "E5 Série\nmensuelle"),
        "E6_erreur_max_compagnie": ("min", "E6 Comparer\nles compagnies"),
        "E6_erreur_pct_retard_global": ("min", "E6 % de retard\nglobal"),
    }
    assert set(objectifs) <= set(DESCRIPTION_MESURES)
    mesures = mesures.assign(nom=[nom_methode(m, e) for m, e in zip(mesures["methode"], mesures["estimateur"])])
    moyennes = mesures.groupby("nom")[list(objectifs)].mean()
    rangs = pd.DataFrame({c: moyennes[c].rank(ascending=(sens == "min"), method="min")
                          for c, (sens, _) in objectifs.items()})
    rangs = rangs.loc[rangs.mean(axis=1).sort_values().index]
    fig, axe = plt.subplots(figsize=(11, 5.5))
    axe.imshow(rangs.to_numpy(), cmap="Greys_r", aspect="auto", vmin=1, vmax=len(rangs))
    for i in range(rangs.shape[0]):
        for j in range(rangs.shape[1]):
            rang = int(rangs.iloc[i, j])
            axe.text(j, i, rang, ha="center", va="center", fontsize=10,
                     color=ROUGE_RETARD if rang == 1 else ("white" if rang <= len(rangs) / 2 else ENCRE),
                     fontweight="bold" if rang == 1 else "normal")
    axe.set_xticks(range(len(objectifs)), [titre for _, titre in objectifs.values()], fontsize=8)
    axe.set_yticks(range(len(rangs)), rangs.index, fontsize=9)
    axe.grid(False)
    titrer(axe, "Aucune méthode n'est la meilleure partout : le bon choix dépend de l'objectif",
           precision="rang moyen sur 100 tirages de 5 000 vols (1 = meilleure, en rouge)", n=None)
    return fig


def s5_correlation_globale_selon_methode(mesures, correlation_population):
    """E4 : la corrélation globale roulage-retard dépend des poids des compagnies dans l'échantillon."""
    mesures = mesures.assign(nom=[nom_methode(m, e) for m, e in zip(mesures["methode"], mesures["estimateur"])])
    ordre = mesures.groupby("nom")["E4_corr_globale_roulage_retard"].median().sort_values().index
    fig, axe = plt.subplots(figsize=(9, 5))
    generateur = np.random.default_rng(1)
    for i, nom in enumerate(ordre):
        valeurs = mesures.loc[mesures["nom"] == nom, "E4_corr_globale_roulage_retard"]
        axe.scatter(valeurs, i + generateur.uniform(-0.18, 0.18, len(valeurs)), s=10, alpha=0.5,
                    color=ROUGE_RETARD if nom == "Stratifié par compagnie" else GRIS_FONCE)
    axe.axvline(correlation_population, color=ENCRE, linestyle="--")
    axe.text(correlation_population, -0.8, f"  population : ρ = {nombre_fr(correlation_population, 3)}",
             fontsize=8.5, va="top")
    axe.set_yticks(range(len(ordre)), ordre)
    axe.set_xlabel("Corrélation globale roulage-retard dans l'échantillon (ρ de Spearman)")
    axe.grid(axis="y", visible=False)
    axe.grid(axis="x", visible=True)
    titrer(axe, "Changer le poids des compagnies change la relation globale (effet de composition)",
           n=len(mesures), unite="échantillons de 5 000 vols")
    return fig


def generer_figures(vols, figures=FIGURES_ANALYSE):
    appliquer_style()
    for nom, fonction in figures.items():
        figure = fonction(vols)
        enregistrer_figure(nom)
        plt.close(figure)
        print("  ", nom)


def generer_figures_echantillonnage():
    """Figures S1-S5, à partir des mesures enregistrées par sampling_study.py."""
    from results import charger_resultats
    from sampling_study import DOSSIER_ECHANTILLONNAGE
    appliquer_style()
    methodes = pd.read_parquet(DOSSIER_ECHANTILLONNAGE / "mesures_methodes.parquet")
    correlation = charger_resultats()["correlation_roulage_retard"]["globale"]
    figures = {
        "s1_erreur_pourcentage_global": lambda: s1_erreur_pourcentage_global(methodes),
        "s2_conclusions_par_methode": lambda: s2_conclusions_par_methode(methodes),
        "s4_methode_par_objectif": lambda: s4_methode_par_objectif(methodes),
        "s5_correlation_globale_selon_methode": lambda: s5_correlation_globale_selon_methode(methodes, correlation),
    }
    fichier_tailles = DOSSIER_ECHANTILLONNAGE / "mesures_tailles.parquet"
    if fichier_tailles.exists():
        figures["s3_tailles"] = lambda: s3_tailles(pd.read_parquet(fichier_tailles))
    for nom, fabrique in figures.items():
        figure = fabrique()
        enregistrer_figure(nom)
        plt.close(figure)
        print("  ", nom)


if __name__ == "__main__":
    from preprocessing import charger_vols_propres
    generer_figures(charger_vols_propres())
    generer_figures_echantillonnage()

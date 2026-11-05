"""Méthodes d'échantillonnage et comparaison avec le dataset complet.

Population = les 319 335 vols au départ d'Atlanta (août 2021 - juillet 2022).
On tire des échantillons de 5 000 vols (≈ 1,6 % de la population) avec les 6 plans
d'échantillonnage classiques : aléatoire simple, systématique, stratifié, par grappes,
temporel et répété (bootstrap).
L'étude complète (plusieurs objectifs, plusieurs tailles) est dans sampling_study.py.

Pourquoi échantillonner ? En exploration, pour sonder un gros fichier (guide, étape 7) ;
pour les graphiques, parce que 300 000 points se superposent et deviennent illisibles.
Mais un échantillon peut déformer certaines conclusions : c'est ce que l'étude mesure.
"""

import numpy as np
import pandas as pd

from analysis import vols_partis

# Taille d'échantillon justifiée par le calcul (voir taille_echantillon et marge_erreur) :
# pour estimer le % de vols en retard (p ≈ 0,186) à ± 1 point près avec 95 % de confiance,
# la formule donne n ≈ 5 700 vols ; 5 000 vols donnent une marge de ± 1,07 point.
TAILLE_ECHANTILLON = 5000
GRAINE = 42  # random_state : le même tirage à chaque exécution (reproductibilité)
Z_95 = 1.96  # niveau de confiance de 95 %


def taille_echantillon(N, p, e, z=Z_95):
    """Taille n d'un échantillon aléatoire simple pour estimer une proportion p à ± e près.

    n₀ = z² · p · (1 − p) / e²          (population infinie)
    n  = n₀ / (1 + (n₀ − 1) / N)        (correction pour une population finie de taille N)

    N : taille de la population ; p : proportion attendue (0,5 si inconnue : cas le plus prudent) ;
    e : marge d'erreur acceptée (0,01 = ± 1 point) ; z : 1,96 pour 95 % de confiance.
    """
    n0 = z ** 2 * p * (1 - p) / e ** 2
    return n0 / (1 + (n0 - 1) / N)


def marge_erreur(n, N, p, z=Z_95):
    """Marge d'erreur (en proportion) d'un échantillon aléatoire simple de taille n tiré parmi N.

    e = z · √(p(1 − p) / n) · √((N − n) / (N − 1))   (formule inverse de taille_echantillon)
    """
    return z * np.sqrt(p * (1 - p) / n) * np.sqrt((N - n) / (N - 1))

# Seuil des retards extrêmes : Q3 + 1,5 × IQR du retard au départ dans la population
# (7 + 1,5 × 11 = 23,5 min), règle IQR (Tukey)
SEUIL_RETARD_EXTREME = 23.5


# --------------------------------------------------------------------------- Méthodes

def echantillon_aleatoire_simple(vols, taille=TAILLE_ECHANTILLON, graine=GRAINE):
    """Tire `taille` vols au hasard : chaque vol a la même chance d'être choisi."""
    return vols.sample(n=taille, random_state=graine)


def echantillon_systematique(vols, taille=TAILLE_ECHANTILLON, graine=GRAINE):
    """Trie les vols par date et heure, puis garde 1 vol tous les k vols (k = pas).

    Ici k = 319 335 / 5 000 ≈ 63,9 : on arrondit chaque position à l'entier inférieur,
    ce qui répartit les 5 000 vols sur toute la période (d'août 2021 à juillet 2022).
    Le point de départ est tiré au hasard dans le premier intervalle.
    """
    vols_tries = vols.sort_values(["FlightDate", "CRSDepTime"])
    pas = len(vols_tries) / taille
    depart = np.random.default_rng(graine).uniform(0, pas)
    positions = (depart + np.arange(taille) * pas).astype(int)
    return vols_tries.iloc[positions]


def echantillon_stratifie(vols, colonne_strate="compagnie", taille=TAILLE_ECHANTILLON, graine=GRAINE):
    """Stratifié PROPORTIONNEL : le même pourcentage de vols est tiré dans chaque groupe (strate).

    Exemple avec les compagnies : Delta fait 64 % des vols, donc 64 % de l'échantillon sera Delta.
    """
    fraction = taille / len(vols)
    return vols.groupby(colonne_strate).sample(frac=fraction, random_state=graine)


def echantillon_stratifie_non_proportionnel(vols, colonne_strate="compagnie", taille=TAILLE_ECHANTILLON,
                                            graine=GRAINE):
    """Stratifié NON PROPORTIONNEL : le même NOMBRE de vols est tiré dans chaque groupe.

    Avec 14 groupes et 5 000 vols : 357 vols par compagnie, qu'elle soit grande ou petite.
    Utile pour comparer les compagnies avec assez de vols chacune, mais l'échantillon ne
    ressemble plus à la population (Delta passe de 64 % à 7 %) : pour estimer une valeur
    globale, il faut repondérer chaque vol (voir `poids_de_sondage`).
    Si un groupe a moins de vols que demandé, on prend tous ses vols.
    """
    nb_groupes = vols[colonne_strate].nunique()
    par_groupe = taille // nb_groupes
    return pd.concat(
        groupe.sample(n=min(len(groupe), par_groupe), random_state=graine)
        for _, groupe in vols.groupby(colonne_strate)
    )


def echantillon_par_grappes(vols, taille=TAILLE_ECHANTILLON, graine=GRAINE):
    """Tire des journées entières au hasard (une grappe = tous les vols d'un jour).

    On tire autant de jours que nécessaire pour atteindre environ `taille` vols.
    Pratique dans la réalité (on collecte quelques jours complets), mais risqué :
    si on tombe sur des jours d'orage, l'échantillon ne ressemble plus à la population.
    """
    jours = vols["FlightDate"].drop_duplicates()
    vols_par_jour = len(vols) / len(jours)
    nb_jours = max(1, round(taille / vols_par_jour))
    jours_tires = jours.sample(n=nb_jours, random_state=graine)
    return vols[vols["FlightDate"].isin(jours_tires)]


def echantillon_stratifie_par_jour(vols, taille=TAILLE_ECHANTILLON, graine=GRAINE):
    """Méthode temporelle : stratifié proportionnel par date, pour que chacun des 365 jours soit représenté."""
    return echantillon_stratifie(vols, colonne_strate="FlightDate", taille=taille, graine=graine)


def echantillon_bootstrap(vols, taille=TAILLE_ECHANTILLON, graine=GRAINE):
    """Bootstrap : tirage au hasard AVEC REMISE, un même vol peut donc être tiré plusieurs fois.

    C'est la base des méthodes de ré-échantillonnage (estimer la variabilité d'un indicateur
    en répétant le tirage). L'échantillon contient donc des lignes en double, c'est normal.
    """
    return vols.sample(n=taille, replace=True, random_state=graine)


# Les 6 plans d'échantillonnage étudiés.
# Le stratifié tire le MÊME nombre de vols dans chaque compagnie : c'est l'allocation adaptée
# à notre question (comparer les compagnies) ; pour un chiffre global, on le repondère.
METHODES = {
    "Aléatoire simple": echantillon_aleatoire_simple,
    "Systématique": echantillon_systematique,
    "Stratifié par compagnie": echantillon_stratifie_non_proportionnel,
    "Par grappes (jours)": echantillon_par_grappes,
    "Temporel (par jour)": echantillon_stratifie_par_jour,
    "Répété (bootstrap)": echantillon_bootstrap,
}

# Méthode dont l'échantillon ne respecte pas les proportions des compagnies, par construction :
# pour estimer une valeur GLOBALE (et pas une valeur par compagnie), il faut repondérer.
METHODES_A_REPONDERER = {"Stratifié par compagnie"}


def poids_de_sondage(echantillon, population, colonne_strate="compagnie"):
    """Poids de chaque vol de l'échantillon = vols du groupe dans la population / dans l'échantillon.

    Un vol d'un groupe sous-représenté « compte » plus, un vol d'un groupe sur-représenté compte moins.
    Pour un échantillon proportionnel, tous les poids sont (presque) égaux.
    """
    poids = population[colonne_strate].value_counts() / echantillon[colonne_strate].value_counts()
    return echantillon[colonne_strate].map(poids)


# --------------------------------------------------------------------------- Analyse d'un échantillon

def indicateurs_du_score(vols):
    """Les 6 indicateurs utilisés pour le score de représentativité (calcul rapide)."""
    partis = vols_partis(vols)
    return {
        "% en retard (>15 min)": partis["DepDel15"].mean() * 100,
        "Retard moyen (min)": partis["DepDelay"].mean(),
        "Écart-type du retard (min)": partis["DepDelay"].std(),
        "% retards extrêmes (>44,5 min)": (partis["DepDelay"] > SEUIL_RETARD_EXTREME).mean() * 100,
        "% annulés": vols["Cancelled"].mean() * 100,
        "% Delta": (vols["compagnie"] == "Delta").mean() * 100,
    }


def indicateurs(vols):
    """Analyse complète d'un échantillon (ou de la population) : taille, qualité et indicateurs de retard.

    Les indicateurs de retard sont calculés sur les vols partis (sans les annulés), comme partout dans le projet.
    """
    partis = vols_partis(vols)
    return {
        "Nombre de vols": len(vols),
        "Lignes en double": int(vols.duplicated().sum()),
        "% retard manquant": vols["DepDelay"].isna().mean() * 100,
        "Retard médian (min)": partis["DepDelay"].median(),
        **indicateurs_du_score(vols),
        "Nombre de jours couverts": vols["FlightDate"].nunique(),
    }


def comparer_echantillons(vols):
    """Tableau comparatif : une ligne par méthode + la population complète en première ligne."""
    lignes = {"Population complète": indicateurs(vols)}
    for nom, methode in METHODES.items():
        lignes[nom] = indicateurs(methode(vols))
    return pd.DataFrame(lignes).T.round(1)


# --------------------------------------------------------------------------- Représentativité

# Le score utilise les 6 indicateurs de `indicateurs_du_score`. Le retard MÉDIAN n'y est pas :
# il vaut 0 dans la population, donc l'erreur relative (division par 0) n'aurait pas de sens.


def erreurs_par_rapport_a_la_population(population, echantillon, valeurs_population=None):
    """Pour chaque indicateur du score : valeur population, valeur échantillon, erreur absolue et relative.

    Erreur absolue  = |échantillon − population|              (dans l'unité de l'indicateur)
    Erreur relative = erreur absolue / |population| × 100      (en %, comparable entre indicateurs)
    `valeurs_population` : indicateurs de la population déjà calculés (évite de les recalculer à chaque tirage).
    """
    if valeurs_population is None:
        valeurs_population = indicateurs_du_score(population)
    valeurs_echantillon = indicateurs_du_score(echantillon)
    lignes = []
    for nom in valeurs_echantillon:
        pop, ech = valeurs_population[nom], valeurs_echantillon[nom]
        erreur_absolue = abs(ech - pop)
        lignes.append({
            "indicateur": nom,
            "population": pop,
            "échantillon": ech,
            "erreur absolue": erreur_absolue,
            "erreur relative (%)": erreur_absolue / abs(pop) * 100,
        })
    return pd.DataFrame(lignes).set_index("indicateur")


def score_representativite(population, echantillon, valeurs_population=None):
    """Score de 0 à 100 : 100 − moyenne des erreurs relatives (en %). 100 = identique à la population.

    Exemple : si l'échantillon se trompe en moyenne de 3 % sur les 6 indicateurs, le score vaut 97.
    """
    erreurs = erreurs_par_rapport_a_la_population(population, echantillon, valeurs_population)
    return max(0.0, 100 - erreurs["erreur relative (%)"].mean())


def pourcentage_retard_pondere(echantillon, population, colonne_strate="compagnie"):
    """% de vols en retard d'un échantillon stratifié, REPONDÉRÉ pour retrouver la population.

    Chaque vol reçoit un poids = (vols du groupe dans la population) / (vols du groupe dans l'échantillon).
    Ainsi un vol Delta « compte » plus qu'un vol d'une petite compagnie surreprésentée.
    """
    partis = vols_partis(echantillon)
    poids_des_vols = poids_de_sondage(echantillon, population, colonne_strate)[partis.index]
    return (partis["DepDel15"] * poids_des_vols).sum() / poids_des_vols.sum() * 100


# --------------------------------------------------------------------------- Répétition des tirages

def stabilite_des_methodes(vols, nb_tirages=100):
    """Répète chaque tirage `nb_tirages` fois (graines 0 à 99) : % de retard et score de chaque tirage.

    Un seul tirage peut être « chanceux ». En répétant, on voit quelle méthode donne
    des résultats stables (proches de la vraie valeur) et laquelle varie beaucoup.
    """
    valeurs_population = indicateurs_du_score(vols)  # calculées une seule fois
    resultats = []
    for nom, methode in METHODES.items():
        for graine in range(nb_tirages):
            echantillon = methode(vols, graine=graine)
            resultats.append({
                "méthode": nom,
                "% en retard": vols_partis(echantillon)["DepDel15"].mean() * 100,
                "score": score_representativite(vols, echantillon, valeurs_population),
            })
    return pd.DataFrame(resultats)


def classement_des_methodes(stabilite):
    """Classe les méthodes selon leur score moyen sur tous les tirages (le plus représentatif en premier)."""
    classement = stabilite.groupby("méthode").agg(
        score_moyen=("score", "mean"),
        score_minimum=("score", "min"),
        ecart_type_pct_retard=("% en retard", "std"),
    ).sort_values("score_moyen", ascending=False).round(2)
    classement.insert(0, "rang", range(1, len(classement) + 1))
    return classement

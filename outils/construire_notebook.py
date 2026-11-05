"""Construit notebooks/analysis.ipynb à partir de resultats.json.

Pourquoi un script ? Les chiffres cités dans les textes du notebook viennent de reports/resultats.json :
si la période ou une décision change, on relance le pipeline puis ce script, et le récit suit
les nouveaux résultats (aucun chiffre recopié à la main).

    python pipeline.py
    python outils/construire_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
"""

import json
import sys
from pathlib import Path

import nbformat as nbf

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from visualizations import nombre_fr  # noqa: E402

R = json.loads((RACINE / "reports/resultats.json").read_text(encoding="utf-8"))
cellules = []  # (bloc, cellule)
BLOC = "debut"

# Ordre du notebook = ordre du cahier des charges (§20) : exploration, qualité, nettoyage,
# échantillonnage, univarié, bivarié, multivarié, puis robustesse des conclusions et synthèse.
# Le script écrit les cellules par thème ; chaque cellule reçoit l'étiquette de son bloc,
# et le notebook est assemblé à la fin dans cet ordre.
ORDRE_BLOCS = ["debut", "descriptif", "echA", "univarie", "biv_temps", "biv_compagnies", "biv_num", "biv_domino",
               "biv_force", "multi_profil", "multi_jourheure", "multi_compo", "multi_saison", "multi_crises",
               "echB", "fin"]


def bloc(nom):
    global BLOC
    assert nom in ORDRE_BLOCS
    BLOC = nom


def md(texte):
    cellules.append((BLOC, nbf.v4.new_markdown_cell(texte.strip())))


def code(texte):
    cellules.append((BLOC, nbf.v4.new_code_cell(texte.strip())))


def f(valeur, decimales=1):
    return nombre_fr(valeur, decimales)


def pts(valeur, decimales=2):
    """« 0,42 point » ou « 2,40 points » (accord du pluriel en français : pluriel à partir de 2)."""
    return f"{f(valeur, decimales)} point" + ("s" if abs(valeur) >= 2 else "")


def lecture(question, observation, interpretation, limites, lien):
    """Encadré d'interprétation placé sous chaque figure importante."""
    md(f"""
> **Question.** {question}
>
> **Ce que l'on observe.** {observation}
>
> **Interprétation.** {interpretation}
>
> **Limites.** {limites}
>
> **Lien avec la question principale.** {lien}
""")


POP, IND, ETE, QUAL = R["population"], R["indicateurs"], R["zoom_ete_2022"], R["qualite"]
CIE, COMPO_H, COMPO_M = R["compagnies"], R["composition_tranche_horaire"], R["composition_mois"]
ENDEAVOR = "Endeavor (Delta Connection)"


def taux_type(type_):
    """% retardés ou annulés d'un type de compagnie (moyenne pondérée par le nombre de vols)."""
    lignes = [c for c in CIE.values() if c["type"] == type_]
    return sum(c["pct_retard_ou_annule"] * c["nb_vols"] for c in lignes) / sum(c["nb_vols"] for c in lignes)


# =========================================================================== 1. Introduction
md("""
# Qu'est-ce qui distingue les compagnies aériennes au départ d'Atlanta ?

**Analyse exploratoire et entrepôt de données — Salma Bouhmid**

> **Question principale.** Qu'est-ce qui distingue les compagnies aériennes au départ d'Atlanta
> (août 2021 – juillet 2022), quels facteurs observables sont associés à ces différences,
> et ces conclusions résistent-elles à l'échantillonnage ?

| | Question secondaire | Méthode |
|---|---|---|
| Q1 | Comment les compagnies diffèrent-elles en ponctualité, gravité des retards, fiabilité et exploitation ? | Décrire, comparer |
| Q2 | Ces écarts sont-ils associés à *quand* elles volent (heure, mois) ? Y a-t-il un effet de composition ? | Comparer, structurer |
| Q3 | Le classement des compagnies est-il stable selon la saison ? | Évoluer |
| Q4 | Quand le retard apparaît-il et comment se propage-t-il d'un vol au suivant (effet domino) ? | Relier |
| Q5 | Quelles méthodes d'échantillonnage, à quelle taille, préservent ces conclusions ? | Sonder, répéter |
| Q6 | Quels biais et limites encadrent ces conclusions ? | Interpréter |

**Hypothèses de départ** (à confirmer ou rejeter par les données) :
- **H1** — les compagnies diffèrent nettement, et pas seulement par hasard ;
- **H2** — une partie des écarts vient de la composition des vols (heure, mois), pas de la compagnie elle-même ;
- **H3** — le classement des compagnies change avec la saison ;
- **H4** — le retard s'accumule au fil de la journée et se transmet d'un vol au suivant du même avion ;
- **H5** — la meilleure méthode d'échantillonnage dépend de l'objectif.

**Prudence.** Les données sont *observationnelles* : on mesure des **associations**, pas des causes.
Les causes officielles des retards (compagnie, météo, contrôle aérien...) ne sont pas dans le fichier utilisé ;
la météo d'Atlanta est ajoutée par **fusion** (section 3 bis) pour expliquer une partie du retard.
""")

# =========================================================================== 2. Dataset
md(f"""
## 1. Le jeu de données

**Source.** Jeu Kaggle *Flight Status Prediction* (Rob Mulla, licence CC0), extrait du tableau
*Marketing Carrier On-Time Performance* du Bureau of Transportation Statistics (BTS, ministère américain des Transports).

**Population étudiée (P12).** Tous les vols programmés au départ d'Atlanta (ATL), du {POP['debut']} au {POP['fin']} :
**{f(POP['nb_vols'], 0)} vols**, {POP['nb_jours']} jours, {POP['nb_compagnies']} compagnies, {POP['nb_destinations']} destinations.

**Une ligne = un vol programmé au départ d'Atlanta** (unité d'observation). Un même avion apparaît
plusieurs fois par jour ; une compagnie apparaît des milliers de fois.

**Pourquoi 12 mois, et pourquoi ceux-là ?**
- La version 1 du projet n'étudiait que juin-juillet 2022 : or l'été est la période la plus en retard
  ({f(ETE['pct_retard'])} % de vols en retard, contre {f(IND['pct_retard'])} % sur 12 mois). Deux mois ne permettent ni de voir les saisons,
  ni de comparer les compagnies équitablement.
- Le fichier Kaggle 2022 s'arrête en juillet : août 2021 – juillet 2022 forme les **12 mois consécutifs les plus récents**,
  et contient l'été 2022 de la version 1 (qui devient un « zoom »).
- Cinq ans (2018-2022) n'étaient pas nécessaires : notre question compare les compagnies sur une même période ;
  cinq ans ajouteraient surtout 2020 (crise du COVID) et une autre question (l'évolution d'une année à l'autre).

**Métadonnées utiles.** Le retard officiel (BTS) : un vol est « en retard » s'il part 15 minutes ou plus après l'horaire prévu
(`DepDel15`). Les retards sont mesurés en minutes à la porte ; un retard négatif = départ en avance.
""")

code("""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "src"))  # le code du projet est dans src/

from analysis import frequences, resume_qualite, statistiques_descriptives
from comparison import (comparaison_brut_standardise, correlation_globale_et_par_groupe,
                        effet_domino_par_compagnie, profil_compagnies, taux_par_compagnie_et_periode)
from data_loading import charger_vols_atlanta
from preprocessing import charger_vols_propres, colonnes_constantes
import figures
from visualizations import appliquer_style

appliquer_style()
pd.set_option("display.max_columns", 30)
vols_bruts = charger_vols_atlanta()   # étape 1 : extraction (data/raw/)
vols = charger_vols_propres()          # étapes 2-4 : nettoyé et enrichi (data/processed/)
print(f"Brut : {vols_bruts.shape[0]} lignes × {vols_bruts.shape[1]} colonnes")
print(f"Du {vols['FlightDate'].min():%d/%m/%Y} au {vols['FlightDate'].max():%d/%m/%Y}")
""")

code("""
# Types de variables : nature statistique (ce qu'on peut calculer) ≠ type informatique (comment c'est stocké)
types = pd.DataFrame({
    "variable": ["DepDelay", "DepDel15", "classe_retard", "compagnie", "type_compagnie", "TaxiOut",
                 "Distance", "heure_depart_prevue", "FlightDate"],
    "nature statistique": ["quantitative continue", "qualitative binaire (0/1)", "qualitative ordinale",
                           "qualitative nominale", "qualitative nominale", "quantitative continue",
                           "quantitative continue", "quantitative discrète (ordonnée)", "temporelle"],
})
types
""")


# =========================================================================== 1 bis. Contexte réel
APP = R["approfondissement"]
md("""
### 1.1 Derrière les données : le contexte réel (niveau N0 du plan de travail)

Avant d'analyser, on se renseigne sur le **métier** : comment fonctionnent l'aéroport et les compagnies ?
Chaque fait est accompagné de sa source et de ce qu'il change dans la lecture des données.
""")
md("| Thème | Ce qu'il faut savoir | Ce que cela change dans nos données | Source |\n|---|---|---|---|\n" + "\n".join(
    f"| {fm['theme']} | {fm['fait']} | {fm['lien_donnees']} | [lien]({fm['source']}) |" for fm in APP["faits_metier"]))

EQ = R.get("equipe")
if EQ:
    md("""
### 1.2 Une découverte en vérifiant les compagnies : « Comair » n'existe plus depuis 2012

 Dans le fichier, le code OH s'appelle
« Comair ». Comair a fermé en septembre 2012 ; en 2021-2022, le code OH est celui de **PSA Airlines**, régionale d'American,
et nos vols « Comair » sont bien vendus sous la marque American. Nom périmé : on garde le code et on le signale (annoter, ne pas supprimer).
""")


# =========================================================================== 3. Qualité
md("""
## 2. Qualité des données

On vérifie, **avant** toute analyse : valeurs manquantes (et leur motif), doublons, valeurs impossibles, valeurs atypiques.
""")
code("resume_qualite(vols_bruts)")
code("""
manquants = vols_bruts.isna().mean().mul(100).round(2)
manquants[manquants > 0].sort_values(ascending=False).head(10).rename("% manquant")
""")
md(f"""
**Lecture : le motif des valeurs manquantes.** Les retards manquent **en bloc** (et non au hasard) : ce sont les
{f(QUAL['nb_annules'], 0)} vols **annulés**, qui n'ont pas décollé. Ce n'est pas une erreur : c'est une information.
Deux faits de qualité à garder en tête :
- {QUAL['nb_annules_partis_de_la_porte']} vols annulés avaient quand même quitté la porte (puis ont été annulés) ;
- {QUAL['nb_sans_immatriculation']} vols n'ont pas d'immatriculation d'avion (tous annulés) : ils sont exclus de l'effet domino.

**Conséquence : le biais du survivant.** Si l'on mesure seulement le retard des vols partis, une compagnie qui annule
beaucoup paraît plus ponctuelle qu'elle ne l'est. On ajoute donc l'indicateur **« retardé ou annulé »**.

Aucun doublon ({QUAL['nb_lignes_en_double']} ligne en double). Les très gros retards sont **gardés** : ce sont de vrais vols (règle IQR = *signaler*, pas supprimer).
""")

# =========================================================================== 4. Nettoyage
md("""
## 3. Nettoyage et préparation

Principe : on ne modifie jamais le fichier brut ; on crée une copie nettoyée, et chaque décision est expliquée.

| Opération | Pourquoi |
|---|---|
| Suppression des 9 colonnes constantes (aéroport de départ : toujours ATL) | Aucune information |
| Vols annulés **gardés** | L'annulation fait partie du phénomène |
| `statut`, `classe_retard` (gravité : à l'heure, petit, moyen, grand retard, annulé) | Lire la gravité, pas seulement « en retard ou non » (classes reprises de l'auteur Kaggle) |
| `retard_ou_annule` | Corriger le biais du survivant |
| `type_compagnie` : **Régionale** (lue dans les données : vol vendu sous la marque d'une autre compagnie, « _CODESHARE »), **Bas coût** (choix documenté), **Majeure** | Variable explicative candidate |
| `tranche_horaire`, `mois`, `saison` | Comparer « à heure égale » et « à période égale » |
| `minutes_rattrapees` = retard au départ − retard à l'arrivée | Ce que l'avion rattrape en vol |
| `rang_rotation`, `vol_precedent_en_retard` | Effet domino : le départ précédent d'ATL du même avion, le même jour, était-il en retard ? |
""")
code("""
print("Colonnes constantes supprimées :", colonnes_constantes(vols_bruts))
vols[["compagnie", "type_compagnie", "classe_retard", "tranche_horaire", "saison", "retard_ou_annule",
      "minutes_rattrapees", "rang_rotation", "vol_precedent_en_retard"]].head()
""")

# =========================================================================== 3 bis. Base de données
if EQ:
    md("""
## 3 bis. La base de données : fusion et vues SQL (travail du data engineer)

Choix d'architecture : **fusionner** les données, **créer des vues** (tables virtuelles) et les **stocker** avant de les manipuler.
Les vols nettoyés sont chargés dans une base SQLite (`data/atlanta_vols.sqlite`, recréée par `python pipeline.py`), en **schéma en étoile** :

| Table | Rôle | Lignes |
|---|---|---|
| `vols` | faits : un vol | 319 335 |
| `compagnies` | dimension : type, marque, base, remarque « monde réel » | 14 |
| `destinations` | dimension : ville, État | 159 |
| `calendrier` | dimension : jour, mois, saison, jour férié | 365 |
| `evenements` | les 10 crises réelles, jour par jour, avec leur source | 30 |
| `meteo_jour`, `meteo_heure` | météo d'Atlanta (Open-Meteo) | 365 / 8 760 |

Scripts : `sql/01_tables.sql` (tables) et `sql/02_vues.sql` (11 vues, une par question). Chaque vue est ensuite **matérialisée**
(stockée en table `mv_...`) pour être relue sans tout recalculer.
""")
    code("""
from base_de_donnees import requete, lire
requete("SELECT type, name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY type, name")
""")
    md("Exemple : la vue de l'effet domino. La fonction de fenêtre `LAG` lit le départ **précédent du même avion, le même jour**.")
    code("""
sql = (Path.cwd().parent / "sql/02_vues.sql").read_text(encoding="utf-8")
debut = sql.find("CREATE VIEW v_domino AS")
print(sql[debut:sql.find(";", debut) + 1])
""")
    code("""
# Contrôle : la base SQL et Python donnent les mêmes chiffres
from analysis import vols_partis
sql_compagnies = lire("ponctualite_compagnie").set_index("compagnie")["pct_retard"]
python_compagnies = (vols_partis(vols).groupby("compagnie", observed=True)["DepDel15"].mean() * 100).round(1)  # la vue arrondit à 0,1
print("Écart maximal SQL / Python (points) :", round(float((sql_compagnies - python_compagnies).abs().max()), 6))
lire("ponctualite_compagnie").head(14)
""")

# =========================================================================== 5. Descriptif
D = R["descriptifs"]["Retard au départ (min)"]
CL = R["classes_retard"]
bloc("descriptif")
md("""
## 4. Statistiques descriptives : le portrait des données

Avant d'échantillonner et d'analyser, on fait le portrait de chaque variable numérique clé :
**centre** (moyenne, médiane), **dispersion** (écart-type, IQR = Q3 − Q1), **forme**
(moyenne − médiane : positif = asymétrie à droite), **extrêmes** (centiles 90 et 99, valeurs atypiques selon la
règle IQR : hors de [Q1 − 1,5 IQR ; Q3 + 1,5 IQR]).
""")
code("statistiques_descriptives(vols)")
bloc("univarie")
md("""
## 6. Analyse univariée : décrire chaque variable

Une variable à la fois. Les variables numériques ont été décrites en section 4 ; ici, les variables
**catégorielles** (effectifs, pourcentages, diagramme en barres).
""")
code("frequences(vols, 'type_compagnie')")
code("figures.f15_vols_par_compagnie(vols); plt.show()")
lecture(
    "Comment les vols se répartissent-ils entre les compagnies ?",
    f"{POP['nb_compagnies']} compagnies, très inégales : Delta opère {f(POP['part_delta_pct'])} % des vols, les plus petites moins de 1 %. "
    f"Par type : majeures {f(POP['part_par_type_pct']['Majeure'])} %, régionales {f(POP['part_par_type_pct']['Régionale'])} %, "
    f"bas coût {f(POP['part_par_type_pct']['Bas coût'])} %.",
    "Population très déséquilibrée : un chiffre « global » décrit surtout Delta. Pour comparer les compagnies, il faut "
    "les regarder une par une, et un échantillon devra protéger les petites (voir l'échantillonnage).",
    "La classification des types est en partie notre choix (« bas coût »), documentée au nettoyage.",
    "Q1 et Q5 : justifie la comparaison compagnie par compagnie et la stratification.",
)
bloc("descriptif")
code("figures.f01_classes_retard(vols); plt.show()")
lecture(
    "Quelle est la gravité des retards au départ ?",
    f"{f(CL[list(CL)[0]])} % des vols partent à l'heure ou en avance, {f(CL['Petit retard (1-15 min)'])} % ont un petit retard, "
    f"{f(CL['Grand retard (> 45 min)'])} % un grand retard (plus de 45 min) et {f(CL['Annulé'])} % sont annulés.",
    "Le retard « officiel » (15 min ou plus) ne concerne qu'une minorité de vols, mais les grands retards ne sont pas rares.",
    "Les bornes des classes (15, 45 min) sont des conventions ; d'autres bornes changeraient les pourcentages.",
    "Q1 : la gravité (et pas seulement la fréquence) servira à comparer les compagnies.",
)
code("figures.f02_distribution_retard(vols); plt.show()")
lecture(
    "Quelle est la forme de la distribution du retard au départ ?",
    f"La médiane vaut {f(D['médiane'])} min, la moyenne {f(D['moyenne'])} min ; 90 % des vols ont moins de {f(D['P90'], 0)} min de retard, "
    f"mais le 99e centile atteint {f(D['P99'], 0)} min. {f(D['atypiques (%)'])} % des vols sont atypiques selon la règle IQR.",
    "Distribution très asymétrique à droite : la moyenne est tirée par une minorité de très gros retards. "
    "Le vol « typique » est mieux décrit par la médiane ; on utilisera surtout des **pourcentages** et des **médianes**.",
    "Le graphique est coupé à 180 min (les retards plus grands sont regroupés à droite) pour rester lisible.",
    "Justifie le choix des indicateurs de toute la suite (% en retard plutôt que retard moyen).",
)

# =========================================================================== 6. Temps
MOIS = R["retard_par_mois_pct"]
mois_min, mois_max = min(MOIS, key=MOIS.get), max(MOIS, key=MOIS.get)
bloc("biv_temps")
md("""
## 7. Analyse bivariée : relier deux variables

Deux variables à la fois, avec le graphique adapté à leurs types (cahier des charges, §11) : temps × mesure → ligne ;
catégorie × mesure → barres ; numérique × numérique → nuage de points puis corrélation.

### 7.1 Le retard selon le jour, la saison et l'heure (temps × retard)
""")
code("figures.f03_retard_par_jour_saisons(vols); plt.show()")
lecture(
    "Le retard varie-t-il au cours de l'année ?",
    f"Par saison : de {f(R['retard_par_saison_global_pct']['Automne'])} % (automne) à {f(R['retard_par_saison_global_pct']['Été'])} % (été). "
    "Jour par jour : des pics de quelques jours dans chaque saison.",
    "Forte variation saisonnière (l'été, étudié seul dans la version 1, est la saison la plus difficile), et le retard arrive par crises courtes (section 7.1 b).",
    "Une seule année : chaque mois n'est observé qu'une fois, on ne peut pas séparer « saison » et « événements de cette année-là ».",
    "Q3 : comparer les compagnies sur l'année entière évite de juger sur une seule saison.",
)
code("figures.f04_retard_par_heure(vols); plt.show()")
lecture(
    "Le retard dépend-il de l'heure de départ ?",
    "Le % de vols en retard augmente régulièrement du matin au soir.",
    "Hypothèse cohérente avec H4 : les retards de la journée s'accumulent.",
    "Association, pas causalité ; les heures très tôt et très tard ont moins de vols.",
    "Prépare Q2 (l'heure, variable de composition) et Q4 (effet domino).",
)
bloc("multi_jourheure")
md("### 8.2 Jour × heure × retard")
code("figures.f11_jour_heure(vols); plt.show()")
lecture(
    "Le motif « matin calme, soir à risque » vaut-il pour tous les jours ?",
    "Le % de vols en retard augmente du matin au soir sur chacune des 7 lignes : le motif se retrouve tous les jours de la semaine.",
    "Hypothèse cohérente avec H4 : les retards de la journée s'accumulent. L'heure est donc une **variable de composition** "
    "à neutraliser quand on compare les compagnies (Q2).",
    "Association, pas causalité ; les heures très tôt et très tard ont moins de vols (plus de variabilité).",
    "Prépare Q2 (effet de composition) et Q4 (effet domino).",
)

# =========================================================================== 7. Compagnies
pire = max(CIE, key=lambda c: CIE[c]["pct_retard_ou_annule"])
bloc("biv_compagnies")
md("### 7.2 Compagnie × retard : comparer les compagnies (Q1)")
code("profil_compagnies(vols)")
code("figures.f06_compagnies_retard_ou_annule(vols); plt.show()")
lecture(
    "Les compagnies diffèrent-elles en ponctualité ?",
    f"De {f(CIE[ENDEAVOR]['pct_retard_ou_annule'])} % (Endeavor) à {f(CIE[pire]['pct_retard_ou_annule'])} % ({pire}) de vols retardés ou annulés. "
    f"En moyenne : compagnies régionales {f(taux_type('Régionale'))} %, majeures {f(taux_type('Majeure'))} %, à bas coût {f(taux_type('Bas coût'))} %.",
    "Les écarts sont bien plus grands que les marges d'erreur (traits) : H1 est confirmée pour les grandes différences. "
    "Pour les compagnies proches (American, United, Envoy, Comair, Delta…), les marges se chevauchent : on ne peut pas les départager.",
    "Delta représente " + f(POP["part_delta_pct"]) + " % des vols : tout chiffre « global » reflète surtout Delta.",
    "Réponse directe à Q1 (fréquence des perturbations).",
)
bloc("multi_profil")
md("""
## 8. Analyse multivariée : plusieurs variables à la fois

Les relations à deux variables peuvent tromper quand une troisième variable intervient (variable de confusion,
effet de composition). On combine donc compagnie, indicateurs, heure, mois et saison.

### 8.1 Le profil complet des compagnies (14 compagnies × 7 indicateurs)
""")
code("figures.f05_profil_compagnies(vols); plt.show()")
lecture(
    "Les compagnies diffèrent-elles aussi en gravité, fiabilité et exploitation ?",
    f"Les compagnies à bas coût cumulent fréquence et gravité (ex. {pire} : {f(CIE[pire]['pct_grand_retard'])} % de grands retards, "
    f"90e centile à {f(CIE[pire]['retard_p90_min'], 0)} min). Endeavor est la plus régulière ({f(CIE[ENDEAVOR]['pct_annules'])} % d'annulations).",
    "Le profil multivarié montre que « ponctualité » n'est pas une seule dimension : fréquence, gravité et annulations ne vont pas toujours ensemble.",
    "La couleur compare chaque compagnie à la moyenne des compagnies (non pondérée) ; « vols après 18 h » est une information de contexte, non colorée.",
    "Q1 (profils) et prépare Q2 : la part de vols du soir diffère selon les compagnies.",
)

# =========================================================================== 8. Composition
AA, DL, SK = COMPO_H["American"], COMPO_H["Delta"], COMPO_M["SkyWest"]
G, PAR = R["correlation_roulage_retard"]["globale"], R["correlation_roulage_retard"]["par_compagnie"]
nb_positives = sum(1 for v in PAR.values() if v["correlation"] > 0)
bloc("multi_compo")
md("""
### 8.3 Compagnie × heure × mois : facteurs associés et effets de composition (Q2)

**Standardisation directe.** On calcule le % de retard qu'aurait chaque compagnie si elle avait **la même répartition**
(par tranche horaire, ou par mois) que l'ensemble des vols : `taux standardisé = Σ poids_strate × taux_dans_la_strate`
(même formule que la moyenne globale du guide : x̄ = Σ w_h μ_h). Si l'écart brut / standardisé est grand, la **composition**
(quand la compagnie vole) explique une partie de sa différence apparente.
""")
code("comparaison_brut_standardise(vols, 'tranche_horaire')")
code("figures.f07_composition(vols); plt.show()")
lecture(
    "Les écarts entre compagnies viennent-ils de quand elles volent ?",
    f"À tranche horaire égale, American passe de {f(AA['pct_retard_brut'])} % à {f(AA['taux_standardise_pct'])} % "
    f"(rang {AA['rang_brut']:.0f} → {AA['rang_standardise']:.0f}) : elle vole surtout le matin. "
    f"À mois égal, SkyWest passe de {f(SK['pct_retard_brut'])} % à {f(SK['taux_standardise_pct'])} % "
    f"(rang {SK['rang_brut']:.0f} → {SK['rang_standardise']:.0f}) : elle vole surtout à l'été 2022. Delta change très peu.",
    "H2 est confirmée pour certaines compagnies : une partie de leur différence apparente est un **effet de composition**. "
    "Pour les grands écarts (bas coût contre régionales), la standardisation ne change pas le classement.",
    "On ne neutralise que les variables mesurées (heure, mois) ; d'autres facteurs (destinations, type d'avion) restent possibles.",
    "Q2 : distingue ce qui tient à la compagnie de ce qui tient à son programme de vols.",
)
code("""
globale, par_compagnie = correlation_globale_et_par_groupe(vols, "TaxiOut", "DepDelayMinutes")
print(f"Corrélation globale roulage-retard : {globale:.3f}")
par_compagnie
""")
code("figures.f09_roulage_retard_simpson(vols); plt.show()")
lecture(
    "La relation entre roulage et retard est-elle la même quand on mélange les compagnies ?",
    f"Toutes compagnies mélangées, ρ = {f(G, 3)} (aucune relation apparente). À l'intérieur des compagnies, la corrélation est "
    f"positive pour {nb_positives} compagnies sur {len(PAR)}.",
    "Effet de composition **de type Simpson** : mélanger des groupes qui diffèrent à la fois par leur roulage et leur niveau de retard "
    "masque une relation présente dans presque chacun. Ce n'est pas une inversion complète : on parle d'un **signal** à analyser, pas d'une preuve.",
    "Les corrélations sont faibles (moins de 0,12) : le roulage n'explique qu'une très petite part du retard.",
    "Q2 : montre pourquoi il faut analyser **par compagnie** avant de conclure sur une relation globale.",
)

# =========================================================================== 9. Saison
bloc("multi_saison")
md("### 8.4 Compagnie × saison : le classement change-t-il ? (Q3)")
code("taux_par_compagnie_et_periode(vols, 'saison')[0]")
code("figures.f08_compagnies_par_saison(vols); plt.show()")
SAISON = R["retard_par_saison_pct"]
SAISONS = ["Automne", "Hiver", "Printemps", "Été"]
plus_ponctuelle = {s: min(SAISON, key=lambda c: SAISON[c][s]) for s in SAISONS}
moins_ponctuelle = {s: max(SAISON, key=lambda c: SAISON[c][s]) for s in SAISONS}
lecture(
    "Le classement des compagnies est-il stable selon la saison ?",
    f"Spirit : {f(SAISON['Spirit']['Hiver'])} % en hiver, {f(SAISON['Spirit']['Été'])} % en été ; "
    f"American : {f(SAISON['American']['Hiver'])} % en hiver, {f(SAISON['American']['Été'])} % en été. "
    "La plus ponctuelle selon la saison : " + ", ".join(f"{s.lower()} {c.split(' (')[0]}" for s, c in plus_ponctuelle.items())
    + " ; la moins ponctuelle : " + ", ".join(f"{s.lower()} {c}" for s, c in moins_ponctuelle.items()) + ".",
    "H3 est confirmée pour le milieu du classement : juger les compagnies sur une seule saison peut inverser leur ordre.",
    "Une seule année ; les petites compagnies ont peu de vols par saison (cases plus fragiles).",
    "Q3 : justifie le passage de 2 mois à 12 mois.",
)

# =========================================================================== 10. Domino
DOM = R["domino_par_compagnie"]
bloc("biv_domino")
md("### 7.4 Vol précédent × vol suivant : l'effet domino (Q4, ancienne problématique)")
code("effet_domino_par_compagnie(vols)")
code("figures.f10_domino(vols); plt.show()")
lecture(
    "Le retard d'un vol se transmet-il au vol suivant du même avion ?",
    f"{f(IND['domino_pct_si_precedent_en_retard'])} % de retard quand le départ précédent du même avion (même jour) était en retard, "
    f"contre {f(IND['domino_pct_si_precedent_a_l_heure'])} % sinon. L'écart existe pour chacune des {len(DOM)} compagnies mesurables.",
    "H4 est confirmée comme **association forte et régulière**. Elle est cohérente avec la cause officielle « avion arrivé en retard » de la BTS, "
    "mais nos données ne permettent pas de la prouver.",
    "Le « vol précédent » est le départ précédent d'Atlanta (pas le vol qui ramène l'avion) ; l'effet n'est mesurable que pour les compagnies "
    f"dont les avions partent plusieurs fois par jour d'Atlanta (Delta : {f(DOM['Delta']['n_vol_precedent_en_retard'], 0)} cas).",
    "Q4 : l'ancienne problématique, confirmée sur 12 mois et détaillée par compagnie.",
)
md(f"""
**Zoom été 2022 (version 1 du projet).** Sur juin-juillet 2022 seulement : {f(ETE['nb_vols'], 0)} vols, {f(ETE['pct_retard'])} % en retard,
effet domino {f(ETE['domino_pct_si_precedent_a_l_heure'])} % → {f(ETE['domino_pct_si_precedent_en_retard'])} %.
Ces chiffres sont identiques à la version 1 (vérifié par un test automatique) : le passage à 12 mois ne casse rien,
il montre que l'été surestimait le niveau annuel ({f(IND['pct_retard'])} %).
""")

# =========================================================================== 6.3 Numérique × numérique
CORR = R["correlations"]
M = CORR["matrice_spearman"]
bloc("biv_num")
md("""
### 7.3 Variables numériques entre elles (numérique × numérique)

Règle : **d'abord le nuage de points, ensuite le coefficient** (Anscombe : un même r peut cacher des formes très
différentes). Les nuages affichent 5 000 vols tirés au hasard : c'est un **sous-échantillonnage informatique** (alléger
l'affichage), pas un échantillonnage statistique ; les coefficients sont calculés sur **tous** les vols.
**Pearson** mesure une relation linéaire ; **Spearman** compare les rangs et résiste aux valeurs extrêmes.
""")
code("figures.f12_depart_arrivee(vols); plt.show()")
lecture(
    "Le retard au départ se retrouve-t-il à l'arrivée ?",
    f"Les points suivent la droite « arrivée = départ ». Pearson r = {f(CORR['depart_arrivee_pearson'], 2)}, "
    f"mais Spearman ρ = {f(CORR['depart_arrivee_spearman'], 2)}.",
    "Les gros retards au départ se reportent presque minute pour minute à l'arrivée : le retard se joue au départ. "
    "L'écart entre les deux coefficients est instructif : Pearson est tiré vers 1 par les très gros retards, alors que "
    "pour les petits écarts (quelques minutes), l'ordre à l'arrivée dépend aussi du roulage et du vol.",
    "Relation en partie mécanique : le retard à l'arrivée contient le retard au départ. Ce n'est pas une découverte causale.",
    "Justifie d'étudier la ponctualité AU DÉPART pour comparer les compagnies.",
)
code("figures.f13_distance_rattrapage(vols); plt.show()")
lecture(
    "Les vols longs rattrapent-ils plus de retard en vol ?",
    f"La médiane des minutes rattrapées augmente avec la distance, mais la relation est faible "
    f"(Spearman ρ = {f(CORR['distance_rattrapage_spearman'], 2)}) et la dispersion très grande.",
    "Un vol long laisse plus de temps pour rattraper. Une explication possible est la marge ajoutée aux horaires des vols longs "
    "(« rembourrage » des horaires, étudié dans la littérature) : c'est une **hypothèse**, nos données ne permettent pas de la vérifier.",
    "Relation faible ; r ≈ 0 ne voudrait pas dire « aucun lien », seulement « pas de lien linéaire » (guide, étape 13).",
    "Q1 : le rattrapage en vol fait partie du profil des compagnies (section 8.1).",
)
code("figures.f14_correlations(vols); plt.show()")
lecture(
    "Quelles variables numériques évoluent ensemble ?",
    f"Retard départ ↔ arrivée : ρ = {f(M['Retard départ']['Retard arrivée'], 2)} ; rattrapé en vol ↔ retard arrivée : "
    f"{f(M['Rattrapé en vol']['Retard arrivée'], 2)} ; rattrapé ↔ roulage au départ : {f(M['Rattrapé en vol']['Roulage départ'], 2)} ; "
    f"la distance est peu liée aux autres variables (au plus {f(max(abs(v) for k, v in M['Distance'].items() if k != 'Distance'), 2)}).",
    "Un long roulage au départ a lieu APRÈS la mesure du retard au départ (à la porte) : il s'ajoute au retard à l'arrivée et "
    "réduit donc le « rattrapage ». Plusieurs liens sont ainsi en partie mécaniques, parce que les variables sont construites "
    "les unes à partir des autres.",
    "Corrélation ≠ causalité ; seulement 6 variables choisies pour la question (le cahier interdit la heatmap de toutes les colonnes).",
    "Prépare l'analyse multivariée : quelles variables entrent dans le profil des compagnies.",
)

# =========================================================================== 11. Échantillonnage
ECH = R.get("echantillonnage")
S5000 = ECH["synthese_5000"] if ECH else {}

# Fiches pédagogiques des méthodes (le principe ne dépend pas des données ; les résultats, si)
FICHES_METHODES = {
    "Aléatoire simple": (
        "Chaque vol a la même chance d'être choisi, comme un tirage au sort dans une urne.",
        "On tire 5 000 vols au hasard parmi les 319 335 (graine fixée pour pouvoir refaire le même tirage).",
        "C'est la méthode de référence : aucune hypothèse, aucun choix de groupes.",
        "Par hasard, elle peut oublier presque complètement les petites compagnies."),
    "Systématique": (
        "On range les vols dans l'ordre (date, heure), puis on prend un vol tous les k vols.",
        "k = 319 335 / 5 000 ≈ 64 : on prend un vol tous les 64, en partant d'un point tiré au hasard parmi les 64 premiers.",
        "L'échantillon est étalé régulièrement sur toute l'année.",
        "Si un cycle des données a la même longueur que le pas, on tombe toujours sur le même genre de vol (périodicité)."),
    "Stratifié par compagnie": (
        "On découpe la population en groupes (les strates, ici les 14 compagnies) et on tire le MÊME NOMBRE de vols "
        "dans chacun (allocation à part égale).",
        "5 000 / 14 ≈ 357 vols par compagnie, qu'elle soit grande ou petite.",
        "Notre question est de COMPARER les compagnies : chacune a assez de vols pour être jugée (même Mesa, 0,6 % des vols).",
        "L'échantillon ne ressemble plus à la population (Delta passe de 64 % à 7 %) : pour un chiffre global, "
        "il faut repondérer chaque vol (poids N_h / n_h)."),
    "Par grappes (jours)": (
        "On tire au hasard des journées entières (les grappes) et on garde tous leurs vols.",
        "Environ 6 jours tirés au hasard, avec tous leurs vols (environ 875 par jour).",
        "Dans la vraie vie, c'est la collecte la plus simple : observer quelques journées complètes.",
        "Les vols d'un même jour se ressemblent (même météo, mêmes incidents) : 6 jours ne représentent pas une année."),
    "Temporel (par jour)": (
        "Chaque jour est une strate : on tire le même pourcentage de vols dans chacun des 365 jours.",
        "1,6 % des vols de chaque jour.",
        "Tous les jours de l'année sont représentés, dans les bonnes proportions : idéal pour suivre le temps.",
        "Ne protège pas les petites compagnies."),
    "Répété (bootstrap)": (
        "Tirage au hasard où un vol déjà tiré est « remis dans l'urne » : il peut être tiré plusieurs fois. "
        "On répète le tirage pour voir comment un résultat varie.",
        "5 000 tirages avec remise : quelques vols apparaissent deux fois.",
        "Ce n'est pas un plan d'enquête : c'est l'outil qui sert à mesurer la variabilité d'un résultat en répétant les tirages.",
        "Contient des doublons ; ne dit rien de plus que l'aléatoire simple sur la représentativité."),
}

# Fiches des mesures : (ce qu'elle mesure, formule ou calcul, comment la lire)
FICHES_MESURES = [
    ("Erreur absolue", "L'écart entre la valeur dans l'échantillon et la vraie valeur (population).",
     "Δ = |T(S) − T(P)| (guide, étape 8). Exemple : 18,6 % dans la population, 19,1 % dans l'échantillon → 0,5 point.",
     "Plus petit = meilleur. Exprimée dans l'unité de la statistique (points de %, minutes...)."),
    ("Distance de Kolmogorov-Smirnov (KS)", "Si la FORME d'une distribution est conservée, pas seulement sa moyenne.",
     "On trace pour chaque valeur x la proportion de vols dont le retard est ≤ x (courbe cumulée), dans la population et "
     "dans l'échantillon ; KS = le plus grand écart vertical entre les deux courbes.",
     "0 = formes identiques ; 0,05 = à un endroit, 5 % des vols sont « mal placés »."),
    ("Erreurs sur les centiles (P50, P90, P99)", "Si le centre (médiane), la queue (P90) et les extrêmes (P99) sont conservés.",
     "Erreur absolue sur chaque centile, en minutes.", "Plus petit = meilleur ; P99 est naturellement instable."),
    ("Plus grand écart de part", "Si les proportions (parts des compagnies, des classes de retard) sont respectées.",
     "Pour chaque catégorie : |part dans l'échantillon − part dans la population| ; on garde le plus grand.",
     "En points de % ; 0 = proportions parfaites."),
    ("Compagnies avec moins de 30 vols", "Si des petits groupes sont « perdus ».",
     "Nombre de compagnies (sur 14) qui ont moins de 30 vols dans l'échantillon. En dessous de 30 vols, la marge "
     "d'erreur d'un pourcentage dépasse ± 18 points (1,96 × √(0,25 / 30)) : on ne peut plus la juger.",
     "0 = aucune compagnie perdue."),
    ("Écart moyen sur une courbe", "Si un profil (retard par heure, par mois) est conservé.",
     "Moyenne des écarts absolus point par point entre la courbe de l'échantillon et celle de la population.",
     "En points de % ; plus petit = meilleur."),
    ("Corrélation de rang (Spearman) du classement", "Si l'ordre des compagnies est conservé.",
     "Corrélation entre le classement des compagnies dans l'échantillon et dans la population (1 = même ordre). "
     "Toujours la lire avec le nombre de compagnies comparables : classer 3 compagnies est plus facile que 14.",
     "Plus proche de 1 = meilleur."),
    ("% de tirages où une conclusion reste vraie", "Si l'échantillon permet d'arriver à la même conclusion que la population.",
     "Sur 100 tirages, combien de fois la conclusion (ex. « Southwest plus en retard que Delta ») est encore vraie.",
     "100 % = conclusion robuste ; moins de 50 % = l'échantillon ne permet pas de conclure."),
]


def resultat(methode, mesure, repondere=False):
    cle = f"{methode} (repondéré)" if repondere else methode
    return S5000.get(cle, {}).get(mesure)


bloc("echA")
md("""
## 5. Échantillonnage (partie 1) : les méthodes et la représentativité (Q5)

L'échantillonnage est présenté en deux temps. **Ici** (avant les analyses) : les 6 plans d'échantillonnage classiques,
les mesures, et ce que chaque méthode garde de la population (E1 distributions, E2 proportions, E5 temps).
**En section 9** (après les analyses) : la robustesse des **conclusions** de l'étude (E3, E4, E6, E7), car il faut d'abord
connaître les conclusions pour vérifier si un échantillon les retrouve.

### 5.1 Le principe de l'étude, en simple

Habituellement, on étudie un échantillon parce qu'on n'a **pas** la population entière. Ici, on a la population
(les 319 335 vols) : on peut donc **vérifier** ce que chaque méthode d'échantillonnage garde ou perd. La démarche :

1. **On connaît la vérité** : chaque statistique calculée sur les 319 335 vols (ex. 18,6 % de vols en retard).
2. **On tire un échantillon** de 5 000 vols avec une méthode.
3. **On mesure l'écart** entre l'échantillon et la vérité (les mesures sont expliquées en 5.3).
4. **On recommence 100 fois** (guide, étape 8 : « plusieurs échantillons plutôt qu'un seul »). Un seul tirage peut être
   chanceux ou malchanceux ; 100 tirages montrent l'erreur **habituelle** (la moyenne) et sa **variabilité** (la précision).
5. **On refait tout avec d'autres tailles** (1 000 à 20 000 vols) pour justifier la taille (E7).

**Ce qu'on compare n'est pas une seule variable** : chaque expérience (E1 à E7) a sa propre question et ses propres colonnes.
Une méthode n'est jamais félicitée pour ce qu'elle garantit **par construction** (ex. : la stratification par compagnie
respecte forcément la part de chaque compagnie — ce n'est pas un résultat, c'est la définition de la méthode).

**Naïf ou repondéré ?** *Naïf* : on calcule sur l'échantillon comme si c'était la population. *Repondéré* : chaque vol
compte avec un poids = (vols du groupe dans la population) / (vols du groupe dans l'échantillon). Indispensable pour les
méthodes qui déforment volontairement les proportions (le stratifié par compagnie, à part égale) quand on veut un chiffre **global**.
""")

md("### 5.2 Les 6 méthodes : principe, application, intérêt, risque")
for methode, (principe, application, interet, risque) in FICHES_METHODES.items():
    observe = ""
    if S5000:
        perdues = resultat(methode, "E2_nb_compagnies_moins_30_vols")
        texte_perdues = ("aucune compagnie avec moins de 30 vols" if perdues == 0
                         else f"en moyenne {f(perdues)} compagnies sur 14 avec moins de 30 vols")
        observe = (f"\n- **Ce qu'elle donne sur nos données** (moyenne de 100 tirages) : erreur de "
                   f"{pts(resultat(methode, 'E6_erreur_pct_retard_global'))} sur le % de retard global, {texte_perdues}, "
                   f"{f(resultat(methode, 'E5_nb_jours_couverts'), 0)} jours couverts sur 365, "
                   f"{f(resultat(methode, 'pct_conclusions_conservees'), 0)} % des conclusions conservées.")
        if methode == "Stratifié par compagnie":
            observe += (f" Repondérée : erreur de {pts(resultat(methode, 'E6_erreur_pct_retard_global', True))} "
                        "sur le % global. La repondération corrige le **biais** (le décalage systématique), mais pas la "
                        "**précision** : les 64 % de vols Delta ne reposent plus que sur 357 vols tirés, d'où une erreur "
                        "plus grande qu'avec un tirage aléatoire.")
    md(f"""
**{methode}**
- **Principe** : {principe}
- **Application à nos vols** : {application}
- **Utile pour** : {interet}
- **Risque** : {risque}{observe}
""")

md("### 5.3 Les mesures : comment on juge un échantillon")
md("| Mesure | Ce qu'elle mesure | Calcul | Comment la lire |\n|---|---|---|---|\n" + "\n".join(
    f"| **{nom}** | {quoi} | {calcul} | {lire} |" for nom, quoi, calcul, lire in FICHES_MESURES))

code("""
from sampling_study import CONCLUSIONS, DESCRIPTION_MESURES, DOSSIER_ECHANTILLONNAGE, synthese
mesures = pd.read_parquet(DOSSIER_ECHANTILLONNAGE / "mesures_methodes.parquet")
mesures_tailles = pd.read_parquet(DOSSIER_ECHANTILLONNAGE / "mesures_tailles.parquet")
resume = synthese(mesures).droplevel("taille")   # moyenne des 100 tirages, par méthode et estimateur
resume.index = [f"{m} (repondéré)" if e == "repondéré" else m for m, e in resume.index]
print(len(mesures), "échantillons de 5 000 vols mesurés")
pd.DataFrame(DESCRIPTION_MESURES, index=["expérience", "description", "unité"]).T
""")


def tableau_experience(colonnes):
    code(f"resume[{colonnes!r}].round(3).sort_values({colonnes[0]!r})")


def valeurs(mesure):
    """(meilleure méthode, valeur, pire méthode, valeur) pour une mesure où plus petit = meilleur."""
    serie = {m: v[mesure] for m, v in S5000.items() if v.get(mesure) is not None}
    meilleure, pire = min(serie, key=serie.get), max(serie, key=serie.get)
    return meilleure, serie[meilleure], pire, serie[pire]


if S5000:
    # ---------------------------------------------------------------- E1
    md("""
### 5.4 E1 — La forme des distributions est-elle conservée ?

**Question** : le retard au départ, le roulage et le temps rattrapé en vol ont-ils la même distribution dans l'échantillon ?
**Colonnes** : `DepDelayMinutes`, `TaxiOut`, `minutes_rattrapees`. **Mesures** : distance KS, erreurs sur P50, P90, P99.
""")
    tableau_experience(["E1_ks_retard", "E1_erreur_p50_retard", "E1_erreur_p90_retard", "E1_erreur_p99_retard",
                        "E1_ks_roulage"])
    m, v, p, vp = valeurs("E1_ks_retard")
    md(f"""
> **Réponse.** La forme est très bien conservée par les méthodes qui tirent des vols un par un : distance KS de
> {f(v, 3)} pour « {m} », contre {f(vp, 3)} pour « {p} ». Les centiles extrêmes (P99) sont les plus difficiles à retrouver,
> car ils reposent sur 1 % des vols (environ 50 vols dans un échantillon de 5 000).
> **À retenir** : pour décrire la forme d'une variable, n'importe quel tirage vol par vol suffit ; les grappes de jours non.
""")

    # ---------------------------------------------------------------- E2
    md("""
### 5.5 E2 — Les proportions sont-elles respectées ? Perd-on des petits groupes ?

**Question** : l'échantillon garde-t-il la part de chaque compagnie et de chaque classe de retard ? Les petites compagnies
ont-elles encore assez de vols pour être étudiées ? **Colonnes** : `compagnie`, `classe_retard`, `DepDelayMinutes > 45`.
""")
    tableau_experience(["E2_nb_compagnies_moins_30_vols", "E2_ecart_max_parts_compagnies", "E2_ecart_max_classes_retard",
                        "E2_erreur_pct_grand_retard"])
    md(f"""
> **Réponse.** Avec un tirage aléatoire simple, en moyenne {f(resultat('Aléatoire simple', 'E2_nb_compagnies_moins_30_vols'))}
> compagnies sur 14 ont moins de 30 vols : on ne peut pas les juger. Le stratifié par compagnie n'en perd aucune
> (357 vols par compagnie), mais il déforme les parts des compagnies
> (plus grand écart : {pts(resultat('Stratifié par compagnie', 'E2_ecart_max_parts_compagnies'), 1)} en naïf — c'est
> voulu, et la repondération le corrige : {pts(resultat('Stratifié par compagnie', 'E2_ecart_max_parts_compagnies', True), 1)}).
> **À retenir** : « représentatif » (bonnes proportions) et « utile pour comparer les groupes » (assez de vols par groupe)
> sont deux objectifs différents, que la même méthode ne peut pas remplir sans repondération.
""")

    # ---------------------------------------------------------------- E3
    bloc("echB")
    md("""
## 9. Échantillonnage (partie 2) : les conclusions résistent-elles ? (Q5)

Maintenant que les analyses ont établi les conclusions de l'étude, on vérifie si un échantillon de 5 000 vols les
retrouve, avec chaque méthode (100 tirages), puis avec d'autres tailles.

### 9.1 E3 — Les relations entre deux variables sont-elles conservées ?

**Question** : l'effet domino, le profil « retard par heure » et la relation retard au départ / retard à l'arrivée
restent-ils les mêmes ? **Colonnes** : `vol_precedent_en_retard` × `DepDel15`, `heure_depart_prevue` × `DepDel15`,
`DepDelay` × `ArrDelay`.
""")
    tableau_experience(["E3_erreur_ecart_domino", "E3_ecart_moyen_profil_horaire", "E3_erreur_corr_depart_arrivee"])
    m, v, p, vp = valeurs("E3_erreur_ecart_domino")
    md(f"""
> **Réponse.** L'écart domino de la population ({f(IND['domino_pct_si_precedent_en_retard'] - IND['domino_pct_si_precedent_a_l_heure'])}
> points) est retrouvé à {pts(v, 1)} près en moyenne avec « {m} », mais à {pts(vp, 1)} près avec « {p} ».
> Une relation repose sur un sous-groupe (ici les vols qui ont un vol précédent, environ 40 % des vols) : elle est donc
> plus fragile qu'une moyenne globale. Explication probable du mauvais score du stratifié à part égale : la plupart de ses vols
> viennent de compagnies dont les avions ne repartent presque jamais d'Atlanta le même jour, donc très peu de cas « domino ».
> **À retenir** (guide, étape 8) : *une moyenne peut être stable alors qu'une relation
> est sensible à l'échantillon* — il faut vérifier les deux.
""")

    # ---------------------------------------------------------------- E4
    md("""
### 9.2 E4 — Les structures à plusieurs variables et la composition sont-elles conservées ?

**Question** : le tableau compagnie × saison est-il retrouvé ? L'effet de composition de la section 8.3 (corrélation
roulage-retard ≈ 0 au global, positive dans les compagnies) est-il conservé, ou créé par la méthode ?
**Colonnes** : `compagnie` × `saison` × `DepDel15` ; `TaxiOut` × `DepDelayMinutes` × `compagnie`.
""")
    tableau_experience(["E4_erreur_moyenne_compagnie_saison", "E4_pct_cases_moins_30_vols",
                        "E4_corr_globale_roulage_retard", "E4_erreur_corr_par_compagnie"])
    md(f"""
> **Réponse.** Avec 5 000 vols tirés au hasard, {f(resultat('Aléatoire simple', 'E4_pct_cases_moins_30_vols'), 0)} % des cases
> compagnie × saison ont moins de 30 vols : le tableau croisé devient en partie inutilisable. Le stratifié par compagnie
> en laisse beaucoup moins ({f(resultat('Stratifié par compagnie', 'E4_pct_cases_moins_30_vols'), 0)} %). Mais, utilisé naïvement,
> il **déplace la corrélation globale** roulage-retard (moyenne {f(resultat('Stratifié par compagnie', 'E4_corr_globale_roulage_retard'), 3)},
> contre {f(G, 3)} dans la population), parce qu'il change le poids des compagnies.
> **À retenir** : changer la composition de l'échantillon change toute statistique **globale** — c'est l'effet de composition
> (et le risque de type Simpson) provoqué par la méthode elle-même.
""")

    # ---------------------------------------------------------------- E5
    bloc("echA")
    md("""
### 5.6 E5 — Le temps est-il préservé ?

**Question** : la série mensuelle du % de retard est-elle retrouvée ? Combien de jours de l'année sont couverts ?
**Colonnes** : `annee_mois`, `FlightDate`, `DepDel15`.
""")
    tableau_experience(["E5_ecart_moyen_serie_mensuelle", "E5_nb_jours_couverts"])
    m, v, p, vp = valeurs("E5_ecart_moyen_serie_mensuelle")
    md(f"""
> **Réponse.** « {m} » retrouve la série mensuelle à {pts(v, 1)} près en moyenne ; « {p} » s'en écarte de {pts(vp, 1)},
> car {f(resultat('Par grappes (jours)', 'E5_nb_jours_couverts'), 0)} jours seulement sont tirés : la plupart des mois ne sont
> pas observés du tout. **À retenir** : dès que la question porte sur le temps, il faut que l'échantillon couvre toute la période
> (temporel par jour, systématique) ; les grappes de jours sont inadaptées.
""")

    # ---------------------------------------------------------------- E6
    bloc("echB")
    md("""
### 9.3 E6 — Les conclusions de la question principale restent-elles vraies ?

C'est l'expérience la plus importante : elle relie l'échantillonnage à la question principale. Pour chacune des 8 conclusions
de l'étude (vraies dans la population), on compte dans combien des 100 tirages elle reste vraie.
""")
    code("pd.Series(CONCLUSIONS, name='Conclusion (vraie dans la population)')")
    code("figures.s2_conclusions_par_methode(mesures); plt.show()")
    alea = S5000["Aléatoire simple"]
    grappes_s = S5000["Par grappes (jours)"]
    md(f"""
> **Réponse.** Les **grandes** conclusions résistent à toutes les méthodes : C1 (bas coût > majeures > régionales),
> C2 (Southwest > Delta), C8 (soir > matin) sont vraies dans presque tous les tirages. Les conclusions **fines** ne résistent pas :
> avec un tirage aléatoire simple, C4 (effet de l'heure pour American) n'est retrouvée que dans {f(alea['pct_tirages_C4'], 0)} %
> des tirages, C5 (biais du survivant American / Delta) dans {f(alea['pct_tirages_C5'], 0)} %, et C3 (« Endeavor la plus
> ponctuelle de toutes ») dans {f(alea['pct_tirages_C3'], 0)} %, parce que des petites compagnies ne sont pas comparables.
> Seul le stratifié à part égale, qui garde assez de vols dans chaque compagnie, retrouve C3
> ({f(S5000['Stratifié par compagnie']['pct_tirages_C3'], 0)} % des tirages).
> **À retenir** : un échantillon de 5 000 vols permet d'affirmer les grandes différences entre compagnies, pas les petits écarts
> entre compagnies proches. C'est pourquoi les conclusions de la section 10 ont un statut (solide / piste / artefact).
""")
    bloc("echA")
    md("""
### 5.7 Le chiffre global : quelle précision selon la méthode ?

Erreur sur le % de vols en retard, pour chacun des 100 tirages de chaque méthode (un point = un tirage).
""")
    code("figures.s1_erreur_pourcentage_global(mesures); plt.show()")
    md(f"""
> **Réponse.** Erreur habituelle (moyenne de 100 tirages) : {pts(alea['E6_erreur_pct_retard_global'])} avec un tirage
> aléatoire simple, {pts(grappes_s['E6_erreur_pct_retard_global'])} avec les grappes de jours, dont les tirages s'étalent
> beaucoup plus. **À retenir** : pour un chiffre global, les méthodes qui tirent les vols un par un suffisent ; tirer des
> jours entiers rend le résultat dépendant des jours tombés au hasard.
""")
    bloc("echB")
    code("figures.s5_correlation_globale_selon_methode(mesures, globale); plt.show()")

    # ---------------------------------------------------------------- Synthèse et E7
    md("### 9.4 Quelle méthode pour quel objectif ?")
    code("figures.s4_methode_par_objectif(mesures); plt.show()")
    meilleures = ECH["meilleure_methode_par_mesure"]
    md(f"""
> **Réponse.** Aucune méthode n'est la meilleure partout (hypothèse H5 confirmée) :
> - pour **décrire la forme** d'une variable : {meilleures['E1_ks_retard']} ;
> - pour un **chiffre global** (% de retard) : {meilleures['E6_erreur_pct_retard_global']} ;
> - pour **suivre le temps** (série mensuelle) : {meilleures['E5_ecart_moyen_serie_mensuelle']} ;
> - pour **comparer les compagnies** (notre question principale) : {meilleures['E6_erreur_max_compagnie']},
>   à condition de **repondérer** pour tout chiffre global ;
> - les **grappes de jours** sont les moins adaptées presque partout.
""")

    md("### 9.5 E7 — Quelle taille d'échantillon ? (pourquoi 5 000 ?)")
    code("figures.s3_tailles(mesures_tailles); plt.show()")
    TM = ECH["taille_minimale_robuste"]
    noms = list(ECH["conclusions"])
    lignes = "\n".join(
        f"| {methode} | " + " | ".join(nombre_fr(v) if v else "> 20 000" for v in tailles.values()) + " |"
        for methode, tailles in TM.items())
    md(f"""
**Taille minimale pour qu'une conclusion reste vraie dans au moins 95 % des tirages :**

| Méthode | {' | '.join(noms)} |
|---|{'---|' * len(noms)}
{lignes}

> **Réponse.** Dans la version 1, la taille 5 000 n'était pas justifiée ; elle l'est maintenant par le calcul (section 5.8 :
> ± 1,07 point sur le % global) et par cette simulation. Le tableau se lit ainsi : avec un tirage aléatoire,
> {nombre_fr(TM['Aléatoire simple']['C1'])} vols suffisent pour C1 (types de compagnies) et
> {nombre_fr(TM['Aléatoire simple']['C6'])} pour C6 (effet domino) ; C3 demande un stratifié à part égale et
> {nombre_fr(TM['Stratifié par compagnie']['C3'])} vols ; C4 et C5 ne sont robustes à aucune taille testée.
> Les courbes montrent qu'il n'y a pas de taille « bonne » dans l'absolu : l'erreur sur un chiffre global diminue vite
> (moins d'un point dès quelques milliers de vols),
> alors que comparer **toutes** les compagnies demande une stratification et plus de vols, et que certaines conclusions fines
> demandent plus de 20 000 vols. **5 000 vols est un compromis raisonnable pour les grandes conclusions**, à condition de
> choisir la méthode selon l'objectif.
""")

# =========================================================================== Taille
TAILLE = R["taille_echantillon"]
bloc("echA")
md(f"""
### 5.8 La taille de l'échantillon : pourquoi n = 5 000 ?

On veut estimer le % de vols en retard (p) à **± e près**, avec **95 % de confiance**. Pour un tirage aléatoire simple :

$$n_0 = \\frac{{z^2 \\, p\\,(1-p)}}{{e^2}} \\qquad\\qquad n = \\frac{{n_0}}{{1 + \\dfrac{{n_0 - 1}}{{N}}}}$$

| Paramètre | Signification | Valeur |
|---|---|---|
| N | taille de la population | {f(TAILLE['N'], 0)} vols |
| p | proportion à estimer (% de vols en retard) | {f(TAILLE['p'], 3)} (0,5 si inconnue : cas le plus prudent) |
| z | valeur du niveau de confiance | {f(TAILLE['z'], 2)} (95 %) |
| e | marge d'erreur acceptée | 0,01 (± 1 point) |
| n / N | fraction de sondage | {f(TAILLE['fraction_pct'], 2)} % pour n = 5 000 |

**Résultat :** pour ± 1 point, **n ≈ {f(TAILLE['n_pour_1_point'], 0)} vols** (≈ {f(TAILLE['n_pour_1_point_p_0_5'], 0)} si p est inconnu ;
≈ {f(TAILLE['n_pour_2_points'], 0)} pour ± 2 points). La formule inverse donne la marge de n = 5 000 :
$e = z\\sqrt{{p(1-p)/n}}\\sqrt{{(N-n)/(N-1)}}$ = **± {f(TAILLE['marge_n_retenu_points'], 2)} point**. 5 000 vols atteignent donc
presque la précision visée de ± 1 point, pour 1,6 % de la population.

**Le nombre de colonnes n'intervient pas** : on tire des lignes, et chaque vol tiré apporte toutes ses colonnes. Seul
compte l'indicateur estimé (son p et sa marge e).

**Comparer les compagnies demande un n par compagnie.** Même formule dans la strate la plus petite (Mesa,
N_h = {f(TAILLE['mesa']['N_h'], 0)}, p_h = {f(TAILLE['mesa']['p_h'], 3)}, e = ± 5 points) : n_h ≈ {f(TAILLE['mesa']['n_h_pour_5_points'], 0)} vols.
Un tirage aléatoire de 5 000 vols n'en donne qu'environ {f(TAILLE['mesa']['n_h_si_aleatoire_5000'], 0)} : c'est pourquoi on stratifie
pour comparer les compagnies. La section 9.5 (E7) vérifie ces tailles par simulation.
""")
code("""
from sampling import marge_erreur, taille_echantillon
N, p = len(vols), vols.loc[~vols["Cancelled"], "DepDel15"].mean()
print(f"n pour ± 1 point : {taille_echantillon(N, p, 0.01):.0f} vols")
print(f"marge de n = 5 000 : ± {100 * marge_erreur(5000, N, p):.2f} point")
""")


# =========================================================================== Approfondissement (insérés à leur place)
JOUR, SEM = APP["journalier"], APP["jour_semaine"]
LIENS, PAIRES, VERS, EXT = APP["liens_bivaries"], APP["paires_compagnies"], APP["sensibilite_types"], APP["valeurs_extremes"]
DR = APP["domino_rang"]
EVTS = APP["evenements"]
CLASSES_A_L_HEURE = list(R["classes_retard"])[0]
JOURS_ORDRE = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

# ---- 4. Valeurs extrêmes, une à une
bloc("descriptif")
md(f"""
### 4.1 Les valeurs extrêmes, une à une (classification du plan de travail)

La règle IQR **signale** des valeurs ; elle ne dit pas si elles sont fausses. On classe donc les plus extrêmes avec la grille
du plan : *erreur vérifiée* → corriger ou exclure ; *correction justifiable* → corriger ; *suspecte* → garder et signaler ;
*indéterminée* → isoler comme piste.

| Contrôle | Résultat |
|---|---|
| Retards de plus de 6 heures | {f(EXT['nb_retards_plus_6h'], 0)} vols |
| Retards de plus de 24 heures | {f(EXT['nb_retards_plus_24h'], 0)} vols (maximum : {f(EXT['retard_max_min'], 0)} min, soit {f(EXT['retard_max_min'] / 60, 0)} h) |
| Heure réelle − heure prévue ≠ retard annoncé | {f(EXT['nb_heures_incoherentes'], 0)} vols sur environ 315 000 |
| Retard à l'arrivée incohérent avec le départ (écart > 5 h) | {f(EXT['nb_arrivee_incoherente'], 0)} vol |

**Décision :** {EXT['decision']}
""")

# ---- 7. Bivarié : les trois cas, annoncés
bloc("biv_temps")
md("""
**Les trois cas de l'analyse bivariée** (deux variables à la fois, pour chercher le **lien** entre elles) :

| Les deux variables | Exemple | Graphique | Mesure de la force du lien |
|---|---|---|---|
| catégorielle × catégorielle | type de compagnie × classe de retard | barres empilées à 100 % | V de Cramér (0 à 1) |
| catégorielle × numérique | compagnie × minutes de retard | médiane, quartile, 90e centile par groupe | ε² de Kruskal-Wallis |
| numérique × numérique | retard au départ × retard à l'arrivée | nuage de points | ρ de Spearman, r de Pearson |

**Attention :** avec plus de 300 000 vols, presque tout lien est « significatif » (p < 0,001). Ce qui compte, c'est la
**force** du lien, pas la p-valeur.

### 7.1 b — Jour par jour : ce que la moyenne du mois cachait

La courbe par mois montre la saison. Mais une moyenne sur 30 jours **efface les crises courtes**. On refait donc la courbe
**jour par jour** (365 points), avec une moyenne mobile sur 7 jours pour voir la tendance.
""")
code("""
from approfondissement import resume_journalier, serie_journaliere
from contexte import verifier_evenements
serie_journaliere(vols).describe().round(1)
""")
code("figures.f16_retard_par_jour(vols); plt.show()")
lecture(
    "Que montre la vue jour par jour que la vue par mois ne montrait pas ?",
    f"Le retard va de {f(JOUR['jour_min']['pct'])} % ({JOUR['jour_min']['date']}) à {f(JOUR['jour_max']['pct'])} % "
    f"({JOUR['jour_max']['date']}) ; l'écart-type des jours ({f(JOUR['ecart_type_jours'])} points) est le double de celui "
    f"des mois ({f(JOUR['ecart_type_mois'])}). {JOUR['nb_mauvais_jours']} jours dépassent 40 % de retard, et les 10 % pires "
    f"jours concentrent {f(JOUR['pct_retards_dans_10pct_pires_jours'])} % de tous les retards de l'année. "
    f"Le {JOUR['jour_max_annulations']['date']}, {f(JOUR['jour_max_annulations']['pct'])} % des vols sont annulés.",
    f"Le retard arrive par **épisodes** : un mauvais jour est souvent suivi d'un autre (corrélation d'un jour au lendemain : "
    f"{f(JOUR['autocorrelation_1_jour'], 2)}). Les 10 épisodes numérotés correspondent à des événements réels documentés "
    "(tableau ci-dessous). C'est aussi pourquoi les **grappes de jours** échouent (section 5) : tirer 6 jours au hasard, "
    "c'est risquer de tomber sur une crise.",
    f"Les causes viennent de sources extérieures (presse, compagnies), pas de notre fichier. "
    f"Deux jours très anormaux ({', '.join(APP['jours_non_expliques'])}) n'avaient pas de source de presse ; la fusion "
    "avec la météo (section 10 du document principal) les explique : rafales à 77 km/h le 12 mars, 11 heures de pluie le 21 juillet.",
    "Q3 et Q4 : la variation dans le temps est surtout faite de crises ; Q5 : explique l'échec des grappes.",
)
md("**Les événements réels, vérifiés dans nos données** (« normal » = la moyenne de ces compagnies sur toute l'année) :\n\n"
   "| N° | Dates | Événement | Compagnies | Annulés (normal) | En retard (normal) | Ce qui s'est passé | Source |\n"
   "|---|---|---|---|---|---|---|---|\n" + "\n".join(
       f"| {i + 1} | {e['debut']} → {e['fin']} | {e['nom']} | {e['compagnies']} | {f(e['pct_annules'])} % ({f(e['pct_annules_normal'])} %) "
       f"| {f(e['pct_retard'])} % ({f(e['pct_retard_normal'])} %) | {e['explication']} | [lien]({e['source']}) |"
       for i, e in enumerate(EVTS)))
md("*Prudence :* le blizzard du 29 janvier ne concerne que 6 vols JetBlue au départ d'Atlanta ce jour-là ; "
   "le pourcentage est donc fragile, même si l'événement est bien documenté.")
code("figures.f17_jour_semaine(vols); plt.show()")
mardi = SEM["Mardi"]["pct_retard"]
vendredi = SEM["Vendredi"]["pct_retard"]
lecture(
    "Le jour de la semaine compte-t-il ?",
    f"Mardi ({f(mardi)} %) et mercredi ({f(SEM['Mercredi']['pct_retard'])} %) sont les jours les plus calmes, vendredi "
    f"({f(vendredi)} %) le plus difficile. Mardi ou mercredi est le meilleur jour **à chaque saison**. "
    f"V de Cramér = {f(SEM['association']['v_cramer'], 3)}.",
    "Les jours calmes sont aussi ceux où il y a le moins de vols (environ 850 par jour contre 916 le lundi et le jeudi) : "
    "moins de trafic, moins d'attente. Mais le samedi, le jour le plus creux (778 vols), n'est pas calme : le nombre de vols "
    "n'explique pas tout (les départs en week-end et les vacances jouent aussi, hypothèse).",
    "Lien **faible** (V = 0,06) : le jour de la semaine compte moins que l'heure ou la compagnie.",
    "Q2 : le jour de la semaine est une variable de composition possible, plus faible que l'heure.",
)

# ---- 7.2 Compagnie × retard : les trois formes du bivarié et le test du classement
bloc("biv_compagnies")
md("**Catégorielle × catégorielle : la gravité selon le type de compagnie.**")
code("from approfondissement import classes_par_type\nclasses_par_type(vols)")
code("figures.f19_classes_par_type(vols); plt.show()")
lecture(
    "Le type de compagnie est-il lié à la gravité du retard ?",
    f"À l'heure : {f(APP['classes_par_type']['Régionale'][CLASSES_A_L_HEURE])} % des vols régionaux, "
    f"{f(APP['classes_par_type']['Majeure'][CLASSES_A_L_HEURE])} % des majeurs, "
    f"{f(APP['classes_par_type']['Bas coût'][CLASSES_A_L_HEURE])} % des vols à bas coût ; les grands retards sont deux fois plus "
    f"fréquents chez les bas coût. V de Cramér (type × retardé ou annulé) = {f(LIENS['type × retardé ou annulé']['valeur'], 3)}.",
    "Les types diffèrent sur **toute** l'échelle de gravité, pas seulement sur le seuil de 15 minutes.",
    "Lien modéré-faible : le type ne dit pas tout (voir la variation à l'intérieur des types).",
    "Q1 : fréquence ET gravité.",
)
md("**Catégorielle × numérique : les minutes de retard selon la compagnie.**")
code("from approfondissement import quantiles_retard_par_compagnie\nquantiles_retard_par_compagnie(vols)")
code("figures.f18_queue_retard_compagnies(vols); plt.show()")
Q = APP["quantiles_retard_par_compagnie"]
lecture(
    "Comment les minutes de retard se répartissent-elles dans chaque compagnie ?",
    f"La **médiane est 0 ou presque partout** : la moitié des vols de chaque compagnie part à l'heure. Les compagnies diffèrent "
    f"par la queue : 90 % des vols ont moins de {f(Q[ENDEAVOR]['p90'], 0)} min de retard chez Endeavor, moins de "
    f"{f(Q['JetBlue']['p90'], 0)} min chez JetBlue. ε² de Kruskal-Wallis = {f(LIENS['compagnie × retard (min)']['valeur'], 3)}.",
    "Un boxplot classique serait écrasé sur 0 : on montre donc les centiles utiles (médiane, 3e quartile, 90e centile). "
    "La différence entre compagnies se joue sur les **mauvais jours et les mauvais vols**, pas sur le vol typique.",
    "SkyWest a peu de retards en nombre mais une longue queue (90e centile de 62 min) : fréquence ≠ gravité.",
    "Q1 : explique pourquoi on compare des pourcentages et des centiles, pas des moyennes.",
)
md(f"""
**Le classement est-il réel ? Test des {PAIRES['nb_paires']} paires de compagnies.** Pour chaque paire, un test de deux proportions
(« retardé ou annulé »). Avec {PAIRES['nb_paires']} tests, environ 5 sortiraient « significatifs » par hasard : on applique la
**correction de Bonferroni** (seuil {PAIRES['seuil_bonferroni']:.5f} au lieu de 0,05).

**Résultat :** {PAIRES['nb_non_distinctes']} paires sur {PAIRES['nb_paires']} ne peuvent pas être départagées, surtout au milieu
du classement (Delta, American, Envoy, Comair, United, Autres, SkyWest, entre 19 et 21 %). Les extrêmes, eux, sont nets :
Endeavor est différente de toutes les autres, et JetBlue aussi.
""")
code("""
from approfondissement import paires_non_distinctes
paires, seuil = paires_non_distinctes(vols)
paires[~paires["distinctes"]].sort_values("ecart_points")
""")
md("""
> **Lecture.** On garde les 14 compagnies et leur classement, mais on lit le milieu avec prudence : « Delta 4e, American 5e »
> n'a pas de sens statistique. Les phrases licites sont : *Endeavor est la plus ponctuelle ; JetBlue la moins ponctuelle ;
> les compagnies à bas coût sont derrière ; au milieu, les écarts sont trop petits pour conclure.*
""")

# ---- 7.4 Domino : la limite du rang dans la journée
bloc("biv_domino")
rangs = DR["pct_retard_par_rang"]
md(f"""
**Une objection, et sa réponse.** Si le retard se transmet, le 3e départ d'un avion devrait être plus en retard que le 1er.
Or, par rang de départ d'Atlanta : {f(rangs['1'])} % (1er), {f(rangs['2'])} % (2e), {f(rangs['3'])} % (3e) — presque pareil.

**Explication :** notre fichier ne contient que les **départs d'Atlanta**. Un « 1er départ d'Atlanta » n'est pas le 1er vol
de la journée de l'avion : {f(DR['pct_premiers_departs_apres_14h'])} % de ces départs ont lieu après 14 h, l'avion a déjà volé
ailleurs (vols absents de nos données). Le soir, ces « premiers départs » ont {f(DR['pct_retard_premier_depart_soir'])} % de
retard. Le rang est donc une **mauvaise mesure** de la position dans la journée.

**Le domino, lui, tient à chaque heure** (le matin est exclu : seulement 9 cas) :

| Tranche horaire | Vol précédent à l'heure | Vol précédent en retard |
|---|---|---|
""" + "\n".join(f"| {t} | {f(v['si_precedent_a_l_heure'])} % | {f(v['si_precedent_en_retard'])} % |"
                for t, v in DR["domino_par_tranche"].items()))

# ---- 7.5 La force des liens
bloc("biv_force")
md("""
### 7.5 Bilan du bivarié : quelle variable est la plus liée au retard ?

Chaque lien est mesuré avec l'outil adapté au type des deux variables. Les mesures ne sont pas identiques, mais elles vont
toutes de 0 (aucun lien) à 1 : on peut comparer leurs ordres de grandeur.
""")
code("""
from approfondissement import liens_bivaries
pd.DataFrame(liens_bivaries(vols)).T[["type", "mesure", "valeur"]]
""")
code("figures.f21_force_des_liens(vols); plt.show()")
lecture(
    "Parmi les variables étudiées, lesquelles sont le plus liées au retard ?",
    f"Heure de départ (ρ = {f(LIENS['heure de départ × retard (min)']['valeur'], 3)}), tranche horaire "
    f"(V = {f(LIENS['tranche horaire × en retard']['valeur'], 3)}) et type de compagnie "
    f"(V = {f(LIENS['type × retardé ou annulé']['valeur'], 3)}) arrivent en tête ; la saison suit "
    f"(V = {f(LIENS['saison × en retard']['valeur'], 3)}) ; le jour de la semaine est le plus faible "
    f"(V = {f(LIENS['jour de la semaine × en retard']['valeur'], 3)}).",
    "**Quand** on vole compte autant que **avec qui** on vole. C'est la justification chiffrée de la standardisation "
    "(section 8.3) : comparer les compagnies sans tenir compte de l'heure serait injuste.",
    "Tous les liens sont faibles (moins de 0,2) : le retard dépend de beaucoup de facteurs absents du fichier (météo, "
    "contrôle aérien). Mesures différentes : on compare des ordres de grandeur, pas des valeurs exactes.",
    "Q2 : hiérarchise les facteurs associés au retard.",
)

# ---- 8.1 Sensibilité de la classification (plusieurs versions)
bloc("multi_profil")
md("""
**Plusieurs versions de la classification (principe « multi-versions » du plan de travail).** Le type « bas coût » est un choix
(JetBlue est parfois dite « hybride »). On vérifie que la conclusion ne dépend pas de ce choix, ni d'une seule grosse compagnie :
""")
md("| Version | Bas coût | Majeures | Régionales | Ordre conservé ? |\n|---|---|---|---|---|\n" + "\n".join(
    f"| {nom} | {f(v['Bas coût'])} % | {f(v['Majeure'])} % | {f(v['Régionale'])} % | {'oui' if v['ordre_conserve'] else 'non'} |"
    for nom, v in VERS.items()))
version_c = VERS["C : sans la plus grosse compagnie de chaque type"]
md(f"""
> **Lecture.** L'ordre « bas coût > majeures > régionales » tient dans les trois versions : la conclusion est **robuste**.
> Mais la version C apporte une **nuance importante** : sans Endeavor (3 vols régionaux sur 4), les régionales passent de
> {f(VERS['A : classification retenue']['Régionale'])} % à {f(version_c['Régionale'])} %, proches des majeures
> ({f(version_c['Majeure'])} %). **L'avantage des régionales est surtout celui d'Endeavor.** Phrase licite : *les compagnies à bas
> coût sont nettement derrière ; Endeavor est nettement devant ; les autres régionales et les majeures sont proches.*
""")

# ---- 8.5 Les crises de chaque compagnie
bloc("multi_crises")
md("""
### 8.5 Compagnie × semaine × annulations : chaque compagnie a ses propres crises

Trois variables : la compagnie, le temps (la semaine) et le % de vols annulés. Ce graphique relie les chiffres au monde réel.
""")
code("figures.f20_annulations_compagnie_semaine(vols); plt.show()")
spirit = next(e for e in EVTS if e["compagnies"] == "Spirit")
lecture(
    "Les crises touchent-elles toutes les compagnies en même temps ?",
    f"Non. Spirit s'effondre début août 2021 ({f(spirit['pct_annules'])} % de vols annulés du {spirit['debut']} au "
    f"{spirit['fin']}, contre {f(spirit['pct_annules_normal'])} % le reste de l'année), Southwest en octobre, American le "
    "week-end d'Halloween et début février, JetBlue fin janvier. Seuls Noël 2021 et la tempête du 16 janvier touchent "
    "presque tout le monde.",
    "Une partie des crises vient de **l'extérieur d'Atlanta** : le blizzard de Boston et New York (JetBlue), le vent et le "
    "verglas à Dallas (American), le contrôle aérien en Floride (Southwest). Une compagnie dont les bases sont ailleurs subit "
    "à Atlanta des problèmes nés loin d'Atlanta.",
    "Les causes viennent des sources citées ; nos données montrent seulement les jours et l'ampleur.",
    "Q1 et Q3 : une partie de l'écart entre compagnies vient de quelques crises, pas d'un défaut régulier.",
)

# =========================================================================== Équipe : météo, destinations, modèle
if EQ:
    bloc("biv_force")
    MC = EQ["meteo_par_classe"]
    JN = EQ["jours_non_expliques_meteo"]
    md("""
### 7.6 Météo × retard (fusion avec Open-Meteo)

Grâce à la fusion, l'hypothèse « les orages » devient une **mesure**. Un point = un jour.
""")
    code("figures.f22_meteo_retard(vols); plt.show()")
    code("lire('jour')[['pct_retard', 'pluie_mm', 'heures_de_pluie', 'rafales_max_kmh', 'neige_cm']].corr(method='spearman')['pct_retard'].round(2)")
    lecture(
        "La météo d'Atlanta est-elle liée au retard du jour ?",
        f"Oui : {f(MC['Sec (0 mm)']['pct_retard'])} % de retard un jour sec, {f(MC['Forte (> 20 mm)']['pct_retard'])} % un jour de "
        f"forte pluie ; ρ = {f(EQ['correlations_meteo']['heures_de_pluie'], 2)} avec les heures de pluie.",
        f"Les deux jours que la presse n'expliquait pas sont expliqués par la météo : le {JN[0]['date']}, rafales à "
        f"{f(JN[0]['rafales_max_kmh'], 0)} km/h ; le {JN[1]['date']}, {f(JN[1]['heures_de_pluie'], 0)} heures de pluie "
        f"(médiane de l'année : {f(EQ['meteo_mediane']['heures_de_pluie'], 0)} h).",
        "Météo d'Atlanta seulement : un orage à l'arrivée (New York, Floride) n'est pas mesuré.",
        "Q2 et Q4 : une partie du retard est liée à des conditions extérieures, pas à la compagnie.",
    )
    D_ = EQ["destinations"]
    md("### 7.7 Destination × retard")
    code("figures.f25_destinations(vols); plt.show()")
    lecture(
        "Certaines destinations sont-elles plus en retard ?",
        f"Vers New York JFK, {f(D_['pires'][0]['pct_retard'])} % des vols partent en retard ; vers la Floride, environ un sur "
        f"quatre ; vers {D_['meilleures'][0]['ville']}, {f(D_['meilleures'][0]['pct_retard'])} %.",
        "Les destinations congestionnées (New York, Floride) sont aussi celles des compagnies à bas coût.",
        "Non standardisé par compagnie et par heure : c'est une **piste**.",
        "Ouvre une nouvelle dimension, grâce à la table `destinations`.",
    )

    bloc("multi_crises")
    M = EQ["modele"]
    RC = M["rapports_de_cotes"]
    def orc(cle):
        return f(RC[cle]["rapport_de_cotes"], 2)
    md("""
### 8.6 Le modèle : tous les facteurs ensemble (travail du data scientist)

Les analyses une à une mélangent les facteurs (JetBlue vole le soir, en été...). Une **régression logistique** les met tous
ensemble : compagnie, heure, saison, jour, vol précédent du même avion, pluie et rafales au départ (vue `v_depart_meteo`).
Chaque **rapport de cotes** se lit « à conditions égales » par rapport à la référence (Delta, matin, automne, mardi, avion à l'heure).
""")
    code("""
import modele
resultat = modele.ajuster()   # environ 1 minute : modèle complet + un modèle sans chaque bloc
print(f"n = {resultat['n']} vols partis ; pseudo-R² = {resultat['pseudo_r2']:.3f} ; AUC = {resultat['auc']:.3f}")
resultat["rapports"].round(2)
""")
    code("figures.f23_modele_rapports(vols); plt.show()")
    code("figures.f24_importance_facteurs(vols); plt.show()")
    sc = list(M["scenarios"].items())
    lecture(
        "Quand on met tous les facteurs ensemble, qu'est-ce qui pèse le plus ?",
        f"Partir le soir multiplie la cote du retard par {orc('Heure : Soir')} ; un avion déjà en retard, par "
        f"{orc('Vol précédent : vol précédent en retard')} ; JetBlue, par {orc('Compagnie : JetBlue')} (vs Delta) ; Endeavor la "
        f"divise presque par deux ({orc('Compagnie : Endeavor (Delta Connection)')}). L'heure est le bloc le plus important, devant la compagnie.",
        f"Du meilleur au pire vol possible : {f(sc[0][1])} % ({sc[0][0]}) à {f(sc[2][1])} % ({sc[2][0]}).",
        f"AUC = {f(M['auc'], 2)}, pseudo-R² = {f(M['pseudo_r2'], 3)} : le modèle n'explique qu'une partie du retard ; associations, pas causes.",
        "Q1 et Q2 : les compagnies diffèrent même à conditions égales, mais l'heure pèse plus.",
    )

# =========================================================================== 12. Synthèse
bloc("fin")
if ECH:
    pct_c1_min = min(m["pct_tirages_C1"] for m in ECH["synthese_5000"].values())
    pct_c6_min = min(m["pct_tirages_C6"] for m in ECH["synthese_5000"].values())
    robustesse_c1 = f"vrai dans au moins {f(pct_c1_min, 0)} % des échantillons de 5 000 vols, quelle que soit la méthode"
    robustesse_c6 = f"vrai dans au moins {f(pct_c6_min, 0)} % des échantillons"
else:
    robustesse_c1 = robustesse_c6 = "étude d'échantillonnage non lancée"
md(f"""
## 10. Synthèse : ce qui est solide, ce qui est une piste, ce qui est un artefact

| Constat | Statut | Pourquoi | Suite |
|---|---|---|---|
| Les compagnies à bas coût sont plus souvent et plus gravement en retard que les régionales ({f(taux_type('Bas coût'))} % contre {f(taux_type('Régionale'))} %) | **Solide** | Grand écart, marges d'erreur séparées, {robustesse_c1} | — |
| Le retard s'accumule au fil de la journée ; l'effet domino ({f(IND['domino_pct_si_precedent_a_l_heure'])} % → {f(IND['domino_pct_si_precedent_en_retard'])} %) | **Solide** | Vrai pour toutes les compagnies mesurables, {robustesse_c6} | Causes : fichiers bruts BTS |
| Une partie des écarts vient de la composition (American et l'heure, SkyWest et le mois) | **Solide** pour ces compagnies | Standardisation directe | Autres facteurs (destinations) |
| Le classement du milieu change selon la saison | **Piste** | Une seule année | Comparer avec d'autres années |
| American « plus ponctuelle » que Delta | **Artefact** de composition et d'indicateur | S'inverse à heure égale et en comptant les annulations ; écarts dans la marge d'erreur | Ne pas conclure |
| Relation roulage-retard masquée par le mélange des compagnies | **Piste** (signal de type Simpson) | Corrélations faibles, mais positives dans {nb_positives} compagnies sur {len(PAR)} | Analyse par compagnie |
| Le retard arrive par épisodes (10 événements réels : Spirit, Southwest, Omicron, tempêtes...) | **Solide** | Visibles dans nos données ET documentés par des sources | Causes officielles (BTS) |
| L'avantage des régionales est surtout celui d'Endeavor | **Solide** | Sans Endeavor : régionales {f(VERS['C : sans la plus grosse compagnie de chaque type']['Régionale'])} % contre majeures {f(VERS['C : sans la plus grosse compagnie de chaque type']['Majeure'])} % | — |
| Le milieu du classement (Delta, American, Envoy, Comair, United, SkyWest) | **Non départageable** | {PAIRES['nb_non_distinctes']} paires sur {PAIRES['nb_paires']} non distinctes (Bonferroni) | Ne pas classer au rang près |
| Mardi et mercredi plus calmes, vendredi plus difficile | **Solide mais faible** | Vrai à chaque saison ; V = {f(SEM['association']['v_cramer'], 2)} | — |
| À conditions égales, l'heure pèse plus que la compagnie | **Solide** | Modèle sur 314 844 vols ; intervalles de confiance serrés | — |
| La pluie et le vent augmentent le retard | **Solide** | Fusion météo, ρ = 0,43 sur 365 jours | Météo des destinations |
| New York et la Floride plus en retard | **Piste** | Non standardisé par compagnie et heure | Modèle avec la destination |

## 11. Limites et biais

- **Observationnel** : associations, pas causes ; les causes officielles ne sont pas dans le fichier utilisé.
- **Périmètre** : Atlanta, départs seulement ; Delta = {f(POP['part_delta_pct'])} % des vols.
- **Une seule année** ; 2021-2022 est une période de reprise après le COVID.
- **Indicateurs** : seuil de 15 min (convention) ; biais du survivant corrigé par « retardé ou annulé ».
- **Compagnie opérante ou commerciale** : Endeavor vole pour Delta ; Republic et SkyWest pour plusieurs marques.
- **Effet domino** : mesuré sur les départs d'Atlanta seulement ; le rang du départ d'Atlanta ne mesure pas la position de l'avion dans sa journée.
- **Contexte réel** : les explications des épisodes viennent de sources extérieures (presse, compagnies) ; deux jours sans source de presse sont expliqués par la météo.
- **Tests** : avec 300 000 vols, tout est « significatif » ; on juge la **force** des liens (tous faibles, moins de 0,2).

## 11 bis. Journal des décisions (ce qui a été remis en question, et pourquoi)

| Décision | Ce qui l'a déclenchée | Ce qui a changé |
|---|---|---|
| Été 2022 → 12 mois | L'été surestime le retard (26 % contre 19 %) | Comparer les compagnies sur toutes les saisons |
| 8 → 6 méthodes d'échantillonnage | Garder les plans classiques | Mêmes conclusions, étude plus lisible |
| Courbe par mois → par jour | Revue critique | Les crises courtes apparaissent, reliées au monde réel |
| « American meilleure que Delta » | Inversion avec les annulations et à heure égale | Classé artefact |
| Classement au rang près | Test des 91 paires | Le milieu du classement est déclaré non départageable |
| « Les régionales sont les plus ponctuelles » | Version sans Endeavor | Nuancé : c'est surtout Endeavor |
| Le domino contredit par le rang ? | Rang 1 ≈ rang 3 | Limite des données expliquée ; domino vérifié à chaque heure |
| Fichiers → base SQL avec vues | Consigne : fusion et tables virtuelles | 5 tables fusionnées, 11 vues stockées et testées |
| Presse seule → presse + météo | Deux jours sans source | Expliqués par la pluie et le vent |
| Analyses une à une → modèle | Les facteurs se mélangent | L'heure pèse plus que la compagnie |
| Nom « Comair » | Recherche sur le code OH | Nom périmé (PSA Airlines) signalé |

## 12. Conclusion

Les compagnies se distinguent surtout par leur **modèle** : les régionales (Endeavor en tête sur l'année) sont les plus ponctuelles,
les compagnies à bas coût les moins ponctuelles, avec des retards plus graves. Une partie des écarts restants tient à la
**composition** de leurs vols (heure, mois). Ces grandes conclusions résistent à toutes les méthodes d'échantillonnage ;
les écarts fins entre compagnies proches, non. **Le bon échantillon dépend de la question** : stratifier par compagnie
pour comparer les compagnies, tirer au hasard pour un chiffre global, éviter les grappes de jours.

Vu **jour par jour**, le retard arrive surtout par **épisodes** : des crises courtes, souvent propres à une compagnie (Spirit en août 2021, Southwest en octobre, JetBlue en janvier) ou à tout l'aéroport (Noël 2021, tempête du 16 janvier), que nous avons reliées à des événements réels documentés. Le milieu du classement n'est pas départageable ; l'avantage des régionales est surtout celui d'Endeavor.

Enfin, **à conditions égales** (modèle), l'heure de départ pèse plus que la compagnie, et la météo d'Atlanta (fusion) explique une
partie des mauvais jours : du meilleur au pire vol possible, le risque de retard va de 1,6 % à 81,5 %.
""")

notebook = nbf.v4.new_notebook()
notebook["cells"] = [c for nom in ORDRE_BLOCS for b, c in cellules if b == nom]
notebook["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(notebook, RACINE / "notebooks/analysis.ipynb")
print(f"{len(cellules)} cellules écrites dans notebooks/analysis.ipynb")

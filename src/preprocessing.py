"""Nettoyage et préparation des données.

Principe : on ne modifie jamais le fichier brut. On crée une copie nettoyée,
et chaque décision est expliquée (voir le document principal, section N1).
"""

import pandas as pd

from config import FICHIER_PROPRE
from data_loading import charger_vols_atlanta

# Compagnies qui opèrent moins de 1 000 vols sur la période : trop peu de vols
# pour comparer leur taux de retard de façon fiable. On les regroupe dans « Autres ».
SEUIL_VOLS_COMPAGNIE = 1000

NOMS_JOURS = {1: "Lundi", 2: "Mardi", 3: "Mercredi", 4: "Jeudi",
              5: "Vendredi", 6: "Samedi", 7: "Dimanche"}
NOMS_MOIS = {1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin",
             7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"}

# Noms courts, plus lisibles dans les graphiques
NOMS_COURTS_COMPAGNIES = {
    "Delta Air Lines Inc.": "Delta",
    "Endeavor Air Inc.": "Endeavor (Delta Connection)",
    "Southwest Airlines Co.": "Southwest",
    "Spirit Air Lines": "Spirit",
    "American Airlines Inc.": "American",
    "Republic Airlines": "Republic",
    "Frontier Airlines Inc.": "Frontier",
    "United Air Lines Inc.": "United",
    "JetBlue Airways": "JetBlue",
    "SkyWest Airlines Inc.": "SkyWest",
    "Mesa Airlines Inc.": "Mesa",
    "Envoy Air": "Envoy",
    "Comair Inc.": "Comair",
    "Alaska Airlines Inc.": "Alaska",
}

# Type de compagnie (variable explicative candidate pour comparer les compagnies) :
# - « Régionale » est lu dans les données : le vol est vendu sous la marque d'une autre
#   compagnie (Operated_or_Branded_Code_Share_Partners = « DL_CODESHARE », « AA_CODESHARE »...) ;
# - parmi les autres, « Bas coût » est un choix documenté (modèle économique connu de ces
#   compagnies), « Majeure » regroupe les compagnies traditionnelles à réseau.
COMPAGNIES_BAS_COUT = {"Southwest Airlines Co.", "Spirit Air Lines", "Frontier Airlines Inc.",
                       "JetBlue Airways"}

# Classes de retard au départ (reprises du notebook de l'auteur du dataset Kaggle,
# d'après la définition « petit / moyen / grand retard » de Wikipédia)
CLASSES_RETARD = ["À l'heure ou en avance", "Petit retard (1-15 min)", "Retard moyen (16-45 min)",
                  "Grand retard (> 45 min)", "Annulé"]

# Saisons météorologiques (hémisphère nord)
SAISONS = {12: "Hiver", 1: "Hiver", 2: "Hiver", 3: "Printemps", 4: "Printemps", 5: "Printemps",
           6: "Été", 7: "Été", 8: "Été", 9: "Automne", 10: "Automne", 11: "Automne"}

# Tranches horaires de départ prévu (pour comparer « à heure égale »)
BORNES_TRANCHES = [0, 10, 14, 18, 24]
NOMS_TRANCHES = ["Matin (avant 10 h)", "Midi (10-14 h)", "Après-midi (14-18 h)", "Soir (après 18 h)"]


def colonnes_constantes(vols):
    """Renvoie la liste des colonnes qui ont une seule valeur sur tout le dataset.

    Exemple : Origin vaut toujours « ATL » puisqu'on n'a gardé que les départs d'Atlanta.
    Ces colonnes n'apportent aucune information pour l'analyse.
    """
    return [col for col in vols.columns if vols[col].nunique(dropna=False) == 1]


def ajouter_statut(vols):
    """Ajoute la colonne « statut » : Annulé, Dévié, En retard (> 15 min) ou À l'heure.

    Le seuil de 15 minutes est la définition officielle du retard aux États-Unis (BTS).
    """
    statut = pd.Series("À l'heure", index=vols.index)
    statut[vols["DepDel15"] == 1] = "En retard"
    statut[vols["Diverted"]] = "Dévié"
    statut[vols["Cancelled"]] = "Annulé"
    vols["statut"] = statut
    return vols


def ajouter_type_compagnie(vols):
    """Ajoute « type_compagnie » (Régionale, Bas coût, Majeure) et « reseau_commercial ».

    reseau_commercial = la marque sous laquelle le vol est vendu (Delta, American, United...).
    Exemple : un vol Endeavor est opéré par Endeavor mais vendu par Delta.
    """
    regionale = vols["Operated_or_Branded_Code_Share_Partners"].str.endswith("_CODESHARE")
    type_compagnie = pd.Series("Majeure", index=vols.index)
    type_compagnie[vols["Airline"].isin(COMPAGNIES_BAS_COUT)] = "Bas coût"
    type_compagnie[regionale] = "Régionale"
    vols["type_compagnie"] = type_compagnie
    vols["reseau_commercial"] = vols["Marketing_Airline_Network"]
    return vols


def ajouter_classe_retard(vols):
    """Ajoute « classe_retard », une variable ordinale de gravité du retard au départ.

    DepDelayMinutes vaut 0 pour un vol parti à l'heure ou en avance (BTS).
    """
    classe = pd.cut(vols["DepDelayMinutes"], bins=[-1, 0, 15, 45, float("inf")],
                    labels=CLASSES_RETARD[:4]).astype(str)
    classe[vols["Cancelled"]] = "Annulé"
    vols["classe_retard"] = pd.Categorical(classe, categories=CLASSES_RETARD, ordered=True)
    return vols


def ajouter_effet_domino(vols):
    """Ajoute deux colonnes pour mesurer l'effet domino des retards d'un même avion.

    Un même avion (identifié par Tail_Number) part souvent plusieurs fois par jour d'Atlanta :
    il part vers une ville, revient, puis repart.
    - « rang_rotation » : 1er, 2e, 3e... départ de la journée de cet avion depuis Atlanta ;
    - « vol_precedent_en_retard » : 1 si le départ précédent du même avion, le même jour,
      était en retard (> 15 min), 0 sinon, vide si c'est son premier départ de la journée.
    Les vols sans immatriculation d'avion (443 vols, tous annulés) restent vides.
    """
    vols = vols.sort_values(["Tail_Number", "FlightDate", "CRSDepTime"])
    vols_du_meme_avion_le_meme_jour = vols.groupby(["Tail_Number", "FlightDate"])
    vols["rang_rotation"] = vols_du_meme_avion_le_meme_jour.cumcount() + 1
    # shift(1) = valeur de la ligne précédente dans le même groupe (le vol d'avant)
    vols["vol_precedent_en_retard"] = vols_du_meme_avion_le_meme_jour["DepDel15"].shift(1)
    vols.loc[vols["Tail_Number"].isna(), "rang_rotation"] = pd.NA
    return vols.sort_index()


def nettoyer_vols(vols_bruts):
    """Crée la version nettoyée du dataset à partir des données brutes.

    Étapes :
    1. suppression des colonnes constantes (aucune information) ;
    2. vérification des doublons ;
    3. création de colonnes lisibles (heure, tranche horaire, jour, mois, saison, compagnie,
       type de compagnie, statut, classe de retard) ;
    4. création d'indicateurs utiles à la problématique (vol perturbé, minutes rattrapées
       en vol, effet domino).

    Les vols annulés sont GARDÉS : l'annulation fait partie du phénomène étudié.
    Leurs valeurs manquantes (heure de départ, retard...) sont normales : un vol annulé n'a pas décollé.
    Les très gros retards sont GARDÉS : ce sont de vrais vols, pas des erreurs de saisie.
    """
    vols = vols_bruts.copy()

    # 1. Colonnes constantes
    vols = vols.drop(columns=colonnes_constantes(vols))

    # 2. Doublons (il n'y en a aucun, mais on le vérifie à chaque exécution)
    vols = vols.drop_duplicates()

    # 3. Colonnes lisibles
    vols["heure_depart_prevue"] = vols["CRSDepTime"] // 100  # 1435 -> 14 h
    vols["jour_semaine"] = vols["DayOfWeek"].map(NOMS_JOURS)
    vols["tranche_horaire"] = pd.cut(vols["heure_depart_prevue"], bins=BORNES_TRANCHES,
                                     labels=NOMS_TRANCHES, right=False).astype(str)
    vols["mois"] = vols["Month"].map(NOMS_MOIS)
    vols["annee_mois"] = vols["FlightDate"].dt.strftime("%Y-%m")  # « 2021-08 » : se trie dans l'ordre
    vols["saison"] = vols["Month"].map(SAISONS)
    vols["compagnie"] = vols["Airline"].map(NOMS_COURTS_COMPAGNIES)
    nb_vols_par_compagnie = vols["Airline"].value_counts()
    petites_compagnies = nb_vols_par_compagnie[nb_vols_par_compagnie < SEUIL_VOLS_COMPAGNIE].index
    vols.loc[vols["Airline"].isin(petites_compagnies), "compagnie"] = "Autres"
    vols = ajouter_type_compagnie(vols)
    vols = ajouter_statut(vols)
    vols = ajouter_classe_retard(vols)

    # 4. Indicateurs
    # Vol perturbé : annulé OU parti avec plus de 15 min de retard. Il corrige le « biais du
    # survivant » : un vol annulé n'a pas de retard, une compagnie qui annule beaucoup
    # paraîtrait sinon plus ponctuelle qu'elle ne l'est.
    vols["retard_ou_annule"] = ((vols["DepDel15"] == 1) | vols["Cancelled"]).astype(int)
    # Minutes rattrapées en vol : positif = l'avion a réduit son retard entre départ et arrivée
    vols["minutes_rattrapees"] = vols["DepDelay"] - vols["ArrDelay"]
    vols = ajouter_effet_domino(vols)

    return vols


def charger_vols_propres():
    """Charge le dataset nettoyé (à utiliser dans le notebook et le dashboard)."""
    return pd.read_parquet(FICHIER_PROPRE)


if __name__ == "__main__":
    vols_bruts = charger_vols_atlanta()
    vols = nettoyer_vols(vols_bruts)
    FICHIER_PROPRE.parent.mkdir(parents=True, exist_ok=True)
    vols.to_parquet(FICHIER_PROPRE, index=False)
    print(f"Brut : {vols_bruts.shape} -> Nettoyé : {vols.shape}")
    print("Colonnes constantes supprimées :", colonnes_constantes(vols_bruts))

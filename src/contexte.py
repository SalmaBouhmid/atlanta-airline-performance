"""Le contexte réel : ce qui s'est passé à Atlanta et dans les compagnies (août 2021 – juillet 2022).

Deux listes, chacune avec ses sources :
- FAITS_METIER : ce qu'il faut savoir du monde aérien pour lire les données (« derrière les chiffres ») ;
- EVENEMENTS   : les épisodes réels qui expliquent les jours anormaux trouvés dans les données.

Règle du projet : un événement n'est retenu que s'il est (1) visible dans NOS données
(vérifié par verifier_evenements) et (2) documenté par une source. Une date anormale sans
source trouvée est signalée comme « non expliquée », jamais inventée.
"""

import pandas as pd

from analysis import vols_partis

FAITS_METIER = [
    {"theme": "L'aéroport",
     "fait": "Hartsfield-Jackson Atlanta est l'aéroport le plus fréquenté du monde en 2021 (75,7 millions de passagers) "
             "et en 2022 (93,7 millions), selon le classement ACI World.",
     "lien_donnees": "Environ 875 départs par jour dans nos données : beaucoup de vols, de compagnies et de destinations.",
     "source": "https://moodiedavittreport.com/hartsfield-jackson-atlanta-tops-aci-world-ranking-of-20-busiest-airports-in-2022"},
    {"theme": "Delta",
     "fait": "Atlanta est le siège et le plus grand hub (aéroport-centre) de Delta : plus de 70 % des passagers de "
             "l'aéroport en 2023, selon Delta.",
     "lien_donnees": "Delta fait 64 % des vols de notre population : un chiffre « global » décrit surtout Delta.",
     "source": "https://news.delta.com/delta-atl-bigger-and-bolder-worlds-largest-airline-hub-summer-2025"},
    {"theme": "Les régionales",
     "fait": "Endeavor Air est une filiale à 100 % de Delta : elle opère les vols « Delta Connection » que Delta "
             "programme et vend. Atlanta est l'un de ses hubs.",
     "lien_donnees": "Dans les données, ses vols portent le code DL_CODESHARE ; elle fait 3 vols régionaux sur 4.",
     "source": "https://www.endeavorair.com/content/endeavor-air/en_us/about-us/Fact_Sheet.html"},
    {"theme": "Les régionales",
     "fait": "Les autres régionales (Republic, SkyWest, Envoy...) volent sous contrat d'achat de capacité "
             "(« capacity purchase agreement ») : la grande compagnie fixe les vols et les prix, la régionale opère.",
     "lien_donnees": "Republic et SkyWest volent sous 3 marques (Delta, American, United) : leur ponctualité mélange "
                     "plusieurs réseaux.",
     "source": "https://www.sec.gov/Archives/edgar/data/6201/000000620125000038/R27.htm"},
    {"theme": "Les causes officielles",
     "fait": "Le BTS classe chaque retard en 5 causes : compagnie, météo extrême, système aérien national, sécurité, "
             "avion arrivé en retard (« le vol précédent du même avion est arrivé en retard »). De 2010 à 2019, "
             "l'avion arrivé en retard est la 1re cause (environ 40 % des retards) ; de 2020 à 2022, c'est la compagnie.",
     "lien_donnees": "Ces causes ne sont pas dans notre fichier ; notre « effet domino » mesure l'équivalent de la "
                     "cause « avion arrivé en retard », vu depuis Atlanta.",
     "source": "https://www.bts.gov/browse-statistical-products-and-data/info-gallery/percent-flight-delay-delay-cause-2010-2022"},
    {"theme": "La période",
     "fait": "2021-2022 est l'année de la reprise après le COVID : les compagnies avaient réduit leur personnel en 2020 "
             "et manquaient d'équipages quand les voyageurs sont revenus.",
     "lien_donnees": "Plusieurs crises de l'année (Spirit en août, Noël 2021) sont liées au manque de personnel.",
     "source": "https://www.mprnews.org/story/2021/12/24/delta-cancels-christmas-eve-flights-due-to-crew-shortages"},
    {"theme": "Delta en été 2022",
     "fait": "Delta a retiré 100 départs par jour de son programme du 1er juillet au 7 août 2022, pour gagner en "
             "fiabilité (météo, contrôle aérien, absences liées au COVID).",
     "lien_donnees": "Malgré cela, juillet 2022 est le mois le plus en retard de notre année.",
     "source": "https://www.fox5ny.com/news/delta-air-lines-to-cut-100-daily-flights-over-the-summer"},
]

# Chaque événement : la période, les compagnies concernées (None = tout l'aéroport), l'explication et la source.
EVENEMENTS = [
    {"debut": "2021-08-02", "fin": "2021-08-05", "compagnies": ["Spirit"], "nom": "Crise opérationnelle de Spirit",
     "explication": "Météo, manque de personnel et équipages arrivés à leur limite d'heures de vol : Spirit annule "
                    "plus de 2 800 vols du 30 juillet au 9 août 2021 (coût : 50 millions de dollars).",
     "source": "https://www.cbsnews.com/amp/news/long-lines-stranded-passengers-and-more-than-half-of-scheduled-flights-cancelled-as-spirit-airlines-enters-fifth-day-of-operational-issues"},
    {"debut": "2021-10-09", "fin": "2021-10-10", "compagnies": ["Southwest"], "nom": "Week-end noir de Southwest",
     "explication": "Problèmes de contrôle aérien en Floride et météo : Southwest annule 29 % de son programme le "
                    "dimanche. Les rumeurs de grève des pilotes ont été démenties.",
     "source": "https://abc13.com/post/southwest-cancellations-airlines-flights-delays/11107033/"},
    {"debut": "2021-10-30", "fin": "2021-10-31", "compagnies": ["American"], "nom": "Halloween d'American",
     "explication": "Vents violents à Dallas-Fort Worth, son plus grand hub (2 pistes sur 5 utilisables), aggravés par "
                    "le manque de personnel : plus de 1 700 vols annulés en un week-end.",
     "source": "https://www.wabe.org/american-airlines-has-cancelled-nearly-1-600-flights-since-friday/"},
    {"debut": "2021-12-24", "fin": "2022-01-03", "compagnies": None, "nom": "Noël 2021 : variant Omicron",
     "explication": "Pilotes et équipages en quarantaine (Omicron), plus la météo : Delta annule 344 de ses 3 000 vols "
                    "le soir de Noël ; les annulations continuent jusqu'au début janvier.",
     "source": "https://www.mprnews.org/story/2021/12/24/delta-cancels-christmas-eve-flights-due-to-crew-shortages"},
    {"debut": "2022-01-16", "fin": "2022-01-16", "compagnies": None, "nom": "Tempête d'hiver Izzy",
     "explication": "Neige et verglas sur le Sud-Est : Delta supprime environ 300 vols de son hub d'Atlanta, "
                    "110 000 foyers sans électricité en Géorgie.",
     "source": "https://news.yahoo.com/winter-storm-izzy-worst-ice-205615918.html"},
    {"debut": "2022-01-29", "fin": "2022-01-29", "compagnies": ["JetBlue"], "nom": "Blizzard du Nord-Est",
     "explication": "Blizzard sur New York (JFK) et Boston, les bases de JetBlue : 68 % de son programme annulé ce jour-là.",
     "source": "https://foxbusiness.com/lifestyle/thousands-flights-delayed-canceled-winter-storm"},
    {"debut": "2022-02-03", "fin": "2022-02-04", "compagnies": ["American", "Southwest"], "nom": "Tempête d'hiver au Texas",
     "explication": "Pluie verglaçante à Dallas : American annule environ 700 vols (23 % de son programme) ; "
                    "Southwest suspend ses vols à Dallas Love Field.",
     "source": "https://thepointsguy.com/news/winter-weather-cancelations-feb-3"},
    {"debut": "2022-03-18", "fin": "2022-03-18", "compagnies": None, "nom": "Orages sur Atlanta",
     "explication": "Orages sur l'agglomération : la FAA retient les avions aux portes et au roulage (31 à 45 min).",
     "source": "https://www.atlantanewsfirst.com/2022/03/18/delays-reported-hartsfield-jackson-atlanta-international-airport-because-thunderstorm"},
    {"debut": "2022-04-02", "fin": "2022-04-03", "compagnies": ["JetBlue", "Spirit", "Frontier", "Southwest"],
     "nom": "Orages en Floride",
     "explication": "Orages en Floride et programmes de régulation du contrôle aérien, en pleine période de vacances : "
                    "plus de 3 300 vols annulés aux États-Unis ; JetBlue et Spirit annulent un tiers de leurs vols.",
     "source": "https://www.wlrn.org/local-news/2022-04-04/airlines-cancel-more-than-3-500-us-flights-over-weekend"},
    {"debut": "2022-06-16", "fin": "2022-06-19", "compagnies": None, "nom": "Week-end de Juneteenth",
     "explication": "Week-end de fête (Juneteenth, fête des pères), trafic record, météo, contrôle aérien et absences : "
                    "plus de 5 000 vols annulés aux États-Unis ; près de 100 à Atlanta le dimanche.",
     "source": "https://www.nbcnews.com/news/holiday-weekend-sees-massive-amount-flight-cancellations-rcna34297"},
]

# Jours très anormaux dans nos données pour lesquels aucune source de presse n'a été trouvée (dit honnêtement).
# La fusion avec la météo d'Atlanta les explique : 12 mars 2022, rafales à 77 km/h et 16 mm de pluie ;
# 21 juillet 2022, 11 heures de pluie (voir analyste.jours_non_expliques_avec_meteo).
JOURS_NON_EXPLIQUES = ["2022-03-12", "2022-07-21"]


def verifier_evenements(vols, evenements=EVENEMENTS):
    """Pour chaque événement : ce que montrent NOS données ces jours-là, comparé à la normale.

    « Normale » = la moyenne des mêmes compagnies sur toute l'année. Un événement est « visible »
    si le taux d'annulation OU de retard est au moins le double de la normale.
    """
    lignes = []
    for ev in evenements:
        perimetre = vols if ev["compagnies"] is None else vols[vols["compagnie"].isin(ev["compagnies"])]
        periode = perimetre["FlightDate"].between(pd.Timestamp(ev["debut"]), pd.Timestamp(ev["fin"]))
        pendant, normal = perimetre[periode], perimetre[~periode]
        annul, annul_normal = pendant["Cancelled"].mean() * 100, normal["Cancelled"].mean() * 100
        retard, retard_normal = (vols_partis(pendant)["DepDel15"].mean() * 100,
                                 vols_partis(normal)["DepDel15"].mean() * 100)
        lignes.append({
            "nom": ev["nom"], "debut": ev["debut"], "fin": ev["fin"],
            "compagnies": "toutes" if ev["compagnies"] is None else ", ".join(ev["compagnies"]),
            "nb_vols": len(pendant),
            "pct_annules": round(annul, 1), "pct_annules_normal": round(annul_normal, 1),
            "pct_retard": round(retard, 1), "pct_retard_normal": round(retard_normal, 1),
            "visible": bool(annul >= 2 * annul_normal or retard >= 2 * retard_normal),
            "explication": ev["explication"], "source": ev["source"],
        })
    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------- Les compagnies, dans le monde réel
# (marque sous laquelle le vol est vendu : lue dans les données, colonne Operated_or_Branded_Code_Share_Partners)
COMPAGNIES_CONTEXTE = {
    "Delta": ("Delta", "Atlanta (siège et plus grand hub)", "Atlanta est sa maison : 64 % des vols de l'aéroport."),
    "Endeavor (Delta Connection)": ("Delta Connection", "hubs Delta, dont Atlanta",
                                    "Filiale à 100 % de Delta ; opère les vols régionaux que Delta programme et vend."),
    "American": ("American", "Dallas-Fort Worth", "Atlanta n'est pas son hub : vols surtout vers ses propres hubs."),
    "United": ("United", "Chicago O'Hare", "Atlanta n'est pas son hub : vols surtout vers ses propres hubs."),
    "Autres": ("Alaska", "Seattle", "Regroupe Alaska Airlines seule (moins de 1 000 vols à Atlanta)."),
    "Southwest": ("Southwest", "Dallas (siège)", "Réseau de point à point (pas de grand hub) ; crise d'octobre 2021."),
    "Spirit": ("Spirit", "Fort Lauderdale", "Bas coût extrême ; crise d'août 2021 (manque de personnel)."),
    "Frontier": ("Frontier", "Denver", "Bas coût extrême."),
    "JetBlue": ("JetBlue", "New York JFK et Boston", "Ses bases sont dans le Nord-Est : le blizzard de janvier 2022 l'a frappée."),
    "Republic": ("American Eagle, United Express, Delta Connection", "—", "Régionale sous contrat, pour 3 grandes compagnies."),
    "SkyWest": ("United Express, Delta Connection, American Eagle", "—", "Régionale sous contrat, pour 3 grandes compagnies."),
    "Mesa": ("United Express", "—", "Régionale sous contrat pour United."),
    "Envoy": ("American Eagle", "—", "Régionale d'American."),
    "Comair": ("American Eagle", "—",
               "Nom périmé dans le fichier : Comair a fermé en 2012 ; le code OH est en 2021-2022 celui de PSA Airlines, "
               "régionale d'American (nos vols « Comair » sont bien vendus par American)."),
}
SOURCES_COMPAGNIES = {
    "Comair fermée en 2012": "https://www.newstribune.com/news/2012/jul/31/delta-shut-down-regional-carrier-comair/",
    "Code OH = PSA Airlines": "https://www.airportia.com/flights/oh5514/cincinnati/philadelphia/",
    "BTS : un code peut changer de compagnie": "https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD",
}


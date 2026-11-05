"""Dashboard « en chapitres » : qu'est-ce qui distingue les compagnies aériennes au départ d'Atlanta ?

Lancer depuis la racine du projet :  streamlit run dashboard/app.py   (ou double-clic sur lancer_dashboard.py)

Le dashboard raconte l'étude comme une histoire, en 10 pages :
Accueil → 1 Données et entrepôt → 2 Qualité → 3 Descriptif et univarié → 4 Échantillonnage →
5 Bivarié → 6 Multivarié et modèle → 7 Fiche compagnie → 8 Simulateur → 9 Conclusions.
Chaque chapitre = quelques grands chiffres, UN graphique principal, une phrase « à retenir »,
et des contrôles pour explorer. Les chiffres de référence viennent de reports/resultats.json
(les mêmes que dans le notebook).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Permet d'importer les modules du dossier src/
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from approfondissement import serie_journaliere  # noqa: E402
from base_de_donnees import FICHIER_BASE, creer_base, requete  # noqa: E402
from contexte import EVENEMENTS  # noqa: E402
from analysis import VARIABLES_DESCRIPTIVES, marge_erreur_95, statistiques_descriptives, tableau_croise_retard, vols_partis  # noqa: E402
from comparison import correlation_globale_et_par_groupe, profil_compagnies  # noqa: E402
from preprocessing import CLASSES_RETARD, NOMS_TRANCHES, charger_vols_propres  # noqa: E402
from results import charger_resultats  # noqa: E402
from sampling import GRAINE, METHODES, METHODES_A_REPONDERER, TAILLE_ECHANTILLON, marge_erreur, poids_de_sondage, taille_echantillon  # noqa: E402
from sampling_study import CONCLUSIONS, DESCRIPTION_MESURES, DOSSIER_ECHANTILLONNAGE, mesurer_echantillon, references_population  # noqa: E402
from visualizations import COULEURS_TYPES, ENCRE, ENCRE_SECONDAIRE, FOND, GRILLE, GRIS_CONTEXTE, GRIS_FONCE, ROUGE_RETARD, couleur_compagnie  # noqa: E402

st.set_page_config(page_title="Compagnies aériennes — Atlanta", page_icon="✈️", layout="wide")

# --------------------------------------------------------------------------- Charte
ACCENT = "#2a78d6"            # la chose à regarder (mise en avant)
ACCENT_CLAIR = "#a9c6ee"
ORANGE = COULEURS_TYPES["Bas coût"]
ECHELLE_RETARD = [[0, "#fbf3f3"], [1, ROUGE_RETARD]]   # séquentielle : une seule teinte (rouge = retard)
ECHELLE_ACCENT = [[0, "#f1f5fb"], [1, ACCENT]]
# 3 méthodes de l'étude des tailles : palette validée (validate_palette.js, mode clair)
COULEURS_METHODES_TAILLE = {"Aléatoire simple": "#2a78d6", "Stratifié par compagnie (repondéré)": "#eb6834",
                            "Temporel (par jour)": "#1baf7a"}
ORDRE_JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
ORDRE_SAISONS = ["Automne", "Hiver", "Printemps", "Été"]
ENDEAVOR = "Endeavor (Delta Connection)"
N_MIN = 30  # en dessous, la marge d'erreur d'un pourcentage dépasse ± 18 points : valeur masquée

CHAPITRES = [
    "Accueil",
    "1 · Données et entrepôt",
    "2 · Qualité et nettoyage",
    "3 · Descriptif et univarié",
    "4 · Échantillonnage",
    "5 · Bivarié",
    "6 · Multivarié et modèle",
    "7 · Fiche compagnie",
    "8 · Simulateur de retard",
    "9 · Conclusions",
]

st.markdown(f"""
<style>
.block-container {{padding-top: 3.2rem; max-width: 1150px;}}
.etiquette-chapitre {{font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
    color: {ACCENT}; margin-bottom: .2rem;}}
.titre-chapitre {{font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin: 0 0 .4rem 0; color: {ENCRE};}}
.accroche {{font-size: 1.12rem; color: {ENCRE_SECONDAIRE}; margin-bottom: 1.2rem; max-width: 60rem;}}
div[data-testid="stMetric"] {{background: white; border: 1px solid {GRILLE}; border-radius: 14px;
    padding: 14px 18px; box-shadow: 0 1px 2px rgba(0,0,0,.04);}}
div[data-testid="stMetricValue"] {{font-size: 2.0rem; font-weight: 800; color: {ENCRE};}}
div[data-testid="stMetricLabel"] p {{font-size: .9rem; color: {ENCRE_SECONDAIRE};}}
.a-retenir {{background: #f1f5fb; border-left: 5px solid {ACCENT}; border-radius: 10px; padding: 14px 18px;
    margin: .6rem 0 1.2rem 0; font-size: 1.05rem; color: {ENCRE};}}
.a-retenir b.titre {{display: block; font-size: .78rem; letter-spacing: .1em; text-transform: uppercase;
    color: {ACCENT}; margin-bottom: .25rem;}}
.notion {{background: #fbf7ef; border: 1px solid #efe3c8; border-radius: 10px; padding: 10px 14px;
    font-size: .93rem; color: {ENCRE_SECONDAIRE}; margin: .4rem 0 1rem 0;}}
.notion b {{color: #8a5a00;}}
.carte {{background: white; border: 1px solid {GRILLE}; border-radius: 14px; padding: 14px 16px; height: 100%;}}
.carte h4 {{margin: 0 0 .3rem 0; font-size: 1rem; color: {ENCRE}; padding: 0;}}
.carte p {{margin: 0; font-size: .92rem; color: {ENCRE_SECONDAIRE};}}
.badge {{display: inline-block; font-size: .72rem; font-weight: 700; padding: 2px 9px; border-radius: 99px;
    margin-left: 6px; vertical-align: middle;}}
.badge-solide {{background: #e3f1e8; color: #1d6b3a;}}
.badge-piste {{background: #fdf0d9; color: #8a5a00;}}
.source {{font-size: .8rem; color: #8a8984; margin-top: 1.5rem;}}
.hero {{background: linear-gradient(120deg, #0b3b5c 0%, {ACCENT} 100%); color: white; border-radius: 18px;
    padding: 26px 30px; margin-bottom: 1.2rem; box-shadow: 0 6px 18px rgba(11,59,92,.18);}}
.hero-etiquette {{font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; opacity: .85;}}
.hero-titre {{font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin: .3rem 0 .4rem 0;}}
.hero-sous {{font-size: 1.02rem; opacity: .92;}}
.kpi {{background: white; border: 1px solid {GRILLE}; border-radius: 14px; padding: 14px 14px 12px; height: 100%;
    box-shadow: 0 2px 6px rgba(0,0,0,.05);}}
.kpi-icone {{font-size: 1.4rem;}}
.kpi-libelle {{font-size: .82rem; color: {ENCRE_SECONDAIRE}; margin-top: .2rem;}}
.kpi-valeur {{font-size: 1.75rem; font-weight: 800; line-height: 1.2;}}
.kpi-detail {{font-size: .78rem; color: #8a8984;}}
.section {{border-left: 5px solid {ACCENT}; padding: 2px 0 2px 12px; margin: 1.6rem 0 .6rem 0;}}
.section span {{font-size: 1.25rem; font-weight: 800; color: {ENCRE};}}
.section small {{display: block; color: {ENCRE_SECONDAIRE}; font-size: .88rem;}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- Données (en cache)
@st.cache_data(show_spinner="Chargement des 319 335 vols…")
def charger_donnees():
    return charger_vols_propres()


@st.cache_data
def charger_resultats_en_cache():
    return charger_resultats()


@st.cache_data
def charger_mesures(nom):
    fichier = DOSSIER_ECHANTILLONNAGE / f"{nom}.parquet"
    if not fichier.exists():
        return None
    mesures = pd.read_parquet(fichier)
    mesures["nom"] = [f"{m} (repondéré)" if e == "repondéré" else m
                      for m, e in zip(mesures["methode"], mesures["estimateur"])]
    return mesures


@st.cache_resource(show_spinner="Préparation de la population de référence…")
def reference_en_cache():
    return references_population(charger_donnees())


@st.cache_data
def descriptifs_en_cache():
    return statistiques_descriptives(charger_donnees())


@st.cache_data
def profil_en_cache():
    return profil_compagnies(charger_donnees())


# --------------------------------------------------------------------------- Petits outils
def fr(nombre, decimales=1):
    """Écrit un nombre à la française : 18,6 au lieu de 18.6, espace pour les milliers."""
    return f"{nombre:,.{decimales}f}".replace(",", " ").replace(".", ",")


def court(compagnie):
    return "Endeavor" if str(compagnie).startswith("Endeavor") else str(compagnie)


def style(figure, hauteur=400):
    """Style commun : fond clair, grille discrète, survol lisible."""
    figure.update_layout(template="plotly_white", height=hauteur, paper_bgcolor=FOND, plot_bgcolor=FOND,
                         font=dict(family="sans-serif", size=13, color=ENCRE_SECONDAIRE),
                         margin=dict(l=10, r=30, t=30, b=10), hoverlabel=dict(bgcolor="white", font_size=13),
                         legend=dict(orientation="h", y=-0.18, x=0), barcornerradius=4, separators=", ")
    figure.update_xaxes(showgrid=False, linecolor=GRILLE)
    figure.update_yaxes(gridcolor=GRILLE, zeroline=False)
    return figure


def afficher(figure, hauteur=400):
    st.plotly_chart(style(figure, hauteur), width="stretch", config={"displayModeBar": False})


def en_tete(numero, titre, accroche):
    st.markdown(f'<div class="etiquette-chapitre">{numero}</div><div class="titre-chapitre">{titre}</div>'
                f'<div class="accroche">{accroche}</div>', unsafe_allow_html=True)


def a_retenir(texte):
    st.markdown(f'<div class="a-retenir"><b class="titre">À retenir</b>{texte}</div>', unsafe_allow_html=True)


def notion(nom, texte):
    if not st.session_state.get("notes_methode", True):
        return
    st.markdown(f'<div class="notion">📘 <b>Note de méthode — {nom}</b> : {texte}</div>', unsafe_allow_html=True)


def grands_chiffres(chiffres):
    """Une rangée de grandes cartes : [(libellé, valeur, aide), ...]."""
    for colonne, (libelle, valeur, aide) in zip(st.columns(len(chiffres)), chiffres):
        colonne.metric(libelle, valeur, help=aide)


def haltere(etiquettes, debut, fin, couleur_lien=GRIS_CONTEXTE):
    """Traits gris reliant deux valeurs par ligne (graphique « haltère »)."""
    figure = go.Figure()
    for etiquette, a, b in zip(etiquettes, debut, fin):
        if pd.notna(a) and pd.notna(b):
            figure.add_trace(go.Scatter(x=[a, b], y=[etiquette] * 2, mode="lines", showlegend=False, hoverinfo="skip",
                                        line=dict(color=couleur_lien, width=4)))
    return figure


@st.cache_data
def serie_journaliere_en_cache():
    return serie_journaliere(charger_donnees())


@st.cache_data
def annulations_journalieres_par_compagnie():
    v = charger_donnees()
    return v.pivot_table(index="FlightDate", columns="compagnie", values="Cancelled", aggfunc="mean") * 100


def zones_evenements(figure, compagnie=None):
    """Bandes rouges pâles sur les dates des événements réels (tous, ou ceux d'une compagnie)."""
    for numero, ev in enumerate(EVENEMENTS, start=1):
        if compagnie is not None and ev["compagnies"] is not None and compagnie not in ev["compagnies"]:
            continue
        debut = pd.Timestamp(ev["debut"]) - pd.Timedelta(hours=12)
        fin = pd.Timestamp(ev["fin"]) + pd.Timedelta(hours=12)
        figure.add_vrect(x0=debut, x1=fin, fillcolor=ROUGE_RETARD, opacity=0.10, line_width=0,
                         annotation_text=str(numero), annotation_position="top left",
                         annotation_font=dict(size=10, color=ROUGE_RETARD))


def figure_journaliere():
    jour = serie_journaliere_en_cache()
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=jour.index, y=jour["pct_retard"], name="chaque jour", mode="lines",
                                line=dict(color=GRIS_CONTEXTE, width=1),
                                customdata=np.c_[jour["pct_annules"], jour["nb_vols"]],
                                hovertemplate="%{x|%d/%m/%Y} : <b>%{y:.1f} %</b> en retard<br>%{customdata[0]:.1f} % annulés"
                                              "<br>%{customdata[1]:,.0f} vols<extra></extra>"))
    figure.add_trace(go.Scatter(x=jour.index, y=jour["moyenne_7_jours"], name="moyenne sur 7 jours", mode="lines",
                                line=dict(color=ROUGE_RETARD, width=2), hoverinfo="skip"))
    zones_evenements(figure)
    figure.update_xaxes(tickformat="%m/%Y")
    figure.update_yaxes(ticksuffix=" %", range=[0, 72], title="% de vols en retard")
    return figure

@st.cache_resource(show_spinner="Première ouverture : construction de l'entrepôt SQLite (≈ 30 s)…")
def preparer_entrepot():
    """La base SQLite (≈ 160 Mo) n'est pas versionnée : on la reconstruit depuis le parquet au premier démarrage."""
    if not FICHIER_BASE.exists():
        try:
            creer_base(verbeux=False)
        except Exception as erreur:  # le dashboard reste utilisable sans les pages SQL
            st.warning(f"Entrepôt SQL indisponible ({erreur}) : les vues SQL sont remplacées par pandas.")
    return FICHIER_BASE.exists()


preparer_entrepot()
vols = charger_donnees()
R = charger_resultats_en_cache()
POP, IND = R["population"], R["indicateurs"]
APP = R["approfondissement"]
EQ = R.get("equipe")          # base SQL + météo + modèle + destinations (None si la base n'est pas créée)
FIGURES = Path(__file__).resolve().parent.parent / "reports" / "figures"
partis_tous = vols_partis(vols)

# --------------------------------------------------------------------------- Navigation
if "chapitre" not in st.session_state:
    # ?page=3 dans l'adresse ouvre directement la partie 3 (lien direct, captures pour la vidéo)
    page = st.query_params.get("page", "0")
    st.session_state.chapitre = CHAPITRES[int(page)] if page.isdigit() and int(page) < len(CHAPITRES) else CHAPITRES[0]


def aller_a(decalage):
    position = CHAPITRES.index(st.session_state.chapitre) + decalage
    st.session_state.chapitre = CHAPITRES[min(max(position, 0), len(CHAPITRES) - 1)]


with st.sidebar:
    st.markdown("### ✈️ Atlanta, 12 mois de vols")
    st.caption("Tableau de bord en 9 parties, avec un simulateur de risque.")
    st.radio("Chapitre", CHAPITRES, key="chapitre", label_visibility="collapsed")
    st.toggle("Notes de méthode", value=True, key="notes_methode",
              help="Affiche les encadrés qui expliquent la méthode statistique de chaque graphique.")
    position = CHAPITRES.index(st.session_state.chapitre)
    st.progress(position / (len(CHAPITRES) - 1), text=f"Partie {position} sur {len(CHAPITRES) - 1}")
    st.divider()
    st.caption("**En retard** = départ ≥ 15 min après l'heure prévue (définition du BTS).  \n"
               "Données : BTS via Kaggle (Rob Mulla), août 2021 – juillet 2022.")


# =========================================================================== Prologue
def reponse_en_bref():
    """La réponse à la problématique, en trois cartes (meilleure, moins bonne, ce qui compte vraiment)."""
    C, Q = R["compagnies"], APP["quantiles_retard_par_compagnie"]
    RC = EQ["modele"]["rapports_de_cotes"] if EQ else {}
    e, j = C[ENDEAVOR], C["JetBlue"]
    st.markdown("#### La réponse en bref")
    cartes = [
        (f"La meilleure : Endeavor ({fr(e['pct_retard_ou_annule'])} %)",
         f"annule très peu ({fr(e['pct_annules'])} %) ; 90 % de ses vols ont moins de {fr(Q[ENDEAVOR]['p90'], 0)} min de retard ; "
         + (f"même à conditions égales, cote du retard × {fr(RC['Compagnie : ' + ENDEAVOR]['rapport_de_cotes'], 2)} par rapport à Delta." if RC else "")),
        (f"La moins bonne : JetBlue ({fr(j['pct_retard_ou_annule'])} %)",
         f"annule {fr(j['pct_annules'])} % ; {fr(j['pct_grand_retard'])} % de retards de plus de 45 min ; "
         + (f"même à conditions égales, cote × {fr(RC['Compagnie : JetBlue']['rapport_de_cotes'], 2)} ; bases dans le Nord-Est (blizzard)." if RC else "")),
        ("Ce qui fait vraiment la différence",
         "le type de compagnie (bas coût > majeures > régionales), mais surtout <b>quand</b> on vole : l'heure pèse plus que la "
         "compagnie, un avion déjà en retard propage le retard, et les crises et la météo font les mauvais jours."),
    ]
    for colonne, (titre, texte) in zip(st.columns(3), cartes):
        colonne.markdown(f'<div class="carte"><h4>{titre}</h4><p>{texte}</p></div>', unsafe_allow_html=True)
    st.write("")


# =========================================================================== 1. Données
COLONNES_CLES = pd.DataFrame([
    ("FlightDate", "Date du vol", "Temps", "Mois, saison, jour de la semaine"),
    ("CRSDepTime", "Heure de départ prévue", "Temps", "Tranche horaire (matin → soir)"),
    ("Operating_Airline", "Compagnie qui opère le vol", "Catégorielle", "Les 14 compagnies"),
    ("Operated_or_Branded_Code_Share_Partners", "Vol vendu sous une autre marque ?", "Catégorielle",
     "Type « Régionale »"),
    ("DepDelay", "Retard au départ (min)", "Numérique", "Variable centrale"),
    ("DepDel15", "En retard ≥ 15 min (0/1)", "Binaire", "Le % de retard"),
    ("Cancelled", "Vol annulé (vrai/faux)", "Binaire", "« Retardé ou annulé »"),
    ("ArrDelay", "Retard à l'arrivée (min)", "Numérique", "Minutes rattrapées en vol"),
    ("TaxiOut", "Roulage au départ (min)", "Numérique", "Piège de Simpson (chap. 7)"),
    ("Distance", "Distance (miles)", "Numérique", "Profil d'exploitation"),
    ("Tail_Number", "Immatriculation de l'avion", "Identifiant", "Effet domino (chap. 8)"),
], columns=["Colonne", "Signification", "Type", "Sert à"])



def chapitre_donnees():
    en_tete("Chapitre 1", "Les données : une ligne = un vol",
            "Le fichier vient du Bureau of Transportation Statistics (BTS), publié sur Kaggle par Rob Mulla. "
            "Nous gardons les vols au départ d'Atlanta sur 12 mois.")
    grands_chiffres([
        ("Lignes (vols)", fr(POP["nb_vols"], 0), "Unité d'observation : le vol."),
        ("Colonnes du fichier", "61", "9 colonnes constantes retirées, 15 créées → 67 colonnes."),
        ("Destinations", str(POP["nb_destinations"]), None),
        ("Part de Delta", f"{fr(POP['part_delta_pct'])} %", "Atlanta est le « hub » de Delta."),
    ])
    notion("unité d'observation", "avant tout calcul, on dit ce qu'est une ligne. Ici, une ligne = un vol "
           "programmé (pas un avion, pas un passager).")
    regroupement = st.radio("Compter les vols par", ["Compagnie", "Type de compagnie"], horizontal=True)
    if regroupement == "Type de compagnie":
        parts = pd.Series(POP["part_par_type_pct"]).reindex(list(COULEURS_TYPES))
        figure = go.Figure(go.Bar(x=parts.values, y=list(parts.index), orientation="h",
                                  marker_color=[COULEURS_TYPES[t] for t in parts.index],
                                  text=[fr(v) + " %" for v in parts], textposition="outside",
                                  hovertemplate="%{y} : <b>%{x:.1f} %</b> des vols<extra></extra>"))
        figure.update_yaxes(autorange="reversed")
        figure.update_xaxes(ticksuffix=" %", range=[0, 80])
        hauteur = 240
    else:
        compagnies = pd.DataFrame(R["compagnies"]).T.sort_values("nb_vols")
        figure = go.Figure(go.Bar(x=compagnies["nb_vols"].astype(float), y=[court(c) for c in compagnies.index],
                                  orientation="h", marker_color=[COULEURS_TYPES[t] for t in compagnies["type"]],
                                  text=[fr(v, 0) for v in compagnies["nb_vols"]], textposition="outside",
                                  customdata=compagnies[["type", "part_vols_pct"]].to_numpy(),
                                  hovertemplate="%{y} (%{customdata[0]})<br><b>%{x:,.0f}</b> vols, "
                                                "%{customdata[1]:.1f} %<extra></extra>"))
        figure.update_xaxes(type="log", range=[np.log10(500), np.log10(compagnies["nb_vols"].max() * 4)],
                            title="Nombre de vols (échelle log : sinon Delta écrase les autres)")
        hauteur = 470
    afficher(figure, hauteur)
    st.caption("Couleur = type de compagnie : ■ Majeure (bleu foncé) · ■ Régionale (bleu clair) · ■ Bas coût (orange).")
    a_retenir(f"Delta fait à elle seule <b>{fr(POP['part_delta_pct'])} %</b> des vols. Un pourcentage « global » "
              "décrit donc surtout Delta : pour comparer les compagnies, il faut les regarder <b>une par une</b>.")
    st.markdown("#### Les colonnes qui servent à répondre à la question")
    st.dataframe(COLONNES_CLES, hide_index=True, width="stretch")
    st.markdown("#### Derrière les données : comment fonctionne ce monde")
    faits = APP["faits_metier"]
    for i in range(0, len(faits), 2):
        for colonne, fait in zip(st.columns(2), faits[i:i + 2]):
            colonne.markdown(f'<div class="carte"><h4>{fait["theme"]}</h4><p>{fait["fait"]}<br><b>Dans nos données :</b> '
                             f'{fait["lien_donnees"]} <a href="{fait["source"]}" target="_blank">source</a></p></div>',
                             unsafe_allow_html=True)
        st.write("")
    if EQ:
        notion("diagnostic avec le monde réel", "la compagnie de code OH s'appelle « Comair » dans le fichier, mais Comair a "
               "fermé en 2012. En 2021-2022, OH = PSA Airlines, régionale d'American (ses vols sont vendus sous la marque "
               "American). Nom périmé : on garde le code, on signale le nom.")


# =========================================================================== 2. Qualité
COLONNES_CREEES = {"heure_depart_prevue", "jour_semaine", "tranche_horaire", "mois", "annee_mois", "saison",
                   "compagnie", "type_compagnie", "reseau_commercial", "statut", "classe_retard",
                   "retard_ou_annule", "minutes_rattrapees", "rang_rotation", "vol_precedent_en_retard"}


@st.cache_data
def manquants_en_cache():
    colonnes = [c for c in vols.columns if c not in COLONNES_CREEES]
    tous = vols[colonnes].isna().mean() * 100
    tous = tous[tous > 0].sort_values()
    realises = vols.loc[~vols["Cancelled"] & ~vols["Diverted"], tous.index].isna().mean() * 100
    return pd.DataFrame({"tous": tous, "realises": realises})


def chapitre_qualite():
    Q = R["qualite"]
    en_tete("Chapitre 2", "Qualité : ce qui manque n'est pas manquant au hasard",
            "Avant d'analyser, on vérifie le fichier : doublons, valeurs manquantes, cas impossibles.")
    grands_chiffres([
        ("Lignes en double", fr(Q["nb_lignes_en_double"], 0), None),
        ("Vols annulés", fr(Q["nb_annules"], 0), "1,4 % des vols programmés."),
        ("Sans retard mesuré", fr(Q["nb_annules_sans_retard_mesure"], 0), "Vols annulés avant le départ."),
        ("Sans immatriculation", fr(Q["nb_sans_immatriculation"], 0), "Tous des vols annulés."),
    ])
    choix = st.radio("Regarder les valeurs manquantes",
                                 ["Sur tous les vols", "Sans les vols annulés ou déroutés"], horizontal=True)
    sans_annules = choix == "Sans les vols annulés ou déroutés"
    manquants = manquants_en_cache()["realises" if sans_annules else "tous"]
    figure = go.Figure(go.Bar(x=manquants.values, y=list(manquants.index), orientation="h",
                              marker_color=ACCENT if sans_annules else GRIS_FONCE,
                              text=[fr(v, 2) + " %" for v in manquants], textposition="outside",
                              hovertemplate="%{y} : <b>%{x:.2f} %</b> manquants<extra></extra>"))
    figure.update_xaxes(ticksuffix=" %", range=[0, manquants_en_cache()["tous"].max() * 1.25], title="% des vols sans valeur")
    afficher(figure, 120 + 22 * len(manquants))
    if sans_annules:
        a_retenir("Sans les vols annulés ou déroutés, <b>toutes les barres tombent à 0</b>. "
                  "Les manquants arrivent <b>en bloc</b> : un vol annulé n'a ni heure de départ ni retard. "
                  "On ne les remplit donc pas, on les explique.")
    else:
        a_retenir("Environ 1,4 % des valeurs manquent dans les colonnes du départ, un peu plus pour l'arrivée. "
                  "👉 Cliquez sur « Sans les vols annulés ou déroutés » pour voir <b>d'où</b> viennent ces trous.")
    notion("motif des manquants", "on ne remplit pas un trou sans savoir pourquoi il existe. Ici le motif est "
           "« en bloc » : il suit une règle (le vol n'a pas eu lieu).")
    notion("biais du survivant", "si on calcule le retard seulement sur les vols partis, on oublie les annulés. "
           f"D'où l'indicateur « retardé ou annulé » ({fr(IND['pct_retard_ou_annule'])} % au lieu de {fr(IND['pct_retard'])} %).")
    section("Le nettoyage : retirer, garder, créer", "on ne modifie jamais le fichier brut : on crée une copie propre")
    st.dataframe(pd.DataFrame([
        ("Retiré", "9 colonnes constantes (ex. : l'aéroport de départ, toujours ATL)", "elles n'apportent aucune information"),
        ("Gardé", f"les {fr(Q['nb_annules'], 0)} vols annulés", "ce sont de vrais événements ; les retirer crée un biais du survivant"),
        ("Créé", "statut, classe de gravité, « retardé ou annulé »", "lire la gravité et corriger le biais du survivant"),
        ("Créé", "heure, tranche horaire, mois, saison, jour de la semaine", "comparer « à heure égale » et « à saison égale »"),
        ("Créé", "type de compagnie (régionale lue dans les données)", "la variable explicative principale"),
        ("Créé", "minutes rattrapées en vol, rang de l'avion, vol précédent en retard", "l'effet domino"),
    ], columns=["Action", "Quoi", "Pourquoi"]), hide_index=True, width="stretch")
    E = APP["valeurs_extremes"]
    cartes_kpi([
        ("⏳", "Retards de plus de 6 h", fr(E["nb_retards_plus_6h"], 0), "valeurs suspectes, gardées", ORANGE),
        ("🕰️", "Retards de plus de 24 h", fr(E["nb_retards_plus_24h"], 0), f"maximum : {fr(E['retard_max_min'] / 60, 1)} h", ROUGE_RETARD),
        ("✅", "Erreurs prouvées", "0", "rien n'est supprimé sans preuve", "#1d8a5a"),
    ])
    a_retenir("Les valeurs extrêmes sont classées selon le plan de travail (erreur vérifiée, correction justifiable, suspecte, "
              "indéterminée) : aucune n'est une erreur prouvée, on les <b>garde et on les signale</b>.")
    notion("non-destruction", "diagnostiquer avant de transformer ; ne rien supprimer sans preuve.")


# =========================================================================== 3. Descriptives
def chapitre_descriptif():
    en_tete("Chapitre 3", "Le portrait en chiffres : la moyenne ment un peu",
            "Pour chaque variable numérique, la grille classique : centre, dispersion, forme, extrêmes.")
    descriptifs = descriptifs_en_cache()
    nom = st.selectbox("Variable", list(descriptifs.index), key="variable_descriptive")
    d = descriptifs.loc[nom]
    grands_chiffres([
        ("Moyenne", fr(d["moyenne"]), "Centre sensible aux valeurs extrêmes."),
        ("Médiane", fr(d["médiane"]), "La moitié des vols est en dessous."),
        ("Écart-type", fr(d["écart-type"]), "Dispersion autour de la moyenne."),
        ("Atypiques (règle IQR)", f"{fr(d['atypiques (%)'])} %", "Hors de [Q1 − 1,5·IQR ; Q3 + 1,5·IQR]."),
    ])
    iqr = d["Q3"] - d["Q1"]
    bas, haut = max(d["min"], d["Q1"] - 1.5 * iqr), min(d["max"], d["Q3"] + 1.5 * iqr)
    figure = go.Figure(go.Box(q1=[d["Q1"]], median=[d["médiane"]], q3=[d["Q3"]], lowerfence=[bas], upperfence=[haut],
                              mean=[d["moyenne"]], y=[nom], orientation="h", boxmean=True, fillcolor="#dbe7f7",
                              line=dict(color=ACCENT), name=nom, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=[d["P90"], d["P99"]], y=[nom, nom], mode="markers+text", text=["P90", "P99"],
                                textposition="top center", marker=dict(color=ROUGE_RETARD, size=10, symbol="diamond"),
                                hovertemplate="%{text} : <b>%{x:.1f}</b><extra></extra>", showlegend=False))
    figure.update_layout(showlegend=False)
    figure.update_xaxes(title=f"{nom} — boîte = 50 % du milieu (Q1 → Q3), trait = médiane, pointillé = moyenne")
    st.markdown(f"#### La boîte à moustaches de « {nom} » (maximum hors du graphique : {fr(d['max'], 0)})")
    afficher(figure, 260)
    ecart = d["moyenne - médiane"]
    if ecart > 1:
        lecture = (f"La moyenne ({fr(d['moyenne'])}) est <b>au-dessus</b> de la médiane ({fr(d['médiane'])}) : "
                   f"distribution <b>asymétrique à droite</b>. Quelques très grandes valeurs (P99 = {fr(d['P99'])}, "
                   f"max = {fr(d['max'], 0)}) tirent la moyenne vers le haut. La médiane décrit mieux un vol « typique ».")
    elif ecart < -1:
        lecture = (f"La moyenne ({fr(d['moyenne'])}) est <b>sous</b> la médiane ({fr(d['médiane'])}) : "
                   "asymétrie à gauche, quelques petites valeurs extrêmes.")
    else:
        lecture = (f"Moyenne et médiane sont proches ({fr(d['moyenne'])} et {fr(d['médiane'])}) : "
                   "distribution à peu près symétrique.")
    a_retenir(lecture)
    notion("détection ≠ décision", "la règle IQR <i>détecte</i> des valeurs atypiques ; elle ne dit pas de les "
           "supprimer. Un retard de 3 heures est un vrai retard : on le garde.")
    st.markdown("#### Le tableau descriptif complet (les 5 variables numériques)")
    st.dataframe(descriptifs.style.format(precision=1), width="stretch")
    a_retenir("Pour les cinq variables, la moyenne est différente de la médiane : il faut toujours regarder <b>les deux</b>, "
              "et les centiles (P90, P99) pour voir la queue. La forme complète de chaque variable est au chapitre 5.")
    notion("boîte à moustaches", "la boîte va de Q1 à Q3 (50 % des vols), le trait est la médiane ; les moustaches s'arrêtent à "
           "1,5 × IQR ; au-delà, ce sont des valeurs atypiques (détectées, pas supprimées).")


# =========================================================================== 4. Échantillonnage
FICHES_METHODES = {
    "Aléatoire simple": ("On tire les vols au hasard : chacun a la même chance.",
                         "un chiffre global (le % de retard de l'année)."),
    "Systématique": ("On trie les vols par date et heure, puis on garde 1 vol tous les k (k = N / n ≈ 64).",
                     "couvrir toute l'année de façon régulière."),
    "Stratifié par compagnie": ("On tire le MÊME nombre de vols dans chaque compagnie (357 chacune), "
                                "puis on repondère pour les chiffres globaux.",
                                "comparer les compagnies, même les petites."),
    "Par grappes (jours)": ("On tire des journées entières au hasard et on garde tous leurs vols.",
                            "collecter vite… mais les vols d'un même jour se ressemblent (orages, pannes)."),
    "Temporel (par jour)": ("On tire dans chacun des 365 jours, en proportion de ses vols.",
                            "suivre l'évolution mois par mois."),
    "Répété (bootstrap)": ("On tire avec remise : un même vol peut sortir deux fois. On répète pour mesurer la variabilité.",
                           "mesurer l'incertitude d'un chiffre."),
}
OBJECTIFS = {
    "E6_erreur_pct_retard_global": "Estimer le % de retard global",
    "E6_erreur_max_compagnie": "Estimer le % de retard de chaque compagnie (pire erreur)",
    "E2_nb_compagnies_moins_30_vols": "Garder assez de vols dans chaque compagnie",
    "E5_ecart_moyen_serie_mensuelle": "Suivre l'évolution mois par mois",
    "E3_erreur_ecart_domino": "Mesurer l'effet domino",
    "E1_ks_retard": "Reproduire la forme de la distribution du retard",
}


def onglet_taille():
    T = R["taille_echantillon"]
    st.markdown("#### La taille n se calcule, elle ne se choisit pas au hasard")
    st.latex(r"n_0 = \frac{z^2\,p\,(1-p)}{e^2} \qquad n = \frac{n_0}{1 + \frac{n_0 - 1}{N}}")
    colonne_e, colonne_p = st.columns(2)
    e = colonne_e.slider("Marge d'erreur acceptée e (en points)", 0.5, 5.0, 1.0, 0.25, format="± %.2f")
    p = colonne_p.slider("Proportion attendue p (en %)", 5.0, 50.0, round(T["p"] * 100, 1), 0.1,
                         help="18,6 % = le % de retard observé. 50 % = le cas le plus prudent.")
    n = taille_echantillon(T["N"], p / 100, e / 100)
    marge_5000 = 100 * marge_erreur(TAILLE_ECHANTILLON, T["N"], p / 100)
    grands_chiffres([
        ("Population N", fr(T["N"], 0), None),
        ("Confiance", "95 %", "z = 1,96 dans la formule."),
        ("Taille nécessaire n", fr(n, 0), "Après correction pour population finie."),
        ("Marge avec n = 5 000", f"± {fr(marge_5000, 2)} pt", None),
    ])
    tailles = np.arange(500, 20001, 250)
    marges = 100 * marge_erreur(tailles, T["N"], p / 100)
    figure = go.Figure(go.Scatter(x=tailles, y=marges, mode="lines", line=dict(color=ACCENT, width=2),
                                  hovertemplate="n = %{x:,} vols → marge ± <b>%{y:.2f}</b> pt<extra></extra>"))
    figure.add_trace(go.Scatter(x=[TAILLE_ECHANTILLON], y=[marge_5000], mode="markers+text",
                                marker=dict(size=12, color=ROUGE_RETARD, line=dict(color="white", width=2)),
                                text=[f"n = 5 000 → ± {fr(marge_5000, 2)} pt"], textposition="top right", hoverinfo="skip"))
    figure.update_layout(showlegend=False)
    figure.update_xaxes(title="Taille de l'échantillon n")
    figure.update_yaxes(title="Marge d'erreur (points)", rangemode="tozero")
    afficher(figure, 320)
    a_retenir(f"Pour connaître le % de retard à ± 1 point près, il faut environ <b>{fr(T['n_pour_1_point'], 0)}</b> vols "
              f"(<b>{fr(T['n_pour_1_point_p_0_5'], 0)}</b> si on ne connaît pas p). Avec <b>5 000 vols</b> "
              f"({fr(T['fraction_pct'])} % de la population), la marge est de ± {fr(T['marge_n_retenu_points'], 2)} point. "
              f"Mais une petite compagnie comme Mesa n'aurait que ~{T['mesa']['n_h_si_aleatoire_5000']} vols au hasard, "
              f"alors qu'il lui en faut {T['mesa']['n_h_pour_5_points']} pour ± 5 points → il faut <b>stratifier</b>.")


def onglet_labo():
    colonne_m, colonne_t, colonne_g = st.columns([3, 2, 1])
    nom = colonne_m.selectbox("Méthode d'échantillonnage", list(METHODES))
    taille = colonne_t.select_slider("Taille", [1000, 2000, 5000, 10000, 20000], value=TAILLE_ECHANTILLON)
    graine = int(colonne_g.number_input("Graine", 0, 9999, GRAINE, help="Même graine = même échantillon."))
    comment, bon_pour = FICHES_METHODES[nom]
    st.markdown(f'<div class="carte"><h4>{nom}</h4><p>{comment}<br><b>Bonne pour :</b> {bon_pour}</p></div>',
                unsafe_allow_html=True)
    st.write("")
    echantillon = METHODES[nom](vols, taille=taille, graine=graine)
    poids = poids_de_sondage(echantillon, vols) if nom in METHODES_A_REPONDERER else None
    resultat = mesurer_echantillon(echantillon, reference_en_cache(), poids)
    conservees = sum(bool(resultat[c]) for c in CONCLUSIONS)
    grands_chiffres([
        ("Vols tirés", fr(len(echantillon), 0), None),
        ("Erreur sur le % global", f"{fr(resultat['E6_erreur_pct_retard_global'], 2)} pt",
         "Écart absolu |échantillon − population|."),
        ("Compagnies comparables", f"{resultat['E6_nb_compagnies_comparables']} / 14", "Au moins 30 vols partis."),
        ("Conclusions conservées", f"{conservees} / {len(CONCLUSIONS)}", None),
    ])
    population = partis_tous.groupby("compagnie")["DepDel15"].mean() * 100
    dans_ech = vols_partis(echantillon).groupby("compagnie")["DepDel15"].agg(["mean", "size"]).reindex(
        population.sort_values().index)
    dans_ech["mean"] *= 100
    dans_ech["size"] = dans_ech["size"].fillna(0)
    etiquettes = [court(c) for c in dans_ech.index]
    figure = haltere(etiquettes, population[dans_ech.index], dans_ech["mean"])
    figure.add_trace(go.Scatter(x=population[dans_ech.index], y=etiquettes, mode="markers", name="Population (vraie valeur)",
                                marker=dict(color=GRIS_FONCE, size=14, symbol="line-ns-open", line=dict(width=3)),
                                hovertemplate="%{y} : vraie valeur <b>%{x:.1f} %</b><extra></extra>"))
    figure.add_trace(go.Scatter(x=dans_ech["mean"], y=etiquettes, mode="markers", name="Échantillon",
                                marker=dict(color=[ACCENT if n >= N_MIN else "#e59a9a" for n in dans_ech["size"]],
                                            size=12, line=dict(color="white", width=2)),
                                customdata=dans_ech["size"].to_numpy(),
                                hovertemplate="%{y} : échantillon <b>%{x:.1f} %</b><br>%{customdata:.0f} vols tirés<extra></extra>"))
    figure.update_xaxes(title="% de vols en retard", ticksuffix=" %")
    st.markdown("#### Chaque compagnie : la vraie valeur (trait) et ce que voit l'échantillon (point)")
    afficher(figure, 470)
    st.caption("Point bleu = au moins 30 vols tirés. Point rose = moins de 30 vols : pourcentage non fiable.")
    st.markdown("**Les 8 conclusions de l'étude, dans cet échantillon :**")
    colonnes = st.columns(2)
    for i, (code, texte) in enumerate(CONCLUSIONS.items()):
        colonnes[i % 2].markdown(f"{'✅' if resultat[code] else '❌'} **{code}** {texte}")
    a_retenir("Changez la méthode et la graine : avec l'<b>aléatoire simple</b>, les petites compagnies ont trop "
              "peu de vols (points roses) ; avec le <b>stratifié par compagnie</b>, toutes sont mesurées. "
              "Avec les <b>grappes de jours</b>, l'erreur saute d'un tirage à l'autre.")


def onglet_etude():
    mesures = charger_mesures("mesures_methodes")
    if mesures is None:
        st.info("Lancez `python src/sampling_study.py` pour calculer l'étude (environ 20 minutes).")
        return
    cle = st.selectbox("Objectif", list(OBJECTIFS), format_func=OBJECTIFS.get)
    resume = mesures.groupby("nom")[cle].quantile([0.1, 0.5, 0.9]).unstack().sort_values(0.5, ascending=False)
    meilleure = resume.index[-1]
    unite = DESCRIPTION_MESURES[cle][2]
    figure = go.Figure()
    for methode, ligne in resume.iterrows():
        gagnante = methode == meilleure
        figure.add_trace(go.Scatter(x=[ligne[0.1], ligne[0.9]], y=[methode] * 2, mode="lines", showlegend=False,
                                    line=dict(color=ACCENT_CLAIR if gagnante else GRIS_CONTEXTE, width=6), hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=[ligne[0.5]], y=[methode], mode="markers+text", showlegend=False,
                                    marker=dict(color=ACCENT if gagnante else GRIS_FONCE, size=13,
                                                line=dict(color="white", width=2)),
                                    text=[fr(ligne[0.5], 2)], textposition="top center",
                                    hovertemplate=f"{methode}<br>médiane <b>%{{x:.2f}}</b> {unite}<br>80 % des tirages "
                                                  f"entre {fr(ligne[0.1], 2)} et {fr(ligne[0.9], 2)}<extra></extra>"))
    figure.update_xaxes(title=f"{DESCRIPTION_MESURES[cle][1]} ({unite}) — plus petit = mieux", rangemode="tozero")
    st.markdown(f"#### 100 tirages de 5 000 vols par méthode — objectif : {OBJECTIFS[cle].lower()}")
    afficher(figure, 400)
    st.caption("Point = médiane des 100 tirages ; barre = 80 % des tirages (du 10ᵉ au 90ᵉ centile).")
    a_retenir(f"Pour cette question, la méthode adaptée est <b>{meilleure}</b> ; la moins bonne est "
              f"<b>{resume.index[0]}</b>. Aucune méthode ne gagne partout : <b>le bon échantillon dépend de la question</b>.")

    st.markdown("#### Les conclusions survivent-elles ? (% des 100 tirages où elles restent vraies)")
    ordre = [n for m in METHODES for n in (m, f"{m} (repondéré)")]
    tableau = (mesures.groupby("nom")[list(CONCLUSIONS)].mean() * 100).reindex(ordre).dropna(how="all")
    figure = go.Figure(go.Heatmap(z=tableau.values, x=list(tableau.columns), y=list(tableau.index),
                                  colorscale=ECHELLE_ACCENT, zmin=0, zmax=100, showscale=False, xgap=2, ygap=2,
                                  text=[[f"{v:.0f} %" for v in ligne] for ligne in tableau.values], texttemplate="%{text}",
                                  customdata=[[CONCLUSIONS[c] for c in tableau.columns]] * len(tableau),
                                  hovertemplate="%{y}<br>%{x} : %{customdata}<br><b>%{z:.0f} %</b> des tirages<extra></extra>"))
    figure.update_xaxes(side="top")
    figure.update_yaxes(autorange="reversed")
    afficher(figure, 340)
    st.caption(" · ".join(f"**{c}** {t}" for c, t in CONCLUSIONS.items()))
    pire_cas = tableau.min()
    solides = [c for c in CONCLUSIONS if pire_cas[c] >= 90]
    fragiles = [c for c in CONCLUSIONS if tableau[c].max() < 50]
    a_retenir(f"Les <b>grandes</b> différences ({', '.join(solides)}) restent vraies dans au moins 90 % des tirages, "
              f"quelle que soit la méthode. Les écarts <b>fins</b> ({', '.join(fragiles)}) se perdent : moins de 50 % "
              "des tirages, même avec la meilleure méthode. C3 (Endeavor la plus ponctuelle) n'est vérifiable "
              "qu'avec le <b>stratifié par compagnie</b>, car les autres méthodes laissent des compagnies sans assez de vols.")

    tailles = charger_mesures("mesures_tailles")
    if tailles is not None:
        st.markdown("#### Et si on prend plus de vols ? (erreur sur la compagnie la plus mal estimée)")
        courbe = tailles.groupby(["nom", "taille"])["E6_erreur_max_compagnie"].mean().reset_index()
        figure = go.Figure()
        for methode, groupe in courbe.groupby("nom"):
            figure.add_trace(go.Scatter(x=groupe["taille"], y=groupe["E6_erreur_max_compagnie"], name=methode,
                                        mode="lines+markers", marker=dict(size=8),
                                        line=dict(width=2, color=COULEURS_METHODES_TAILLE.get(methode, GRIS_FONCE)),
                                        hovertemplate=f"{methode}<br>n = %{{x:,}} : <b>%{{y:.1f}}</b> points<extra></extra>"))
        figure.update_xaxes(type="log", title="Taille de l'échantillon (échelle log)",
                            tickvals=[1000, 2000, 5000, 10000, 20000], ticktext=["1 000", "2 000", "5 000", "10 000", "20 000"])
        figure.update_yaxes(title="Erreur maximale (points)", rangemode="tozero")
        afficher(figure, 360)


def chapitre_echantillonnage():
    en_tete("Chapitre 4", "Peut-on se contenter de 5 000 vols ?",
            "Nous avons la population entière : nous pouvons donc <b>vérifier</b> ce qu'un échantillon garde "
            "ou perd, avec 6 plans d'échantillonnage classiques.")
    taille, labo, etude, inter, reponses = st.tabs(["① Combien de vols ?", "② Tirer un échantillon",
                                                    "③ Par question : ce que l'échantillon m'apprend", "④ Inter et intra",
                                                    "⑤ Ce que l'échantillon répond seul"])
    with taille:
        onglet_taille()
    with labo:
        onglet_labo()
    with etude:
        onglet_etude()
    with inter:
        onglet_inter_intra()
    with reponses:
        onglet_reponses_echantillon()


def onglet_inter_intra():
    mesures = charger_mesures("mesures_methodes")
    if mesures is None:
        st.info("Lancez `python src/sampling_study.py` pour calculer l'étude.")
        return
    naif = mesures[mesures["estimateur"] == "naïf"]
    st.markdown("#### Inter-échantillons : d'un tirage à l'autre, avec la même méthode (100 tirages de 5 000 vols)")
    ordre = naif.groupby("methode")["E6_erreur_pct_retard_global"].median().sort_values().index
    figure = go.Figure()
    for methode in ordre:
        valeurs = naif.loc[naif["methode"] == methode, "E6_erreur_pct_retard_global"]
        figure.add_trace(go.Box(x=valeurs, name=methode, orientation="h", boxpoints="all", jitter=0.4, pointpos=0,
                                marker=dict(size=4, color=ROUGE_RETARD if methode.startswith("Par grappes") else ACCENT),
                                line=dict(color=GRIS_FONCE), fillcolor="rgba(0,0,0,0)",
                                hovertemplate=methode + " : erreur <b>%{x:.2f}</b> point<extra></extra>"))
    figure.update_layout(showlegend=False)
    figure.update_xaxes(title="Erreur sur le % de retard global (points) — un point = un tirage", rangemode="tozero")
    afficher(figure, 380)
    T = R["taille_echantillon"]
    alea = naif.loc[naif["methode"] == "Aléatoire simple", "E6_erreur_pct_retard_global"]
    a_retenir(f"La théorie se vérifie : le calcul promettait ± {fr(T['marge_n_retenu_points'], 2)} point avec 5 000 vols ; "
              f"au hasard, 95 % des tirages se trompent de moins de <b>{fr(alea.quantile(0.95), 2)}</b> point. "
              "Les <b>grappes de jours</b> sont instables : l'erreur change beaucoup d'un tirage à l'autre.")
    notion("variabilité d'échantillonnage", "chaque tirage donne un résultat un peu différent ; on la mesure en répétant le tirage.")
    DV = APP.get("decomposition_variance")
    if DV:
        st.markdown("#### Intra et inter groupes : les vols d'un même groupe se ressemblent-ils ?")
        noms = {"FlightDate": "Jours", "compagnie": "Compagnies", "tranche_horaire": "Tranches horaires", "saison": "Saisons"}
        parts = pd.Series({noms[c]: DV[c]["part_inter_pct"] for c in noms})
        figure = go.Figure(go.Bar(x=parts.values, y=list(parts.index), orientation="h",
                                  marker_color=[ROUGE_RETARD if n == "Jours" else GRIS_FONCE for n in parts.index],
                                  text=[fr(v, 1) + " %" for v in parts], textposition="outside",
                                  hovertemplate="%{y} : <b>%{x:.2f} %</b> de la variance entre groupes<extra></extra>"))
        figure.update_yaxes(autorange="reversed")
        figure.update_xaxes(title="Part de la variance du retard qui vient des différences ENTRE groupes (η²)", ticksuffix=" %")
        afficher(figure, 260)
        G = DV["grappes_jours"]
        grands_chiffres([
            ("Corrélation intra-classe (jours)", fr(DV["FlightDate"]["icc"], 3), "Les vols d'un même jour se ressemblent."),
            ("Effet de plan", f"≈ {fr(G['effet_de_plan'], 0)}", "1 + (m − 1) × ICC, avec m ≈ 863 vols par jour."),
            ("5 000 vols en grappes valent", f"≈ {G['taille_efficace']} vols", "tirés au hasard."),
            ("Erreur attendue", f"± {fr(G['marge_attendue_points'], 1)} pts", "Observée : 95 % des tirages < 7,6."),
        ])
        a_retenir("Les vols d'un même jour se ressemblent (même météo, mêmes incidents) : prendre des jours entiers apporte peu "
                  "d'information. <b>Le calcul de l'effet de plan explique l'échec des grappes</b>, et c'est une preuve de plus des crises.")
        notion("strates et grappes", "une bonne <b>strate</b> est homogène à l'intérieur et différente des autres ; une bonne "
               "<b>grappe</b> est variée à l'intérieur (une mini-population). Nos jours sont trop homogènes : de mauvaises grappes.")


# =========================================================================== 5. Univarié
VARIABLES_CATEGORIELLES = {"classe_retard": "Classe de retard", "tranche_horaire": "Tranche horaire",
                           "jour_semaine": "Jour de la semaine", "saison": "Saison", "type_compagnie": "Type de compagnie"}
ORDRES_CATEGORIES = {"classe_retard": CLASSES_RETARD, "tranche_horaire": NOMS_TRANCHES, "jour_semaine": ORDRE_JOURS,
                     "saison": ORDRE_SAISONS, "type_compagnie": list(COULEURS_TYPES)}


def chapitre_univarie():
    section("Univarié : une variable à la fois, sa forme",
            "avant de croiser, on regarde chaque variable seule : où sont la plupart des vols ? y a-t-il une longue traîne ?")
    genre = st.radio("Type de variable", ["Numérique", "Catégorielle"], horizontal=True)
    if genre == "Catégorielle":
        colonne = st.selectbox("Variable", list(VARIABLES_CATEGORIELLES), format_func=VARIABLES_CATEGORIELLES.get,
                               key="variable_categorielle")
        parts = vols[colonne].value_counts(normalize=True).mul(100).reindex(ORDRES_CATEGORIES[colonne]).dropna()
        couleurs = [COULEURS_TYPES[c] for c in parts.index] if colonne == "type_compagnie" else [GRIS_FONCE] * len(parts)
        figure = go.Figure(go.Bar(x=list(parts.index), y=parts.values, marker_color=couleurs,
                                  text=[fr(v) + " %" for v in parts], textposition="outside",
                                  hovertemplate="%{x} : <b>%{y:.1f} %</b> des vols<extra></extra>"))
        figure.update_yaxes(ticksuffix=" %", range=[0, parts.max() * 1.2])
        afficher(figure, 380)
        a_retenir(f"La modalité la plus fréquente est <b>{parts.idxmax()}</b> ({fr(parts.max())} % des vols). "
                  "Une distribution déséquilibrée signifie qu'une moyenne globale reflète surtout cette modalité.")
        notion("fréquences", "pour une variable catégorielle, on décrit les effectifs et les pourcentages, "
               "dans un ordre fixe (pas trié au hasard).")
        return
    colonne = st.selectbox("Variable", list(VARIABLES_DESCRIPTIVES), format_func=VARIABLES_DESCRIPTIVES.get,
                           key="variable_numerique")
    valeurs = vols[colonne].dropna()
    queue = st.toggle("Montrer toute la traîne (jusqu'au maximum, échelle log)", value=False,
                      help="Par défaut on coupe aux centiles 1 et 99 pour voir le cœur de la distribution.")
    bas, haut = (valeurs.min(), valeurs.max()) if queue else (valeurs.quantile(0.01), valeurs.quantile(0.99))
    effectifs, bords = np.histogram(valeurs[valeurs.between(bas, haut)], bins=70)
    centres = (bords[:-1] + bords[1:]) / 2
    figure = go.Figure(go.Bar(x=centres, y=effectifs, width=(bords[1] - bords[0]) * 0.9, marker_color=GRIS_CONTEXTE,
                              customdata=np.c_[bords[:-1], bords[1:]],
                              hovertemplate="de %{customdata[0]:.0f} à %{customdata[1]:.0f} : <b>%{y:,}</b> vols<extra></extra>"))
    q1, mediane, q3, moyenne = valeurs.quantile(0.25), valeurs.median(), valeurs.quantile(0.75), valeurs.mean()
    asymetrie = valeurs.skew()
    grands_chiffres([
        ("Médiane", fr(mediane), "Trait bleu sur le graphique."),
        ("Moyenne", fr(moyenne), "Pointillé rouge sur le graphique."),
        ("50 % des vols entre", f"{fr(q1, 0)} et {fr(q3, 0)}", "Q1 et Q3 : zone bleue sur le graphique."),
        ("Asymétrie", fr(asymetrie, 2), "0 = symétrique ; > 1 = longue traîne à droite."),
    ])
    figure.add_vrect(x0=q1, x1=q3, fillcolor=ACCENT, opacity=0.10, line_width=0)
    figure.add_vline(x=mediane, line_color=ACCENT, line_width=2)
    figure.add_vline(x=moyenne, line_color=ROUGE_RETARD, line_width=2, line_dash="dot")
    # Légende : des traces vides qui expliquent les repères (au lieu d'étiquettes qui se chevauchent)
    for nom, ligne in [("Médiane", dict(color=ACCENT, width=2)), ("Moyenne", dict(color=ROUGE_RETARD, width=2, dash="dot"))]:
        figure.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nom, line=ligne))
    figure.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="50 % des vols (Q1 → Q3)",
                                marker=dict(symbol="square", size=12, color=ACCENT, opacity=0.25)))
    figure.update_yaxes(type="log" if queue else "linear", title="Nombre de vols" + (" (échelle log)" if queue else ""))
    figure.update_xaxes(title=VARIABLES_DESCRIPTIVES[colonne])
    figure.data[0].showlegend = False
    afficher(figure, 420)
    a_retenir(f"Coefficient d'asymétrie = <b>{fr(asymetrie, 2)}</b>. "
              + ("Une <b>longue traîne à droite</b> : la plupart des vols sont proches de la médiane, quelques-uns très "
                 "loin. C'est pour cela qu'on regarde la médiane et les centiles, pas seulement la moyenne."
                 if asymetrie > 1 else "Distribution assez équilibrée autour de son centre."))
    notion("forme", "si la moyenne est à droite de la médiane, la distribution est asymétrique à droite.")


# =========================================================================== 6. Bivarié
INDICATEURS = {"pct_retard_ou_annule": "% retardés ou annulés", "pct_retard": "% en retard (vols partis)",
               "pct_annules": "% annulés", "pct_grand_retard": "% de grands retards (> 45 min)",
               "retard_p90_min": "Retard P90 (min)"}


def onglet_temps():
    st.markdown("#### Temps × retard, jour par jour")
    afficher(figure_journaliere(), 380)
    J = APP["journalier"]
    grands_chiffres([
        ("Meilleur jour", f"{fr(J['jour_min']['pct'])} %", J["jour_min"]["date"]),
        ("Pire jour", f"{fr(J['jour_max']['pct'])} %", J["jour_max"]["date"]),
        ("Jours > 40 %", str(J["nb_mauvais_jours"]), None),
        ("Retards dans les 10 % pires jours", f"{fr(J['pct_retards_dans_10pct_pires_jours'])} %", None),
    ])
    a_retenir("Le retard arrive par <b>crises</b>, et une crise dure souvent plusieurs jours (corrélation d'un jour au "
              f"lendemain : {fr(J['autocorrelation_1_jour'], 2)}). Les numéros renvoient aux événements réels ci-dessous.")
    st.markdown("#### Les mêmes 365 jours, avec les saisons")
    afficher(figure_saisons(), 380)
    saisons = R["retard_par_saison_global_pct"]
    a_retenir(f"Le retard change avec la saison : <b>{fr(saisons['Automne'])} %</b> en automne, {fr(saisons['Hiver'])} % en hiver, "
              f"{fr(saisons['Printemps'])} % au printemps, <b>{fr(saisons['Été'])} %</b> en été. L'été seul donnerait "
              f"{fr(R['zoom_ete_2022']['pct_retard'])} % : c'est pourquoi on étudie une année entière.")
    notion("cadrage", "la période étudiée change la conclusion : un seul été surestime le retard de l'année.")
    st.markdown("#### Les crises de l'année, vérifiées dans nos données et documentées")
    evts = pd.DataFrame(APP["evenements"])
    evts.index = range(1, len(evts) + 1)
    tableau = pd.DataFrame({
        "Dates": [d if d == f else f"{d} → {f}" for d, f in zip(evts["debut"], evts["fin"])],
        "Événement": evts["nom"], "Compagnies": evts["compagnies"],
        "Annulés (normal)": [f"{fr(a)} % ({fr(n)} %)" for a, n in zip(evts["pct_annules"], evts["pct_annules_normal"])],
        "En retard (normal)": [f"{fr(a)} % ({fr(n)} %)" for a, n in zip(evts["pct_retard"], evts["pct_retard_normal"])],
        "Ce qui s'est passé": evts["explication"], "Source": evts["source"],
    }, index=evts.index)
    st.dataframe(tableau, width="stretch", column_config={"Source": st.column_config.LinkColumn("Source", display_text="lien")})
    st.caption(f"Deux jours très anormaux ({', '.join(APP['jours_non_expliques'])}) n'avaient pas de source de presse : la météo "
               "les explique (rafales à 77 km/h le 12 mars, 11 h de pluie le 21 juillet). Le blizzard du 29/01 ne concerne que 6 vols JetBlue au départ d'Atlanta.")
    st.markdown("#### Jour de la semaine × retard")
    semaine = pd.DataFrame({j: APP["jour_semaine"][j] for j in ORDRE_JOURS}).T
    couleurs = [ACCENT if j in ("Mardi", "Mercredi") else ROUGE_RETARD if j == "Vendredi" else GRIS_CONTEXTE
                for j in semaine.index]
    figure = go.Figure(go.Bar(x=list(semaine.index), y=semaine["pct_retard"], marker_color=couleurs,
                              text=[f"{fr(v)} %" for v in semaine["pct_retard"]], textposition="outside",
                              customdata=semaine["vols_par_jour"],
                              hovertemplate="%{x} : <b>%{y:.1f} %</b><br>%{customdata:.0f} vols par jour<extra></extra>"))
    figure.update_yaxes(ticksuffix=" %", range=[0, semaine["pct_retard"].max() * 1.25])
    afficher(figure, 320)
    a_retenir(f"Mardi et mercredi sont les plus calmes, vendredi le plus difficile, à chaque saison. Mais le lien est "
              f"<b>faible</b> (V de Cramér = {fr(APP['jour_semaine']['association']['v_cramer'], 2)}) : le jour compte "
              "moins que l'heure.")


def onglet_trois_cas():
    st.markdown("#### Les trois cas du bivarié")
    st.markdown("| Les deux variables | Graphique | Force du lien |\n|---|---|---|\n"
                "| catégorielle × catégorielle | barres à 100 % | V de Cramér |\n"
                "| catégorielle × numérique | médiane, quartile, 90e centile | ε² de Kruskal-Wallis |\n"
                "| numérique × numérique | nuage de points | ρ de Spearman |")
    st.markdown("#### Catégorielle × catégorielle : type × gravité")
    classes = pd.DataFrame(APP["classes_par_type"]).T.reindex(["Régionale", "Majeure", "Bas coût"])[CLASSES_RETARD]
    couleurs = ["#e9e8e2", "#f6c9c4", "#e07b73", ROUGE_RETARD, GRIS_FONCE]
    figure = go.Figure()
    for classe, couleur in zip(CLASSES_RETARD, couleurs):
        figure.add_trace(go.Bar(y=list(classes.index), x=classes[classe], name=classe, orientation="h", marker_color=couleur,
                                text=[f"{fr(v, 0)} %" if v >= 4 else "" for v in classes[classe]], textposition="inside",
                                hovertemplate="%{y} : <b>%{x:.1f} %</b> " + classe + "<extra></extra>"))
    figure.update_layout(barmode="stack")
    figure.update_xaxes(ticksuffix=" %", range=[0, 100])
    figure.update_yaxes(autorange="reversed")
    afficher(figure, 280)
    a_retenir(f"79 % des vols régionaux partent à l'heure, 46 % des vols à bas coût. "
              f"V de Cramér (type × perturbé) = {fr(APP['liens_bivaries']['type × retardé ou annulé']['valeur'], 2)}.")
    st.markdown("#### Catégorielle × numérique : compagnie × minutes de retard")
    q = pd.DataFrame(APP["quantiles_retard_par_compagnie"]).T.sort_values("p90")
    etiquettes = [court(c) for c in q.index]
    figure = go.Figure()
    for etiquette, (c, ligne) in zip(etiquettes, q.iterrows()):
        figure.add_trace(go.Scatter(x=[ligne["mediane"], ligne["p90"]], y=[etiquette] * 2, mode="lines", showlegend=False,
                                    hoverinfo="skip", line=dict(color=COULEURS_TYPES[ligne["type"]], width=4)))
    for nom, cle, symbole in [("médiane", "mediane", "circle"), ("3e quartile", "q3", "circle-open"),
                              ("90e centile", "p90", "diamond")]:
        figure.add_trace(go.Scatter(x=q[cle], y=etiquettes, mode="markers", name=nom,
                                    marker=dict(size=10, symbol=symbole, color=ENCRE),
                                    hovertemplate="%{y} : " + nom + " <b>%{x:.0f} min</b><extra></extra>"))
    figure.update_xaxes(title="Retard au départ (minutes)")
    afficher(figure, 500)
    a_retenir("La médiane vaut 0 presque partout : le vol typique part à l'heure. Les compagnies diffèrent par leurs "
              f"<b>pires retards</b> : 90e centile de {fr(q.loc[ENDEAVOR, 'p90'], 0)} min chez Endeavor, "
              f"{fr(q.loc['JetBlue', 'p90'], 0)} min chez JetBlue.")
    st.markdown("#### Numérique × numérique : retard au départ × retard à l'arrivée")
    afficher(figure_densite_depart_arrivee(), 420)
    st.caption("Couleur = nombre de vols dans la case (échelle logarithmique), sur les 314 844 vols partis : pas de points superposés.")
    a_retenir(f"Les points suivent la diagonale : un vol parti en retard arrive en retard (ρ de Spearman = "
              f"{fr(APP['liens_bivaries']['retard départ × retard arrivée']['valeur'], 2)}, le lien le plus fort du projet). "
              "L'avion rattrape peu en vol : c'est le mécanisme de l'effet domino.")
    notion("Pearson ou Spearman", "Pearson mesure une ligne droite et se laisse tirer par les très gros retards ; Spearman "
           "travaille sur les rangs et résiste aux valeurs extrêmes.")
    st.markdown("#### Bilan : quelle variable est la plus liée au retard ?")
    liens = {k: v for k, v in APP["liens_bivaries"].items()
             if k not in ("retard départ × retard arrivée", "distance × minutes rattrapées")}
    serie = pd.Series({k: abs(v["valeur"]) for k, v in liens.items()}).sort_values()
    couleurs = {"V de Cramér": ACCENT, "rho de Spearman": GRIS_FONCE, "epsilon² (Kruskal-Wallis)": "#eb6834"}
    figure = go.Figure(go.Bar(x=serie.values, y=list(serie.index), orientation="h",
                              marker_color=[couleurs[liens[k]["mesure"]] for k in serie.index],
                              text=[fr(v, 3) for v in serie.values], textposition="outside",
                              customdata=[liens[k]["mesure"] for k in serie.index],
                              hovertemplate="%{y}<br>%{customdata} = <b>%{x:.3f}</b><extra></extra>"))
    figure.update_xaxes(range=[0, serie.max() * 1.3], title="Force du lien (0 = aucun)")
    afficher(figure, 380)
    st.caption("Couleur = mesure utilisée : ■ V de Cramér (bleu) · ■ ρ de Spearman (gris) · ■ ε² de Kruskal-Wallis (orange).")
    a_retenir("L'<b>heure de départ</b> est aussi liée au retard que le <b>type de compagnie</b> : quand on vole compte "
              "autant que avec qui on vole. Tous les liens sont faibles (moins de 0,2).")


def chapitre_bivarie():
    en_tete("Chapitre 5", "Deux variables ensemble : le lien, et sa force",
            "Univarié = une variable seule. Bivarié = deux variables ensemble : on cherche le lien entre elles, et on mesure sa force.")
    compagnies, temps, meteo, domino, trois_cas = st.tabs(["Compagnie × retard, heure × retard", "Le temps, jour par jour",
                                                           "Météo et destinations", "L'effet domino", "Les trois cas et la force des liens"])
    with compagnies:
        onglet_compagnies()
    with temps:
        onglet_temps()
    with meteo:
        onglet_meteo()
    with domino:
        chapitre_domino()
    with trois_cas:
        onglet_trois_cas()


def onglet_meteo():
    if not EQ:
        st.info("Créer la base d'abord : python pipeline.py")
        return
    classes = EQ["meteo_par_classe"]
    grands_chiffres([(f"Pluie : {nom}", f"{fr(v['pct_retard'])} %", f"{v['nb_jours']} jours") for nom, v in classes.items()])
    figure = figure_meteo()
    if figure is not None:
        afficher(figure, 420)
    jours = EQ["jours_non_expliques_meteo"]
    a_retenir(f"Plus il pleut à Atlanta, plus les vols partent en retard (ρ = {fr(EQ['correlations_meteo']['heures_de_pluie'], 2)}). "
              f"Les deux jours que la presse n'expliquait pas sont expliqués par la météo : le {jours[0]['date']}, rafales à "
              f"{fr(jours[0]['rafales_max_kmh'], 0)} km/h ; le {jours[1]['date']}, {fr(jours[1]['heures_de_pluie'], 0)} heures de pluie.")
    notion("fusion", "relier une source externe (Open-Meteo, heure par heure) à nos vols pour expliquer, pas seulement décrire.")
    st.markdown("#### Destination × retard : la carte")
    carte, destinations = carte_destinations()
    if carte is not None:
        afficher(carte, 440)
        st.caption("Couleur = % de vols en retard vers chaque État (le fichier n'a pas les coordonnées des aéroports : "
                   "la carte est faite par État de destination). Étoile = Atlanta.")
        st.markdown("**Les 8 destinations les plus et les moins en retard (au moins 300 vols)**")
        afficher(figure_destinations(destinations), 520)
    D = EQ["destinations"]
    a_retenir(f"Vers New York JFK, {fr(D['pires'][0]['pct_retard'])} % des vols partent en retard ; vers la Floride, environ "
              f"un sur quatre ; vers {D['meilleures'][0]['ville']}, {fr(D['meilleures'][0]['pct_retard'])} %. "
              "<span class='badge badge-piste'>piste</span> non standardisé par compagnie et par heure.")


def onglet_compagnies():
    profil = profil_en_cache().copy()
    colonne_i, colonne_t = st.columns([2, 3])
    indicateur = colonne_i.selectbox("Indicateur", list(INDICATEURS), format_func=INDICATEURS.get)
    types = colonne_t.multiselect("Types de compagnie", list(COULEURS_TYPES), default=list(COULEURS_TYPES))
    profil = profil[profil["type"].isin(types)].sort_values(indicateur)
    if profil.empty:
        st.warning("Choisissez au moins un type de compagnie.")
        return
    est_pct = indicateur.startswith("pct")
    marges = marge_erreur_95(profil[indicateur] / 100, profil["nb_vols"]) if est_pct else None
    figure = go.Figure(go.Bar(
        x=profil[indicateur], y=[court(c) for c in profil.index], orientation="h",
        marker_color=[COULEURS_TYPES[t] for t in profil["type"]],
        error_x=dict(type="data", array=marges, color=ENCRE_SECONDAIRE, thickness=1.2) if est_pct else None,
        customdata=np.c_[profil["type"], profil["nb_vols"]],
        hovertemplate="%{y} (%{customdata[0]})<br><b>%{x:.1f}</b><br>%{customdata[1]:,} vols<extra></extra>",
        showlegend=False))
    # Étiquettes placées après le trait d'erreur (sinon le trait barre le chiffre)
    fin = profil[indicateur] + (marges if est_pct else 0)
    figure.add_trace(go.Scatter(x=fin, y=[court(c) for c in profil.index], mode="text", textposition="middle right",
                                text=["  " + fr(v) + (" %" if est_pct else " min") for v in profil[indicateur]],
                                hoverinfo="skip", showlegend=False))
    figure.update_xaxes(range=[0, fin.max() * 1.15])
    figure.update_xaxes(title=INDICATEURS[indicateur] + (" — trait = marge d'erreur à 95 %" if est_pct else ""))
    afficher(figure, 90 + 34 * len(profil))
    st.caption("■ Majeure (bleu foncé) · ■ Régionale (bleu clair) · ■ Bas coût (orange)")
    a_retenir(f"De <b>{fr(profil[indicateur].iloc[0])}</b> ({court(profil.index[0])}) à "
              f"<b>{fr(profil[indicateur].iloc[-1])}</b> ({court(profil.index[-1])}). Les <b>régionales</b> sont les plus "
              "ponctuelles, les <b>bas coût</b> les moins. Quand les traits de deux compagnies se chevauchent, "
              "on ne peut pas les départager.")
    paires = APP["paires_compagnies"]
    notion("test du classement", f"les {paires['nb_paires']} paires de compagnies testées, avec la correction de Bonferroni : "
           f"{paires['nb_non_distinctes']} paires ne sont pas différentes, surtout au milieu (19 à 21 %). On garde les 14 compagnies, "
           "mais on ne classe pas le milieu au rang près.")

    st.markdown("#### Le retard selon l'heure de départ")
    choix = st.multiselect("Compagnies à comparer (4 au plus)", list(profil_en_cache().index),
                           default=["Delta", "Southwest", "Endeavor (Delta Connection)"], format_func=court,
                           max_selections=4)
    if choix:
        selection = vols[vols["compagnie"].isin(choix)]
        par_heure = tableau_croise_retard(selection, "heure_depart_prevue", "compagnie")
        effectifs = vols_partis(selection).groupby(["heure_depart_prevue", "compagnie"]).size().unstack()
        par_heure = par_heure.where(effectifs >= N_MIN)
        figure = go.Figure()
        for compagnie in choix:
            if compagnie in par_heure:
                serie = par_heure[compagnie].dropna()
                figure.add_trace(go.Scatter(x=serie.index, y=serie.values, name=court(compagnie), mode="lines+markers",
                                            line=dict(color=couleur_compagnie(compagnie), width=2), marker=dict(size=8),
                                            hovertemplate="%{x} h : <b>%{y:.1f} %</b><extra>" + court(compagnie) + "</extra>"))
                figure.add_annotation(x=serie.index[-1], y=serie.values[-1], text=court(compagnie), showarrow=False,
                                      xanchor="left", xshift=8, font=dict(color=ENCRE_SECONDAIRE))
        figure.update_xaxes(title="Heure de départ prévue", ticksuffix=" h")
        figure.update_yaxes(title="% de vols en retard", ticksuffix=" %", rangemode="tozero")
        afficher(figure, 380)
    tranche = pd.Series(R["retard_par_tranche_pct"])
    a_retenir(f"Pour toutes les compagnies, le retard <b>monte au fil de la journée</b> : "
              f"{fr(tranche['Matin (avant 10 h)'])} % le matin, {fr(tranche['Soir (après 18 h)'])} % le soir. "
              "Une compagnie qui vole surtout le soir paraîtra donc moins ponctuelle → chapitre 7.")
    notion("Cleveland & McGill", "comparer des longueurs alignées sur une même base (barres horizontales triées) "
           "est la lecture la plus précise pour l'œil.")


# =========================================================================== 7. Multivarié
def onglet_composition():
    strate = st.radio("Comparer à composition égale selon", ["tranche_horaire", "annee_mois"], horizontal=True,
                                  format_func=lambda s: "l'heure (tranche horaire)" if s == "tranche_horaire" else "le mois")
    cle = "composition_mois" if strate == "annee_mois" else "composition_tranche_horaire"
    compo = pd.DataFrame(R[cle]).T.sort_values("pct_retard_brut")
    etiquettes = [court(c) for c in compo.index]
    fort = (compo["ecart_points"].abs() >= 2) | (compo["rang_brut"] - compo["rang_standardise"]).abs().ge(3)
    figure = haltere(etiquettes, compo["pct_retard_brut"], compo["taux_standardise_pct"])
    figure.add_trace(go.Scatter(x=compo["pct_retard_brut"], y=etiquettes, mode="markers", name="Brut (observé)",
                                marker=dict(color=GRIS_FONCE, size=11, line=dict(color="white", width=2)),
                                hovertemplate="%{y} : brut <b>%{x:.1f} %</b><extra></extra>"))
    figure.add_trace(go.Scatter(x=compo["taux_standardise_pct"], y=etiquettes, mode="markers",
                                name="Standardisé (même composition que l'ensemble des vols)",
                                marker=dict(color=[ACCENT if f else ACCENT_CLAIR for f in fort], size=11,
                                            line=dict(color="white", width=2)),
                                customdata=compo[["ecart_points", "rang_brut", "rang_standardise"]].to_numpy(),
                                hovertemplate="%{y} : standardisé <b>%{x:.1f} %</b><br>écart %{customdata[0]:+.1f} pts"
                                              "<br>rang %{customdata[1]:.0f} → %{customdata[2]:.0f}<extra></extra>"))
    figure.update_xaxes(title="% de vols en retard", ticksuffix=" %")
    afficher(figure, 520)
    deplacement = (compo["rang_brut"] - compo["rang_standardise"]).abs()
    cible = compo["ecart_points"].abs().idxmax() if strate != "annee_mois" else deplacement.idxmax()
    ligne = compo.loc[cible]
    a_retenir(f"Le plus grand changement : <b>{court(cible)}</b> passe de {fr(ligne['pct_retard_brut'])} % à "
              f"{fr(ligne['taux_standardise_pct'])} % (rang {ligne['rang_brut']:.0f} → {ligne['rang_standardise']:.0f}, "
              "1 = le plus en retard). Une partie de l'écart entre compagnies vient de <b>quand</b> elles volent.")
    notion("standardisation directe", "x̄ = Σ w<sub>h</sub> · μ<sub>h</sub> : on applique le taux de chaque compagnie "
           "dans chaque strate (μ<sub>h</sub>) à la répartition commune des vols (w<sub>h</sub>).")


def onglet_saisons():
    saisons = pd.DataFrame(R["retard_par_saison_pct"]).T.reindex(columns=ORDRE_SAISONS)
    saisons = saisons.loc[saisons.mean(axis=1).sort_values().index]
    rangs = pd.DataFrame(R["rang_par_saison"]).T.reindex(index=saisons.index, columns=ORDRE_SAISONS)
    figure = go.Figure(go.Heatmap(z=saisons.values, x=ORDRE_SAISONS, y=[court(c) for c in saisons.index],
                                  colorscale=ECHELLE_RETARD, xgap=2, ygap=2,
                                  colorbar=dict(title="% retard", ticksuffix=" %"),
                                  text=[[f"{v:.0f} %" for v in ligne] for ligne in saisons.values], texttemplate="%{text}",
                                  customdata=rangs.values,
                                  hovertemplate="%{y}, %{x} : <b>%{z:.1f} %</b><br>rang %{customdata:.0f} / 14<extra></extra>"))
    figure.update_xaxes(side="top")
    afficher(figure, 500)
    a_retenir("L'été est la pire saison pour presque toutes les compagnies. Les extrêmes (Endeavor la plus ponctuelle, "
              "JetBlue la moins) gardent leur place ; <b>le milieu du classement change</b> d'une saison à l'autre. "
              "<span class='badge badge-piste'>piste</span> une seule année observée.")


def onglet_simpson():
    globale, par_compagnie = correlation_globale_et_par_groupe(vols, "TaxiOut", "DepDelayMinutes")
    par_compagnie = par_compagnie[par_compagnie["n"] >= N_MIN].sort_values("correlation")
    figure = go.Figure(go.Scatter(
        x=par_compagnie["correlation"], y=[court(c) for c in par_compagnie.index], mode="markers",
        marker=dict(size=12, color=[ACCENT if r > 0 else GRIS_FONCE for r in par_compagnie["correlation"]],
                    line=dict(color="white", width=2)),
        customdata=par_compagnie["n"].to_numpy(),
        hovertemplate="%{y} : ρ = <b>%{x:.3f}</b><br>%{customdata:,.0f} vols<extra></extra>"))
    figure.add_vline(x=globale, line_dash="dash", line_color=ROUGE_RETARD,
                     annotation_text=f"toutes compagnies mélangées : ρ = {fr(globale, 3)}", annotation_position="top")
    figure.add_vline(x=0, line_color=ENCRE_SECONDAIRE, line_width=1)
    figure.update_xaxes(title="Corrélation de Spearman : roulage au départ ↔ retard au départ")
    afficher(figure, 460)
    positives = int((par_compagnie["correlation"] > 0).sum())
    a_retenir(f"Mélangées, les compagnies donnent une corrélation ≈ 0 ({fr(globale, 3)}). Séparées, elle est "
              f"<b>positive dans {positives} compagnies sur {len(par_compagnie)}</b>. Le mélange masque la relation : "
              "c'est un <b>effet de type Simpson</b>. Il faut analyser par groupe avant de conclure.")
    notion("Simpson", "une relation vraie dans chaque groupe peut disparaître, ou s'inverser, quand on mélange les groupes.")


def onglet_versions():
    versions = pd.DataFrame(APP["sensibilite_types"]).T
    tableau = versions[["Bas coût", "Majeure", "Régionale"]].map(lambda v: f"{fr(v)} %")
    tableau["Ordre conservé ?"] = versions["ordre_conserve"].map({True: "oui", False: "non"})
    st.dataframe(tableau, width="stretch")
    c = versions.loc["C : sans la plus grosse compagnie de chaque type"]
    a_retenir(f"L'ordre <b>bas coût > majeures > régionales</b> tient dans les trois versions. Mais sans Endeavor, les "
              f"régionales passent à {fr(c['Régionale'])} %, proches des majeures ({fr(c['Majeure'])} %) : "
              "<b>l'avantage des régionales est surtout celui d'Endeavor</b>.")
    notion("multi-versions", "garder plusieurs traitements quand un choix (ici : qui est « bas coût ») peut changer la conclusion.")


def onglet_crises():
    afficher(figure_annulations_semaine(), 460)
    a_retenir("Chaque case = une compagnie pendant une semaine ; plus c'est rouge, plus elle a annulé. Chaque compagnie a "
              "<b>ses propres crises</b> : Spirit début août, Southwest en octobre, JetBlue fin janvier ; presque toutes à Noël.")
    notion("trois variables sur une carte", "compagnie (lignes) × semaine (colonnes) × % annulés (couleur).")
    st.markdown("#### Zoom sur une compagnie, jour par jour")
    annulations = annulations_journalieres_par_compagnie()
    choix = st.selectbox("Compagnie", list(profil_en_cache().index), format_func=court,
                         index=list(profil_en_cache().index).index("Spirit"))
    serie = annulations[choix].dropna()
    figure = go.Figure(go.Bar(x=serie.index, y=serie.values, marker_color=[ROUGE_RETARD if v > 10 else GRIS_CONTEXTE
                                                                          for v in serie.values],
                              hovertemplate="%{x|%d/%m/%Y} : <b>%{y:.0f} %</b> des vols annulés<extra></extra>"))
    zones_evenements(figure, choix)
    figure.update_xaxes(tickformat="%m/%Y")
    figure.update_yaxes(ticksuffix=" %", range=[0, 100], title=f"% des vols {court(choix)} annulés ce jour-là")
    afficher(figure, 360)
    lies = [ev for ev in APP["evenements"] if ev["compagnies"] == "toutes" or court(choix) in ev["compagnies"]
            or choix in ev["compagnies"]]
    if lies:
        for ev in lies:
            st.markdown(f"- **{ev['nom']}** ({ev['debut']}) : {ev['explication']} [source]({ev['source']})")
    a_retenir("Chaque compagnie a ses propres crises : Spirit début août 2021, Southwest en octobre, American à Halloween "
              "et début février, JetBlue fin janvier. Beaucoup naissent <b>loin d'Atlanta</b> (blizzard à New York, "
              "verglas à Dallas, contrôle aérien en Floride).")


def chapitre_multivarie():
    en_tete("Chapitre 6", "Plusieurs variables : les pièges qui changent la conclusion",
            "Une troisième variable (l'heure, le mois, la compagnie) peut modifier ou cacher la relation qu'on croit voir.")
    profil, acp, bulles, jour_heure, composition, saisons, simpson, versions, crises, modele = st.tabs(
        ["Profil des compagnies", "ACP", "Bulles (4 dimensions)", "Jour × heure", "Effet de composition", "Saisons",
         "Effet de type Simpson", "Plusieurs versions", "Les crises de chaque compagnie", "Le modèle : tout ensemble"])
    with acp:
        onglet_acp()
    with bulles:
        onglet_bulles()
    with jour_heure:
        onglet_jour_heure()
    with profil:
        afficher(figure_profil_compagnies(), 520)
        st.caption("Couleur = écart à la moyenne des compagnies (rouge = moins bien, bleu = mieux) ; le chiffre est la vraie valeur.")
        a_retenir("Chaque compagnie sur plusieurs indicateurs à la fois : les bas coût (en haut) sont rouges presque partout "
                  "(fréquence, gravité, annulations) ; Endeavor (en bas) est bleue. On ne juge pas une compagnie sur un seul chiffre.")
        notion("score z", "la couleur = (valeur − moyenne des compagnies) ÷ écart-type : on compare des indicateurs d'unités différentes.")
    with composition:
        onglet_composition()
    with saisons:
        onglet_saisons()
    with simpson:
        onglet_simpson()
    with versions:
        onglet_versions()
    with crises:
        onglet_crises()
    with modele:
        onglet_modele()


def onglet_modele():
    if not EQ:
        st.info("Créer la base d'abord : python pipeline.py")
        return
    M = EQ["modele"]
    RC = M["rapports_de_cotes"]
    grands_chiffres([
        ("Partir le soir", f"× {fr(RC['Heure : Soir']['rapport_de_cotes'], 2)}", "cote du retard, à conditions égales"),
        ("Avion déjà en retard", f"× {fr(RC['Vol précédent : vol précédent en retard']['rapport_de_cotes'], 2)}", None),
        ("JetBlue (vs Delta)", f"× {fr(RC['Compagnie : JetBlue']['rapport_de_cotes'], 2)}", None),
        ("Endeavor (vs Delta)", f"× {fr(RC['Compagnie : Endeavor (Delta Connection)']['rapport_de_cotes'], 2)}", None),
    ])
    figure, nb = figure_modele_rapports()
    afficher(figure, 120 + 24 * nb)
    st.caption("Point = rapport de cotes ; trait = intervalle de confiance à 95 % ; gris = pas différent de la référence "
               f"(Delta, matin, automne, mardi, avion à l'heure, temps sec). AUC sur des vols jamais vus : "
               f"{fr(EQ['modele'].get('auc_test', EQ['modele']['auc']), 3)}.")
    afficher(figure_importance(), 300)
    st.markdown("#### Le meilleur et le pire vol possibles")
    for colonne, (scenario, valeur) in zip(st.columns(3), M["scenarios"].items()):
        colonne.metric(scenario, f"{fr(valeur)} %")
    a_retenir("À conditions égales, <b>l'heure de départ pèse plus que la compagnie</b>. Même aéroport, même année : "
              "de 1,6 % à 81,5 % de risque selon les conditions.")
    notion("régression logistique", f"chaque facteur est mesuré « toutes choses égales par ailleurs ». Rapport de cotes > 1 = "
           f"plus de retard que la référence. AUC = {fr(M['auc'], 2)} : le modèle n'explique qu'une partie du retard "
           f"(n = {fr(M['n'], 0)} vols partis).")


# =========================================================================== 8. Domino
def chapitre_domino():
    st.markdown("Pour chaque départ, on regarde le départ <b>précédent</b> du même avion, le même jour. Était-il déjà en retard ?",
                unsafe_allow_html=True)
    avant, apres = IND["domino_pct_si_precedent_a_l_heure"], IND["domino_pct_si_precedent_en_retard"]
    grands_chiffres([
        ("Vol précédent à l'heure", f"{fr(avant)} %", "% de vols partis en retard."),
        ("Vol précédent en retard", f"{fr(apres)} %", "% de vols partis en retard."),
        ("Écart", f"+ {fr(apres - avant)} pts", None),
    ])
    domino = pd.DataFrame(R["domino_par_compagnie"]).T.sort_values("ecart_points")
    etiquettes = [f"{court(c)} ({fr(n, 0)} cas)" for c, n in domino["n_vol_precedent_en_retard"].items()]
    figure = haltere(etiquettes, domino["pct_si_precedent_a_l_heure"], domino["pct_si_precedent_en_retard"])
    figure.add_trace(go.Scatter(x=domino["pct_si_precedent_a_l_heure"], y=etiquettes, mode="markers",
                                name="Vol précédent à l'heure",
                                marker=dict(color=GRIS_FONCE, size=12, line=dict(color="white", width=2)),
                                hovertemplate="%{y}<br>précédent à l'heure : <b>%{x:.1f} %</b><extra></extra>"))
    figure.add_trace(go.Scatter(x=domino["pct_si_precedent_en_retard"], y=etiquettes, mode="markers",
                                name="Vol précédent en retard",
                                marker=dict(color=ROUGE_RETARD, size=12, line=dict(color="white", width=2)),
                                customdata=domino["ecart_points"].to_numpy(),
                                hovertemplate="%{y}<br>précédent en retard : <b>%{x:.1f} %</b>"
                                              "<br>écart +%{customdata:.1f} pts<extra></extra>"))
    figure.update_xaxes(title="% de vols partis en retard", ticksuffix=" %",
                        range=[0, domino["pct_si_precedent_en_retard"].max() * 1.08])
    st.markdown("#### Dans chaque compagnie, le retard précédent fait grimper le risque")
    afficher(figure, 140 + 42 * len(domino))
    a_retenir(f"Quand l'avion arrive d'un départ en retard, le risque passe de <b>{fr(avant)} %</b> à "
              f"<b>{fr(apres)} %</b>, et l'écart existe dans <b>toutes</b> les compagnies mesurables. "
              "<span class='badge badge-solide'>solide</span> mais c'est une association, pas une preuve de cause.")
    st.markdown("#### L'heure n'explique pas tout")
    avec_precedent = partis_tous.dropna(subset=["vol_precedent_en_retard"])
    par_heure = tableau_croise_retard(avec_precedent, "heure_depart_prevue", "vol_precedent_en_retard")
    effectifs = avec_precedent.groupby(["heure_depart_prevue", "vol_precedent_en_retard"]).size().unstack()
    par_heure = par_heure.where(effectifs >= N_MIN)
    figure = go.Figure()
    for valeur, nom, couleur in [(1.0, "Vol précédent en retard", ROUGE_RETARD), (0.0, "Vol précédent à l'heure", GRIS_FONCE)]:
        serie = par_heure[valeur].dropna()
        figure.add_trace(go.Scatter(x=serie.index, y=serie.values, name=nom, mode="lines+markers", marker=dict(size=8),
                                    line=dict(color=couleur, width=2),
                                    hovertemplate="%{x} h : <b>%{y:.1f} %</b><extra>" + nom + "</extra>"))
    figure.update_xaxes(title="Heure de départ prévue", ticksuffix=" h")
    figure.update_yaxes(title="% de vols en retard", ticksuffix=" %", rangemode="tozero")
    afficher(figure, 340)
    a_retenir("À <b>chaque heure</b> de la journée, la courbe rouge reste au-dessus de la grise : "
              "l'effet domino n'est pas seulement un effet « soir ».")
    DR = APP["domino_rang"]
    rangs = DR["pct_retard_par_rang"]
    notion("une objection, et sa réponse", f"le 3e départ d'Atlanta d'un avion n'est pas plus en retard que le 1er "
           f"(1er : {fr(rangs['1'])} % ; 3e : {fr(rangs['3'])} %). Pourquoi ? Nous ne voyons que les départs d'Atlanta : "
           f"{fr(DR['pct_premiers_departs_apres_14h'])} % des « 1ers départs » ont lieu après 14 h, l'avion a déjà volé ailleurs. "
           "Le rang ne mesure pas la journée de l'avion ; le domino, lui, se vérifie à chaque heure.")


# =========================================================================== 9. Conclusion
def chapitre_conclusion():
    C = R["compagnies"]
    meilleure = min(C, key=lambda c: C[c]["pct_retard_ou_annule"])
    pire = max(C, key=lambda c: C[c]["pct_retard_ou_annule"])
    T = R["taille_echantillon"]
    en_tete("Chapitre 9", "Ce que l'on peut affirmer",
            "Pour chaque résultat : sa valeur, et sa solidité (<span class='badge badge-solide'>solide</span> "
            "résiste à tout ; <span class='badge badge-piste'>piste</span> à confirmer).")
    resultats = [
        ("Les compagnies diffèrent nettement", "solide",
         f"De {fr(C[meilleure]['pct_retard_ou_annule'])} % ({court(meilleure)}) à {fr(C[pire]['pct_retard_ou_annule'])} % "
         f"({court(pire)}) de vols retardés ou annulés. Régionales devant, bas coût derrière."),
        ("Une partie de l'écart vient de l'horaire", "solide",
         "À heure égale, American passe de 17,4 % à 20,9 % ; à mois égal, SkyWest recule du 5ᵉ au 11ᵉ rang."),
        ("Le milieu du classement change avec la saison", "piste",
         "Les extrêmes restent ; le milieu bouge. Une seule année : à confirmer sur d'autres années."),
        ("Le retard se propage", "solide",
         f"{fr(IND['domino_pct_si_precedent_a_l_heure'])} % → {fr(IND['domino_pct_si_precedent_en_retard'])} % selon le "
         "départ précédent du même avion, dans toutes les compagnies."),
        ("Le bon échantillon dépend de la question", "solide",
         f"5 000 vols suffisent pour un chiffre global (± {fr(T['marge_n_retenu_points'], 2)} pt). Pour comparer les "
         "compagnies : stratifier par compagnie. Les grappes de jours sont les moins fiables."),
        ("Associations, pas causes", "piste",
         "Équipages, contrôle aérien : les causes officielles ne sont pas dans le fichier ; la météo (fusion) en explique une partie."),
    ]
    for i in range(0, len(resultats), 2):
        for colonne, (titre, niveau, texte) in zip(st.columns(2), resultats[i:i + 2]):
            colonne.markdown(f'<div class="carte"><h4>{titre}<span class="badge badge-{niveau}">{niveau}</span></h4>'
                             f'<p>{texte}</p></div>', unsafe_allow_html=True)
        st.write("")
    a_retenir("<b>Voir ≠ comprendre ≠ expliquer.</b> Les graphiques montrent des associations solides ; "
              "pour expliquer, il faut d'autres données : la météo, ajoutée par fusion, en explique une partie.")
    st.markdown("#### Ce que l'approfondissement a ajouté")
    PA = APP["paires_compagnies"]
    ajouts = [
        ("Le retard arrive par crises", "solide",
         "Vu jour par jour : 10 crises réelles (Spirit, Southwest, Omicron, tempêtes...), visibles dans nos données ET documentées."),
        ("L'avantage des régionales, c'est surtout Endeavor", "solide",
         "Sans Endeavor, les régionales sont proches des majeures. L'ordre bas coût > majeures > régionales tient quand même."),
        ("Le milieu du classement n'est pas départageable", "piste",
         f"{PA['nb_non_distinctes']} paires de compagnies sur {PA['nb_paires']} ne sont pas différentes (test avec correction de Bonferroni)."),
        ("Quand on vole compte autant que avec qui", "solide",
         "L'heure de départ est aussi liée au retard que le type de compagnie (V ≈ 0,14)."),
        ("À conditions égales, l'heure pèse plus que la compagnie", "solide",
         "Le modèle (tous les facteurs ensemble) : le soir et un avion déjà en retard multiplient la cote du retard par 3."),
        ("La pluie et le vent comptent", "solide",
         "Fusion avec la météo : 14,1 % de retard un jour sec, 24,5 % un jour de forte pluie."),
    ]
    for i in range(0, len(ajouts), 2):
        for colonne, (titre, niveau, texte) in zip(st.columns(2), ajouts[i:i + 2]):
            colonne.markdown(f'<div class="carte"><h4>{titre}<span class="badge badge-{niveau}">{niveau}</span></h4>'
                             f'<p>{texte}</p></div>', unsafe_allow_html=True)
        st.write("")
    st.markdown("#### La réponse à la problématique")
    reponse_en_bref()
    CC = R.get("conclusions_compagnies")
    if CC:
        st.markdown("#### La conclusion pour chaque compagnie")
        st.dataframe(pd.DataFrame([(v["rang"], court(c), v["type"], f"{fr(v['pct_retard_ou_annule'])} %", v["position"])
                                   for c, v in CC.items()], columns=["Rang", "Compagnie", "Type", "Retardés ou annulés", "Position"]),
                     hide_index=True, width="stretch")
        choix = st.selectbox("Voir la fiche d'une compagnie", list(CC), format_func=court)
        fiche = CC[choix]
        st.markdown(f'<div class="carte"><h4>{fiche["conclusion"]}</h4><p>' + "<br>".join("• " + x for x in fiche["constats"])
                    + (f'<br><i>Dans le monde réel : {fiche["contexte"]}</i>' if fiche["contexte"] else "") + "</p></div>",
                    unsafe_allow_html=True)
        st.write("")
        a_retenir("En tête : les régionales, surtout Endeavor. Au milieu : les écarts sont dans la marge d'erreur. En queue : les "
                  "bas coût, et l'écart tient même à conditions égales.")
    recommandations()
    st.markdown("#### Journal des décisions : ce que nous avons remis en question")
    st.dataframe(pd.DataFrame([
        ("Été 2022 → 12 mois", "l'été surestime le retard (26 % contre 19 %)", "comparer sur toutes les saisons"),
        ("Courbe par mois → par jour", "revue critique", "les crises apparaissent, reliées au monde réel"),
        ("Classement au rang près", "test des 91 paires", "le milieu est déclaré non départageable"),
        ("« Les régionales sont les plus ponctuelles »", "version sans Endeavor", "nuancé : c'est surtout Endeavor"),
        ("« American meilleure que Delta »", "annulations, heure égale", "classé artefact"),
        ("Le domino contredit par le rang ?", "rang 1 ≈ rang 3", "limite des données ; domino vérifié à chaque heure"),
        ("Fichiers → base SQL avec vues", "consigne : fusion et tables virtuelles", "5 tables fusionnées, 11 vues testées"),
        ("Presse seule → presse + météo", "deux jours sans source", "expliqués par la pluie et le vent"),
        ("Analyses une à une → modèle", "les facteurs se mélangent", "l'heure pèse plus que la compagnie"),
        ("Nom « Comair »", "recherche sur le code OH", "nom périmé (PSA Airlines) signalé"),
    ], columns=["Décision", "Déclencheur", "Ce qui a changé"]), hide_index=True, width="stretch")
    st.markdown("#### Limites")
    st.markdown(f"""
- **Observationnel** : des associations, pas des causes.
- **Périmètre** : Atlanta, départs seulement ; Delta = {fr(POP['part_delta_pct'])} % des vols.
- **Une seule année** (août 2021 – juillet 2022), période de reprise après le COVID.
- **Compagnie opérante ou commerciale** : Endeavor vole pour Delta ; Republic et SkyWest pour plusieurs marques.
- **Contexte réel** : les causes des crises viennent de sources extérieures ; deux jours expliqués par la météo seulement.
- **Domino** : vu depuis Atlanta seulement ; les autres vols de l'avion ne sont pas dans le fichier.
""")



# =========================================================================== Nouvelles pages (version 3)
TYPES_ORDRE = ["Régionale", "Majeure", "Bas coût"]
NOMS_DIMENSIONS = {"compagnie": "Compagnie", "type_compagnie": "Type de compagnie", "saison": "Saison",
                   "tranche_horaire": "Tranche horaire", "jour_semaine": "Jour de la semaine", "Dest": "Destination"}
ORDRES_DIMENSIONS = {"saison": ORDRE_SAISONS, "tranche_horaire": NOMS_TRANCHES, "jour_semaine": ORDRE_JOURS,
                     "type_compagnie": TYPES_ORDRE}


def hero(etiquette, titre, sous_titre):
    st.markdown(f'<div class="hero"><div class="hero-etiquette">{etiquette}</div><div class="hero-titre">{titre}</div>'
                f'<div class="hero-sous">{sous_titre}</div></div>', unsafe_allow_html=True)


def cartes_kpi(cartes):
    """Cartes d'indicateurs : [(icône, libellé, valeur, détail, couleur), ...]."""
    for colonne, (icone, libelle, valeur, detail, couleur) in zip(st.columns(len(cartes)), cartes):
        colonne.markdown(f'<div class="kpi" style="border-top:4px solid {couleur}"><div class="kpi-icone">{icone}</div>'
                         f'<div class="kpi-libelle">{libelle}</div><div class="kpi-valeur" style="color:{couleur}">{valeur}</div>'
                         f'<div class="kpi-detail">{detail}</div></div>', unsafe_allow_html=True)
    st.write("")


def section(titre, sous_titre=""):
    st.markdown(f'<div class="section"><span>{titre}</span>' + (f'<small>{sous_titre}</small>' if sous_titre else "")
                + "</div>", unsafe_allow_html=True)


@st.cache_data
def calendrier_en_cache():
    jour = serie_journaliere_en_cache().copy()
    jour["date"] = jour.index
    jour["jour_semaine"] = jour["date"].dt.dayofweek
    jour["semaine"] = (jour["date"] - pd.to_timedelta(jour["jour_semaine"], unit="D")).dt.normalize()
    return jour


def figure_calendrier(mesure="pct_retard"):
    jour = calendrier_en_cache()
    grille = jour.pivot_table(index="jour_semaine", columns="semaine", values=mesure)
    dates = jour.pivot_table(index="jour_semaine", columns="semaine", values="date", aggfunc="first")
    texte = dates.map(lambda d: d.strftime("%d/%m/%Y") if pd.notna(d) else "")
    echelle = ECHELLE_RETARD if mesure == "pct_retard" else [[0, "#f4f4f2"], [1, GRIS_FONCE]]
    figure = go.Figure(go.Heatmap(z=grille.values, x=list(grille.columns), y=ORDRE_JOURS, colorscale=echelle, xgap=2, ygap=2,
                                  customdata=texte.values, zmin=0, zmax=60 if mesure == "pct_retard" else 35,
                                  colorbar=dict(title="%", ticksuffix=" %"),
                                  hovertemplate="%{customdata} : <b>%{z:.1f} %</b><extra></extra>"))
    figure.update_yaxes(autorange="reversed")
    mois = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
    debuts = [c for i, c in enumerate(grille.columns) if i == 0 or c.month != grille.columns[i - 1].month]
    figure.update_xaxes(tickvals=debuts, ticktext=[f"{mois[d.month - 1]} {d.strftime('%y')}" for d in debuts], showgrid=False)
    return figure


def page_accueil():
    J = APP["journalier"]
    C = R["compagnies"]
    hero("Analyse de données · Salma Bouhmid",
         "Qu'est-ce qui distingue les compagnies aériennes au départ d'Atlanta ?",
         f"{fr(POP['nb_vols'], 0)} vols · 14 compagnies · 365 jours (août 2021 – juillet 2022) · BTS via Kaggle · "
         "base SQL, modèle, échantillonnage")
    cartes_kpi([
        ("⏱️", "Vols partis en retard", f"{fr(IND['pct_retard'])} %", "départ ≥ 15 min après l'heure prévue", ACCENT),
        ("❌", "Vols annulés", f"{fr(IND['pct_annules'])} %", f"{fr(R['qualite']['nb_annules'], 0)} vols", GRIS_FONCE),
        ("🐢", "Retard moyen des vols en retard", f"{fr(IND['retard_moyen_des_vols_en_retard_min'], 0)} min", "quand il y a retard", ORANGE),
    ])
    cartes_kpi([
        ("🎯", "Effet domino", f"{fr(IND['domino_pct_si_precedent_a_l_heure'], 0)} → {fr(IND['domino_pct_si_precedent_en_retard'], 0)} %",
         "si l'avion arrive déjà en retard", ROUGE_RETARD),
        ("🥇", "La plus ponctuelle", "Endeavor", f"{fr(C[ENDEAVOR]['pct_retard_ou_annule'])} % retardés ou annulés", "#1d8a5a"),
        ("⛈️", "Le pire jour", f"{fr(J['jour_max']['pct'])} %", f"le {J['jour_max']['date']} (orages)", ROUGE_RETARD),
    ])
    reponse_en_bref()
    section("Le calendrier de l'année", "une case = un jour ; plus c'est rouge, plus il y a de retard")
    mesure = st.radio("Mesure", ["pct_retard", "pct_annules"], horizontal=True, key="mesure_calendrier",
                      format_func=lambda m: "% de vols en retard" if m == "pct_retard" else "% de vols annulés")
    afficher(figure_calendrier(mesure), 300)
    a_retenir("Les cases rouges ne sont pas réparties au hasard : elles forment des <b>paquets</b> (Noël, mi-janvier, mars, été). "
              "Le retard arrive par <b>crises</b> de quelques jours, chacune reliée à un événement réel (page 5).")
    notion("calendrier (heatmap)", "deux dimensions du temps à la fois : la semaine (en colonnes) et le jour de la semaine (en lignes).")
    section("Le classement des 14 compagnies", "retardés ou annulés, avec la moyenne de l'aéroport")
    figure_classement()
    section("Ce que contient ce dashboard")
    resumes = ["Le fichier, l'entrepôt de données (schéma en étoile) et le cube", "Ce qui manque, et pourquoi on garde les annulés",
               "Les chiffres et la forme de chaque variable", "Peut-on se contenter de 5 000 vols ?",
               "Deux variables ensemble, et la force du lien", "Plusieurs variables : les pièges, le modèle",
               "La fiche de chaque compagnie", "Prédire le risque de retard d'un vol", "Les réponses et les recommandations"]
    for ligne in range(3):
        for i, colonne in enumerate(st.columns(3)):
            k = ligne * 3 + i
            colonne.markdown(f'<div class="carte"><h4>{CHAPITRES[k + 1]}</h4><p>{resumes[k]}</p></div>', unsafe_allow_html=True)
        st.write("")


def figure_classement(types=None):
    profil = profil_en_cache().copy()
    if types:
        profil = profil[profil["type"].isin(types)]
    profil = profil.sort_values("pct_retard_ou_annule")
    moyenne = IND["pct_retard_ou_annule"]
    figure = go.Figure(go.Bar(x=profil["pct_retard_ou_annule"], y=[court(c) for c in profil.index], orientation="h",
                              marker_color=[COULEURS_TYPES[t] for t in profil["type"]],
                              text=[f"{fr(v)} %" for v in profil["pct_retard_ou_annule"]], textposition="outside",
                              customdata=profil[["type", "nb_vols"]].to_numpy(),
                              hovertemplate="%{y} (%{customdata[0]})<br><b>%{x:.1f} %</b><br>%{customdata[1]:,} vols<extra></extra>"))
    figure.add_vline(x=moyenne, line_dash="dash", line_color=ENCRE_SECONDAIRE,
                     annotation_text=f"moyenne {fr(moyenne)} %", annotation_position="top")
    figure.update_xaxes(ticksuffix=" %", range=[0, profil["pct_retard_ou_annule"].max() * 1.15])
    afficher(figure, 90 + 30 * len(profil))
    st.caption("■ Régionale (bleu clair) · ■ Majeure (bleu foncé) · ■ Bas coût (orange)")
    a_retenir("Toutes les compagnies <b>à bas coût</b> sont au-dessus de la moyenne ; toutes les <b>régionales</b> sauf SkyWest sont en dessous.")


# --------------------------------------------------------------------------- Entrepôt de données (multidimensionnel)
def entrepot_de_donnees():
    section("L'entrepôt de données : un modèle en étoile", "la table des faits (les vols) au centre, les dimensions autour")
    st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#c9d6e8", fillcolor="#f1f5fb"];
  vols [label="VOLS (faits)\\n319 335 lignes\\nretard, annulé, roulage…", fillcolor="#2a78d6", fontcolor="white", color="#2a78d6"];
  compagnies [label="COMPAGNIES\\ntype, marque, base"];
  destinations [label="DESTINATIONS\\nville, État"];
  calendrier [label="CALENDRIER\\njour, saison, férié"];
  evenements [label="ÉVÉNEMENTS\\n10 crises sourcées"];
  meteo_jour [label="MÉTÉO DU JOUR\\npluie, neige, vent"];
  meteo_heure [label="MÉTÉO HORAIRE\\nà l'heure du départ"];
  compagnies -> vols [label=" compagnie"]; destinations -> vols [label=" Dest"];
  calendrier -> vols [label=" date"]; evenements -> vols [label=" date"];
  meteo_jour -> vols [label=" date"]; meteo_heure -> vols [label=" date + heure"];
}""")
    st.dataframe(pd.DataFrame([
        ("Bronze (brut)", "data/raw/…parquet", "les vols tels que publiés par le BTS, filtrés sur Atlanta"),
        ("Argent (nettoyé)", "data/processed/…_propre.parquet", "9 colonnes retirées, 15 créées, annulés gardés"),
        ("Or (analytique)", "data/atlanta_vols.sqlite", "5 dimensions + 11 vues SQL stockées (tables mv_*), prêtes pour l'analyse"),
    ], columns=["Couche", "Où", "Contenu"]), hide_index=True, width="stretch")
    notion("entrepôt de données et schéma en étoile", "une table de <b>faits</b> (ce qu'on mesure : chaque vol) reliée à des "
           "tables de <b>dimensions</b> (les axes d'analyse : qui, où, quand, quel temps). On peut alors « couper » les données "
           "selon n'importe quelle combinaison de dimensions : c'est l'analyse <b>multidimensionnelle</b> (OLAP).")
    section("Le cube : choisissez deux dimensions", "chaque case = le croisement des deux dimensions")
    c1, c2, c3 = st.columns(3)
    lignes = c1.selectbox("En lignes", list(NOMS_DIMENSIONS), format_func=NOMS_DIMENSIONS.get, index=0, key="cube_lignes")
    colonnes = c2.selectbox("En colonnes", [d for d in NOMS_DIMENSIONS if d != lignes], format_func=NOMS_DIMENSIONS.get,
                            index=1, key="cube_colonnes")
    mesure = c3.selectbox("Mesure", ["pct_retard", "pct_annules", "pct_retard_ou_annule", "nb_vols"], key="cube_mesure",
                          format_func={"pct_retard": "% en retard", "pct_annules": "% annulés",
                                       "pct_retard_ou_annule": "% retardés ou annulés", "nb_vols": "Nombre de vols"}.get)
    figure, nb = figure_cube(lignes, colonnes, mesure)
    afficher(figure, max(320, 26 * nb + 120))
    a_retenir("Le même entrepôt répond à des dizaines de questions sans nouveau calcul : il suffit de choisir les dimensions. "
              "Les cases grises ont moins de 30 vols (trop peu pour un pourcentage fiable).")
    with st.expander("Les 11 vues SQL et un exemple (l'effet domino)"):
        st.dataframe(pd.DataFrame([
            ("v_ponctualite_compagnie", "Q1"), ("v_ponctualite_type", "Q1"), ("v_composition_heure", "Q2 (standardisation)"),
            ("v_compagnie_saison", "Q3"), ("v_domino / v_domino_compagnie", "Q4 (fonction LAG)"),
            ("v_jour", "temps × météo × crises"), ("v_jour_semaine", "jour de la semaine"), ("v_destination", "destinations"),
            ("v_creneau", "compagnie × heure × saison"), ("v_depart_meteo", "modèle"),
        ], columns=["Vue SQL (stockée en table mv_*)", "Question"]), hide_index=True, width="stretch")
        sql = (Path(__file__).resolve().parent.parent / "sql" / "02_vues.sql").read_text(encoding="utf-8")
        debut = sql.find("CREATE VIEW v_domino AS")
        st.code(sql[debut:sql.find(";", debut) + 1], language="sql")


@st.cache_data
def figure_cube(lignes, colonnes, mesure):
    if FICHIER_BASE.exists():   # requête SQL sur l'entrepôt (GROUP BY sur deux dimensions)
        table = requete(f"""SELECT {lignes} AS l, {colonnes} AS c, COUNT(*) AS nb_vols,
                                   100.0 * AVG(CASE WHEN Cancelled = 0 THEN DepDel15 END) AS pct_retard,
                                   100.0 * AVG(Cancelled) AS pct_annules, 100.0 * AVG(retard_ou_annule) AS pct_retard_ou_annule
                            FROM vols GROUP BY {lignes}, {colonnes}""").set_index(["l", "c"])
    else:
        v = vols.copy()
        v["retard_parti"] = v["DepDel15"].where(~v["Cancelled"])
        g = v.groupby([lignes, colonnes], observed=True)
        table = pd.DataFrame({"nb_vols": g.size(), "pct_retard": g["retard_parti"].mean() * 100,
                              "pct_annules": g["Cancelled"].mean() * 100, "pct_retard_ou_annule": g["retard_ou_annule"].mean() * 100})
    grille = table[mesure].unstack()
    effectifs = table["nb_vols"].unstack()
    if mesure != "nb_vols":
        grille = grille.where(effectifs >= 30)
    for dimension, axe in [(lignes, 0), (colonnes, 1)]:
        ordre = ORDRES_DIMENSIONS.get(dimension)
        if ordre:
            grille = grille.reindex([o for o in ordre if o in (grille.index if axe == 0 else grille.columns)], axis=axe)
    if lignes == "Dest" or colonnes == "Dest":
        principales = vols["Dest"].value_counts().head(20).index
        grille = grille.loc[[i for i in grille.index if i in principales]] if lignes == "Dest" else grille[[c for c in grille.columns if c in principales]]
    if lignes == "compagnie":
        grille = grille.loc[grille.mean(axis=1).sort_values().index]
    etiquettes_y = [court(i) for i in grille.index]
    etiquettes_x = [court(c) for c in grille.columns]
    unite = "" if mesure == "nb_vols" else " %"
    figure = go.Figure(go.Heatmap(z=grille.values, x=etiquettes_x, y=etiquettes_y, xgap=2, ygap=2,
                                  colorscale=ECHELLE_ACCENT if mesure == "nb_vols" else ECHELLE_RETARD,
                                  text=[[("" if pd.isna(v) else (fr(v, 0) if mesure == "nb_vols" else fr(v, 0))) for v in ligne] for ligne in grille.values],
                                  texttemplate="%{text}", colorbar=dict(ticksuffix=unite),
                                  hovertemplate="%{y} × %{x} : <b>%{z:.1f}</b>" + unite + "<extra></extra>"))
    figure.update_xaxes(side="top")
    return figure, len(grille)


# --------------------------------------------------------------------------- Multivarié : bulles et jour × heure
def onglet_bulles():
    profil = profil_en_cache().copy()
    figure = go.Figure()
    for type_ in TYPES_ORDRE:
        p = profil[profil["type"] == type_]
        figure.add_trace(go.Scatter(x=p["pct_retard"], y=p["pct_annules"], mode="markers+text", name=type_,
                                    text=[court(c) for c in p.index], textposition="top center",
                                    marker=dict(size=np.sqrt(p["nb_vols"]) / 4 + 8, color=COULEURS_TYPES[type_], opacity=0.85,
                                                line=dict(color="white", width=2)),
                                    customdata=np.c_[p["nb_vols"], p["pct_grand_retard"]],
                                    hovertemplate="%{text}<br>%{x:.1f} % en retard · %{y:.1f} % annulés<br>"
                                                  "%{customdata[0]:,} vols · %{customdata[1]:.1f} % de grands retards<extra></extra>"))
    figure.add_vline(x=IND["pct_retard"], line_dash="dash", line_color=GRIS_CONTEXTE)
    figure.add_hline(y=IND["pct_annules"], line_dash="dash", line_color=GRIS_CONTEXTE)
    figure.update_xaxes(title="% de vols partis en retard", ticksuffix=" %")
    figure.update_yaxes(title="% de vols annulés", ticksuffix=" %")
    afficher(figure, 520)
    a_retenir("Quatre dimensions sur un seul graphique : retard (horizontal), annulations (vertical), taille (nombre de vols) et "
              "couleur (type). En haut à droite, les compagnies qui cumulent retards <b>et</b> annulations : JetBlue, Spirit. "
              "En bas à gauche, Endeavor. Delta, énorme, est proche de la moyenne (pointillés).")
    notion("graphique à bulles", "chaque point porte plusieurs informations (position, taille, couleur) : une figure multidimensionnelle.")


def onglet_jour_heure():
    grille = partis_tous.pivot_table(index="jour_semaine", columns="heure_depart_prevue", values="DepDel15", aggfunc="mean") * 100
    effectifs = partis_tous.pivot_table(index="jour_semaine", columns="heure_depart_prevue", values="DepDel15", aggfunc="size")
    grille = grille.where(effectifs >= 30).reindex(ORDRE_JOURS)
    figure = go.Figure(go.Heatmap(z=grille.values, x=[f"{h} h" for h in grille.columns], y=list(grille.index),
                                  colorscale=ECHELLE_RETARD, xgap=2, ygap=2, colorbar=dict(ticksuffix=" %"),
                                  hovertemplate="%{y}, %{x} : <b>%{z:.1f} %</b><extra></extra>"))
    figure.update_yaxes(autorange="reversed")
    afficher(figure, 340)
    a_retenir("Le retard monte de gauche à droite (le matin → le soir) <b>chaque jour</b> de la semaine ; les cases les plus rouges "
              "sont les soirs de jeudi et de vendredi. L'heure compte plus que le jour.")
    notion("carte de chaleur", "trois variables : le jour (lignes), l'heure (colonnes) et le % de retard (couleur).")


# --------------------------------------------------------------------------- Fiche compagnie
@st.cache_data
def profil_horaire(compagnie):
    p = partis_tous[partis_tous["compagnie"] == compagnie].groupby("tranche_horaire", observed=True)["DepDel15"].mean() * 100
    g = partis_tous.groupby("tranche_horaire", observed=True)["DepDel15"].mean() * 100
    return p.reindex(NOMS_TRANCHES), g.reindex(NOMS_TRANCHES)


def page_compagnie():
    C, CC = R["compagnies"], R.get("conclusions_compagnies", {})
    ordre = sorted(C, key=lambda c: C[c]["pct_retard_ou_annule"])
    choix = st.selectbox("Choisir une compagnie", ordre, format_func=court, key="fiche_compagnie")
    c, fiche = C[choix], CC.get(choix, {})
    hero(f"Fiche compagnie · {c['type']}", court(choix),
         fiche.get("conclusion", "") + (f"<br><i>{fiche['contexte']}</i>" if fiche.get("contexte") else ""))
    cartes = []
    for icone, libelle, cle, ref in [
            ("📊", "Retardés ou annulés", "pct_retard_ou_annule", IND["pct_retard_ou_annule"]),
            ("⏱️", "En retard (vols partis)", "pct_retard", IND["pct_retard"]),
            ("❌", "Annulés", "pct_annules", IND["pct_annules"]),
            ("🐢", "Grands retards (> 45 min)", "pct_grand_retard", R["classes_retard"]["Grand retard (> 45 min)"])]:
        ecart = c[cle] - ref
        couleur = "#1d8a5a" if ecart < 0 else ROUGE_RETARD
        cartes.append((icone, libelle, f"{fr(c[cle])} %", f"{'▼' if ecart < 0 else '▲'} {fr(abs(ecart))} pts vs moyenne ({fr(ref)} %)", couleur))
    cartes.append(("✈️", "Vols dans l'année", fr(c["nb_vols"], 0), f"{fr(c['part_vols_pct'])} % des vols d'Atlanta", ACCENT))
    cartes_kpi(cartes)
    gauche, droite = st.columns(2)
    with gauche:
        section("Selon la saison", "la compagnie (couleur) et la moyenne de l'aéroport (gris)")
        s = pd.Series(R["retard_par_saison_pct"][choix]).reindex(ORDRE_SAISONS)
        g = pd.Series(R["retard_par_saison_global_pct"]).reindex(ORDRE_SAISONS)
        figure = go.Figure([go.Bar(x=ORDRE_SAISONS, y=g.values, name="moyenne", marker_color=GRIS_CONTEXTE),
                            go.Bar(x=ORDRE_SAISONS, y=s.values, name=court(choix), marker_color=COULEURS_TYPES[c["type"]])])
        figure.update_yaxes(ticksuffix=" %")
        afficher(figure, 300)
    with droite:
        section("Selon l'heure de départ", "la compagnie et la moyenne")
        p, g = profil_horaire(choix)
        figure = go.Figure([go.Scatter(x=[t.split(" (")[0] for t in g.index], y=g.values, name="moyenne", mode="lines+markers",
                                       line=dict(color=GRIS_CONTEXTE, width=2)),
                            go.Scatter(x=[t.split(" (")[0] for t in p.index], y=p.values, name=court(choix), mode="lines+markers",
                                       line=dict(color=COULEURS_TYPES[c["type"]], width=3), marker=dict(size=9))])
        figure.update_yaxes(ticksuffix=" %", rangemode="tozero")
        afficher(figure, 300)
    section("Ses annulations jour par jour", "les bandes rouges sont les crises réelles qui la concernent")
    serie = annulations_journalieres_par_compagnie()[choix].dropna()
    figure = go.Figure(go.Bar(x=serie.index, y=serie.values,
                              marker_color=[ROUGE_RETARD if v > 10 else GRIS_CONTEXTE for v in serie.values],
                              hovertemplate="%{x|%d/%m/%Y} : <b>%{y:.0f} %</b> annulés<extra></extra>"))
    zones_evenements(figure, choix)
    figure.update_xaxes(tickformat="%m/%Y")
    figure.update_yaxes(ticksuffix=" %", range=[0, 100])
    afficher(figure, 280)
    if fiche.get("constats"):
        a_retenir("<br>".join("• " + x for x in fiche["constats"]))
    notion("comparer à une référence", "une valeur seule ne dit rien : on la compare à la moyenne de l'aéroport (les écarts en "
           "points, sous chaque carte : vert = mieux que la moyenne, rouge = moins bien).")


# --------------------------------------------------------------------------- Simulateur
CLE_MODELE = {"compagnie": "C(compagnie, Treatment('Delta'))[T.{}]", "tranche": "C(tranche, Treatment('Matin'))[T.{}]",
              "saison": "C(saison, Treatment('Automne'))[T.{}]", "jour": "C(jour, Treatment('Mardi'))[T.{}]",
              "precedent": "C(precedent, Treatment('a_l_heure'))[T.{}]"}


def risque(coef, compagnie, tranche, saison, jour, precedent, pluie, rafales, journee_pluvieuse):
    z = coef["Intercept"]
    for variable, valeur in [("compagnie", compagnie), ("tranche", tranche), ("saison", saison), ("jour", jour),
                             ("precedent", precedent)]:
        z += coef.get(CLE_MODELE[variable].format(valeur), 0.0)
    z += coef.get("pluie_depart", 0) * pluie + coef.get("rafales_fortes", 0) * rafales + coef.get("jour_pluvieux", 0) * journee_pluvieuse
    return 100 / (1 + np.exp(-z))


def page_simulateur():
    en_tete("Chapitre 8", "Prédire le risque de retard d'un vol",
            "Le modèle du chapitre 6 (régression logistique sur 314 844 vols) devient un outil : choisissez un vol, il calcule "
            "la probabilité qu'il parte avec plus de 15 minutes de retard.")
    if not EQ or "coefficients" not in EQ["modele"]:
        st.info("Lancez `python pipeline.py` pour calculer le modèle.")
        return
    coef = EQ["modele"]["coefficients"]
    c1, c2, c3, c4 = st.columns(4)
    compagnie = c1.selectbox("Compagnie", sorted(R["compagnies"], key=court), format_func=court,
                             index=sorted(R["compagnies"], key=court).index("Delta"), key="sim_cie")
    tranche = c2.selectbox("Heure de départ", ["Matin", "Midi", "Après-midi", "Soir"], index=3, key="sim_tranche")
    saison = c3.selectbox("Saison", ORDRE_SAISONS, index=3, key="sim_saison")
    jour = c4.selectbox("Jour", ORDRE_JOURS, index=4, key="sim_jour")
    c5, c6, c7, c8 = st.columns(4)
    precedent = c5.selectbox("L'avion arrive…", ["a_l_heure", "en_retard", "aucun"], key="sim_prec",
                             format_func={"a_l_heure": "à l'heure", "en_retard": "déjà en retard", "aucun": "1er départ du jour"}.get)
    pluie = c6.toggle("Pluie au départ", key="sim_pluie")
    rafales = c7.toggle("Rafales ≥ 40 km/h", key="sim_rafales")
    journee = c8.toggle("Journée pluvieuse", key="sim_journee")
    p = risque(coef, compagnie, tranche, saison, jour, precedent, pluie, rafales, journee)
    gauche, droite = st.columns([2, 3])
    with gauche:
        couleur = "#1d8a5a" if p < 15 else ORANGE if p < 30 else ROUGE_RETARD
        figure = go.Figure(go.Indicator(mode="gauge+number", value=p, number=dict(suffix=" %", font=dict(size=44, color=couleur)),
                                        gauge=dict(axis=dict(range=[0, 100], ticksuffix=" %"), bar=dict(color=couleur),
                                                   steps=[dict(range=[0, 15], color="#e3f1e8"), dict(range=[15, 30], color="#fdf0d9"),
                                                          dict(range=[30, 100], color="#fbe3e3")],
                                                   threshold=dict(line=dict(color=ENCRE, width=3), value=IND["pct_retard"]))))
        afficher(figure, 280)
        st.caption(f"Trait noir : la moyenne de l'aéroport ({fr(IND['pct_retard'])} %).")
    with droite:
        heures = {t: risque(coef, compagnie, t, saison, jour, precedent, pluie, rafales, journee) for t in ["Matin", "Midi", "Après-midi", "Soir"]}
        cies = {c: risque(coef, c, tranche, saison, jour, precedent, pluie, rafales, journee) for c in R["compagnies"]}
        meilleure_heure = min(heures, key=heures.get)
        meilleure_cie = min(cies, key=cies.get)
        figure = go.Figure(go.Bar(x=list(heures), y=list(heures.values()),
                                  marker_color=[ACCENT if t == tranche else GRIS_CONTEXTE for t in heures],
                                  text=[f"{fr(v)} %" for v in heures.values()], textposition="outside",
                                  hovertemplate="%{x} : <b>%{y:.1f} %</b><extra></extra>"))
        figure.update_yaxes(ticksuffix=" %", range=[0, max(heures.values()) * 1.25])
        st.markdown("**Le même vol, à une autre heure**")
        afficher(figure, 250)
    conseils = []
    if meilleure_heure != tranche:
        conseils.append(f"partir le <b>{meilleure_heure.lower()}</b> : {fr(heures[meilleure_heure])} % au lieu de {fr(p)} %")
    if meilleure_cie != compagnie:
        conseils.append(f"à conditions égales, <b>{court(meilleure_cie)}</b> donnerait {fr(cies[meilleure_cie])} %")
    if precedent == "en_retard":
        conseils.append(f"si l'avion arrivait à l'heure : {fr(risque(coef, compagnie, tranche, saison, jour, 'a_l_heure', pluie, rafales, journee))} %"
                        " (pour la compagnie : du temps tampon entre deux vols)")
    a_retenir(f"Risque estimé : <b>{fr(p)} %</b>. " + ("Pour le réduire : " + " ; ".join(conseils) + "." if conseils else
              "C'est déjà la combinaison la plus sûre pour ces conditions."))
    notion("prédiction", "p = 1 ÷ (1 + e<sup>−z</sup>), avec z = la somme des coefficients du modèle pour les choix faits. "
           f"C'est une estimation moyenne (AUC = {fr(EQ['modele']['auc'], 2)}) : elle dit un risque, pas une certitude.")


# --------------------------------------------------------------------------- Échantillonnage : ce que l'échantillon répond seul
def onglet_reponses_echantillon():
    st.markdown("#### Ce que les échantillons répondent, sans regarder les 319 335 vols")
    RE = APP.get("reponses_echantillon")
    if not RE:
        st.info("Lancez `python pipeline.py` pour calculer les réponses des échantillons.")
        return
    resume = pd.DataFrame(RE["par_compagnie"]).T.sort_values("mediane")
    types = {c: R["compagnies"][c]["type"] for c in resume.index}
    figure = go.Figure()
    for c, ligne in resume.iterrows():
        figure.add_trace(go.Scatter(x=[ligne["bas"], ligne["haut"]], y=[court(c)] * 2, mode="lines", showlegend=False,
                                    line=dict(color=COULEURS_TYPES[types[c]], width=8), opacity=0.35, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=resume["mediane"], y=[court(c) for c in resume.index], mode="markers", showlegend=False,
                                marker=dict(size=12, color=[COULEURS_TYPES[types[c]] for c in resume.index], line=dict(color="white", width=2)),
                                customdata=resume[["bas", "haut"]].to_numpy(),
                                hovertemplate="%{y} : médiane <b>%{x:.1f} %</b><br>90 % des échantillons entre %{customdata[0]:.1f} "
                                              "et %{customdata[1]:.1f} %<extra></extra>"))
    figure.update_xaxes(title="% retardés ou annulés, estimé par 30 échantillons stratifiés de 5 000 vols", ticksuffix=" %")
    afficher(figure, 470)
    a_retenir("Avec seulement 5 000 vols (357 par compagnie), l'échantillon donne <b>la même réponse</b> : régionales en bas, bas coût en "
              "haut, JetBlue la plus perturbée. Les barres montrent la <b>plage de stabilité</b> : quand deux barres se chevauchent, "
              "l'échantillon ne peut pas départager les deux compagnies.")
    st.markdown("#### La stabilité de chaque statistique d'un échantillon à l'autre (30 tirages au hasard de 5 000 vols)")
    tableau = pd.DataFrame(RE["stabilite"]).T.rename(columns={"moyenne": "Moyenne des tirages", "ecart_type": "Écart-type",
                                                              "cv_pct": "Coefficient de variation (%)"})
    tableau["Stabilité"] = np.where(tableau["Coefficient de variation (%)"] < 5, "très stable",
                                    np.where(tableau["Coefficient de variation (%)"] < 15, "stable", "instable"))
    st.dataframe(tableau, width="stretch")
    a_retenir("Les pourcentages (retard, écart soir − matin) sont <b>stables</b> d'un tirage à l'autre ; les statistiques qui dépendent "
              "des valeurs extrêmes (retard moyen) ou d'un petit nombre de cas (annulations, domino) <b>bougent davantage</b>.")
    notion("coefficient de variation", "CV = écart-type ÷ moyenne × 100 : il permet de comparer la stabilité de statistiques "
           "d'unités différentes. Moins de 5 % : très stable.")


# --------------------------------------------------------------------------- Recommandations
def recommandations():
    section("Ce que l'on peut conseiller", "tiré des chiffres de l'étude")
    t = R["retard_par_tranche_pct"]
    blocs = [
        ("🧳 Au voyageur", [f"partir le <b>matin</b> : {fr(t['Matin (avant 10 h)'])} % de retard contre {fr(t['Soir (après 18 h)'])} % le soir",
                           "préférer <b>mardi ou mercredi</b>", "préférer une <b>régionale</b> ou Delta à un bas coût",
                           "éviter les vols du <b>soir en été</b> des bas coût (Southwest : 65 %)",
                           "garder une marge pour une correspondance le soir"]),
        ("🛫 Aux compagnies", [f"ajouter du <b>temps tampon</b> entre deux vols du même avion le soir (effet domino : "
                              f"{fr(IND['domino_pct_si_precedent_a_l_heure'])} → {fr(IND['domino_pct_si_precedent_en_retard'])} %)",
                              "renforcer les équipages en <b>été</b> et pendant les fêtes (Omicron, Juneteenth)",
                              "suivre le risque <b>jour par jour</b>, pas par mois"]),
        ("🏢 À l'aéroport", ["deux plans de crise : <b>orages</b> → beaucoup de retards, peu d'annulations ; <b>neige / verglas</b> → "
                            "beaucoup d'annulations", "surveiller les crises <b>nées ailleurs</b> (New York, Texas, Floride)",
                            "utiliser l'entrepôt de données pour un tableau de bord quotidien"]),
    ]
    for colonne, (titre, lignes) in zip(st.columns(3), blocs):
        colonne.markdown(f'<div class="carte"><h4>{titre}</h4><p>' + "<br>".join("• " + ligne for ligne in lignes) + "</p></div>",
                         unsafe_allow_html=True)
    st.write("")


# =========================================================================== Figures interactives (remplacent les images PNG)
def base_sql_disponible():
    return FICHIER_BASE.exists()


@st.cache_data
def sql(requete_sql):
    return requete(requete_sql)


def figure_saisons():
    jour = serie_journaliere_en_cache()
    saison_du_jour = partis_tous.groupby("FlightDate")["saison"].first()
    par_saison = R["retard_par_saison_global_pct"]
    fonds = {"Automne": "#f6efe3", "Hiver": "#e9f0f7", "Printemps": "#eaf4ec", "Été": "#fbe9e9"}
    figure = go.Figure()
    blocs = (saison_du_jour != saison_du_jour.shift()).cumsum()
    for _, bloc in saison_du_jour.groupby(blocs):
        saison, debut, fin = bloc.iloc[0], bloc.index[0], bloc.index[-1] + pd.Timedelta(days=1)
        figure.add_vrect(x0=debut, x1=fin, fillcolor=fonds[saison], opacity=1, line_width=0, layer="below")
        if len(bloc) > 40:
            figure.add_annotation(x=debut + (fin - debut) / 2, y=74, text=f"<b>{saison}</b><br>{fr(par_saison[saison])} %",
                                  showarrow=False, font=dict(size=12, color=ENCRE))
    figure.add_trace(go.Scatter(x=jour.index, y=jour["pct_retard"], name="chaque jour", mode="lines",
                                line=dict(color=GRIS_CONTEXTE, width=1), hovertemplate="%{x|%d/%m/%Y} : <b>%{y:.1f} %</b><extra></extra>"))
    figure.add_trace(go.Scatter(x=jour.index, y=jour["moyenne_7_jours"], name="moyenne sur 7 jours", mode="lines",
                                line=dict(color=ROUGE_RETARD, width=2.5), hoverinfo="skip"))
    figure.add_hline(y=IND["pct_retard"], line_dash="dash", line_color=ENCRE_SECONDAIRE,
                     annotation_text=f"moyenne de l'année {fr(IND['pct_retard'])} %", annotation_position="bottom right")
    figure.update_yaxes(ticksuffix=" %", range=[0, 78], title="% de vols partis en retard")
    figure.update_xaxes(tickformat="%m/%Y")
    return figure


@st.cache_data
def figure_densite_depart_arrivee():
    d = partis_tous[["DepDelay", "ArrDelay"]].dropna()
    d = d[d["DepDelay"].between(-30, 240) & d["ArrDelay"].between(-60, 240)]
    comptes, bx, by = np.histogram2d(d["DepDelay"], d["ArrDelay"], bins=[90, 100])
    z = np.log10(np.where(comptes > 0, comptes, np.nan))
    figure = go.Figure(go.Heatmap(x=(bx[:-1] + bx[1:]) / 2, y=(by[:-1] + by[1:]) / 2, z=z.T, colorscale="YlOrRd",
                                  customdata=comptes.T, colorbar=dict(title="vols", tickvals=[0, 1, 2, 3, 4],
                                                                      ticktext=["1", "10", "100", "1 000", "10 000"]),
                                  hovertemplate="départ %{x:.0f} min, arrivée %{y:.0f} min : <b>%{customdata:,.0f}</b> vols<extra></extra>"))
    figure.add_trace(go.Scatter(x=[-30, 240], y=[-30, 240], mode="lines", line=dict(color=ACCENT, dash="dash"),
                                name="arrivée = départ", hoverinfo="skip"))
    figure.update_xaxes(title="Retard au départ (min)")
    figure.update_yaxes(title="Retard à l'arrivée (min)")
    return figure


def figure_meteo():
    if not base_sql_disponible():
        return None
    j = sql("SELECT date_vol, pct_retard, heures_de_pluie, rafales_max_kmh, neige_cm FROM mv_jour")
    moyennes = j.groupby(pd.cut(j["heures_de_pluie"], [-0.1, 0, 3, 6, 12, 24]), observed=True)["pct_retard"].mean()
    figure = go.Figure(go.Scatter(x=j["heures_de_pluie"], y=j["pct_retard"], mode="markers",
                                  marker=dict(size=7, color=GRIS_FONCE, opacity=0.45), name="un jour",
                                  customdata=np.c_[j["date_vol"], j["rafales_max_kmh"], j["neige_cm"]],
                                  hovertemplate="%{customdata[0]} : <b>%{y:.1f} %</b> en retard<br>%{x:.0f} h de pluie · rafales "
                                                "%{customdata[1]:.0f} km/h · neige %{customdata[2]:.1f} cm<extra></extra>"))
    figure.add_trace(go.Scatter(x=[0, 1.5, 4.5, 9, 18][:len(moyennes)], y=moyennes.values, mode="lines+markers",
                                line=dict(color=ROUGE_RETARD, width=3), marker=dict(size=10), name="moyenne par classe de pluie"))
    figure.update_xaxes(title="Heures de pluie dans la journée à Atlanta (Open-Meteo, fusion SQL)")
    figure.update_yaxes(title="% de vols partis en retard ce jour-là", ticksuffix=" %")
    return figure


def carte_destinations():
    if not base_sql_disponible():
        return None, None
    etats = sql("""SELECT DestState AS etat, MIN(DestStateName) AS nom, COUNT(*) AS nb_vols,
                          100.0 * AVG(CASE WHEN Cancelled = 0 THEN DepDel15 END) AS pct_retard
                   FROM vols GROUP BY DestState""")
    figure = go.Figure(go.Choropleth(locations=etats["etat"], z=etats["pct_retard"], locationmode="USA-states",
                                     colorscale=ECHELLE_RETARD, marker_line_color="white", colorbar=dict(ticksuffix=" %"),
                                     customdata=np.c_[etats["nom"], etats["nb_vols"]],
                                     hovertemplate="%{customdata[0]} : <b>%{z:.1f} %</b> en retard<br>%{customdata[1]:,} vols depuis Atlanta<extra></extra>"))
    figure.add_trace(go.Scattergeo(lon=[-84.43], lat=[33.64], text=["ATL"], mode="markers+text", textposition="bottom center",
                                   marker=dict(size=14, color=ACCENT, symbol="star"), hoverinfo="skip", showlegend=False))
    figure.update_geos(scope="usa", bgcolor="rgba(0,0,0,0)")
    destinations = sql("SELECT dest, ville, nb_vols, pct_retard FROM mv_destination ORDER BY pct_retard")
    return figure, destinations


def figure_destinations(destinations):
    choix = pd.concat([destinations.head(8), destinations.tail(8)]).sort_values("pct_retard")
    figure = go.Figure(go.Bar(x=choix["pct_retard"], y=[f"{v} ({d})" for v, d in zip(choix["ville"], choix["dest"])], orientation="h",
                              marker_color=[ACCENT if p < IND["pct_retard"] else ROUGE_RETARD for p in choix["pct_retard"]],
                              text=[f"{fr(p)} %" for p in choix["pct_retard"]], textposition="outside",
                              customdata=choix["nb_vols"], hovertemplate="%{y} : <b>%{x:.1f} %</b><br>%{customdata:,} vols<extra></extra>"))
    figure.add_vline(x=IND["pct_retard"], line_dash="dash", line_color=ENCRE_SECONDAIRE)
    figure.update_xaxes(ticksuffix=" %", range=[0, choix["pct_retard"].max() * 1.18])
    return figure


@st.cache_data
def figure_annulations_semaine():
    v = vols.assign(semaine=vols["FlightDate"].dt.to_period("W-SUN").dt.start_time)
    g = v.groupby(["compagnie", "semaine"], observed=True)["Cancelled"].agg(["mean", "size"])
    grille = (g["mean"] * 100).where(g["size"] >= 10).unstack()
    ordre = profil_en_cache().sort_values("pct_retard_ou_annule").index
    grille = grille.reindex(ordre)
    figure = go.Figure(go.Heatmap(z=grille.values, x=list(grille.columns), y=[court(c) for c in grille.index],
                                  colorscale=[[0, "#fbf3f3"], [1, "#7a0f1c"]], zmin=0, zmax=40, xgap=1, ygap=2,
                                  colorbar=dict(ticksuffix=" %", title="annulés"),
                                  hovertemplate="%{y}, semaine du %{x|%d/%m/%Y} : <b>%{z:.1f} %</b> annulés<extra></extra>"))
    figure.update_xaxes(tickformat="%m/%Y")
    return figure


def figure_profil_compagnies():
    profil = profil_en_cache().sort_values("pct_retard_ou_annule")
    colonnes = {"pct_retard_ou_annule": "Retardés ou annulés (%)", "pct_annules": "Annulés (%)",
                "pct_grand_retard": "Grands retards > 45 min (%)", "retard_p90_min": "Retard du 90e centile (min)",
                "roulage_median_min": "Roulage médian (min)", "rattrapage_median_min": "Rattrapé en vol (min)",
                "part_vols_soir_pct": "Vols après 18 h (%)"}
    valeurs = profil[list(colonnes)]
    z = (valeurs - valeurs.mean()) / valeurs.std()
    z["rattrapage_median_min"] = -z["rattrapage_median_min"]          # rattraper plus = mieux : on inverse la couleur
    figure = go.Figure(go.Heatmap(z=z.values, x=list(colonnes.values()), y=[f"{court(c)} ({t})" for c, t in zip(profil.index, profil["type"])],
                                  colorscale="RdBu_r", zmid=0, xgap=2, ygap=2, text=valeurs.round(1).values, texttemplate="%{text}",
                                  colorbar=dict(title="écart à la<br>moyenne<br>(écarts-types)"),
                                  hovertemplate="%{y}<br>%{x} : <b>%{text}</b><extra></extra>"))
    figure.update_xaxes(side="top")
    return figure


@st.cache_data
def acp_compagnies():
    from sklearn.decomposition import PCA
    profil = profil_en_cache()
    colonnes = {"pct_retard": "% retard", "pct_annules": "% annulés", "pct_grand_retard": "% grands retards",
                "retard_p90_min": "retard P90", "roulage_median_min": "roulage", "rattrapage_median_min": "rattrapé en vol",
                "distance_mediane": "distance", "part_vols_soir_pct": "% vols du soir"}
    x = profil[list(colonnes)]
    xs = (x - x.mean()) / x.std()
    acp = PCA(n_components=2).fit(xs)
    coords = acp.transform(xs)
    correlations = acp.components_.T * np.sqrt(acp.explained_variance_)
    return profil, coords, correlations, list(colonnes.values()), acp.explained_variance_ratio_


def onglet_acp():
    profil, coords, correlations, noms, variance = acp_compagnies()
    gauche, droite = st.columns(2)
    with gauche:
        st.markdown(f"**Le cercle des corrélations** (axes 1 et 2 : {fr(variance[0] * 100)} % + {fr(variance[1] * 100)} % de l'information)")
        t = np.linspace(0, 2 * np.pi, 200)
        figure = go.Figure(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color=GRILLE), hoverinfo="skip", showlegend=False))
        for (cx, cy), nom in zip(correlations, noms):
            figure.add_annotation(x=cx, y=cy, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                                  arrowhead=2, arrowcolor=ACCENT, arrowwidth=2)
            figure.add_annotation(x=cx * 1.12, y=cy * 1.12, text=nom, showarrow=False, font=dict(size=11, color=ENCRE))
        figure.update_xaxes(range=[-1.3, 1.3], autorange=False, title=f"Axe 1 ({fr(variance[0] * 100)} %)", zeroline=True)
        figure.update_yaxes(range=[-1.3, 1.3], autorange=False, title=f"Axe 2 ({fr(variance[1] * 100)} %)", zeroline=True)
        afficher(figure, 460)
    with droite:
        st.markdown("**Les 14 compagnies dans le plan de l'ACP**")
        figure = go.Figure()
        for type_ in TYPES_ORDRE:
            masque = (profil["type"] == type_).to_numpy()
            figure.add_trace(go.Scatter(x=coords[masque, 0], y=coords[masque, 1], mode="markers+text", name=type_,
                                        text=[court(c) for c in profil.index[masque]], textposition="top center",
                                        marker=dict(size=13, color=COULEURS_TYPES[type_], line=dict(color="white", width=2)),
                                        hovertemplate="%{text}<extra></extra>"))
        figure.add_hline(y=0, line_color=GRILLE)
        figure.add_vline(x=0, line_color=GRILLE)
        figure.update_xaxes(title="Axe 1")
        figure.update_yaxes(title="Axe 2")
        afficher(figure, 460)
    a_retenir(f"Les deux premiers axes résument {fr((variance[0] + variance[1]) * 100)} % des différences entre compagnies. "
              "Les flèches proches l'une de l'autre sont des indicateurs qui vont ensemble (retards, grands retards et annulations). "
              "Les compagnies <b>à bas coût</b> se regroupent du côté de ces flèches, les <b>régionales</b> à l'opposé : "
              "la typologie apparaît <b>sans qu'on la donne</b> à la méthode.")
    notion("ACP (analyse en composantes principales)", "résumer beaucoup de variables en 2 axes qui gardent le plus d'information ; "
           "le cercle montre comment chaque variable est liée aux axes (longueur de la flèche = qualité de représentation).")


def figure_modele_rapports():
    rapports = pd.DataFrame(EQ["modele"]["rapports_de_cotes"]).T
    rapports = rapports[~rapports.index.str.startswith("Jour :")].sort_values("rapport_de_cotes")
    couleurs = [GRIS_CONTEXTE if (b <= 1 <= h) else (ROUGE_RETARD if r > 1 else ACCENT)
                for r, b, h in zip(rapports["rapport_de_cotes"], rapports["ic_bas"], rapports["ic_haut"])]
    figure = go.Figure(go.Scatter(x=rapports["rapport_de_cotes"], y=list(rapports.index), mode="markers",
                                  marker=dict(size=11, color=couleurs, line=dict(color="white", width=1)),
                                  error_x=dict(type="data", symmetric=False, array=rapports["ic_haut"] - rapports["rapport_de_cotes"],
                                               arrayminus=rapports["rapport_de_cotes"] - rapports["ic_bas"], color=GRIS_FONCE, thickness=1.5),
                                  hovertemplate="%{y} : <b>× %{x:.2f}</b><extra></extra>"))
    figure.add_vline(x=1, line_color=ENCRE_SECONDAIRE)
    figure.update_xaxes(type="log", tickvals=[0.5, 0.75, 1, 1.5, 2, 3], ticktext=["0,5", "0,75", "1", "1,5", "2", "3"],
                        title="Rapport de cotes (échelle log) — à droite de 1 : plus de retard que la référence")
    return figure, len(rapports)


def figure_importance():
    importance = pd.Series(EQ["modele"]["importance_pseudo_r2"]).sort_values()
    figure = go.Figure(go.Bar(x=importance.values, y=list(importance.index), orientation="h",
                              marker_color=[ROUGE_RETARD if v == importance.max() else GRIS_FONCE for v in importance.values],
                              text=[fr(v, 4) for v in importance.values], textposition="outside",
                              hovertemplate="%{y} : perte de pseudo-R² <b>%{x:.4f}</b><extra></extra>"))
    figure.update_xaxes(title="Ce que le modèle perd si on retire ce facteur (pseudo-R²)", range=[0, importance.max() * 1.2])
    return figure


def page_donnees():
    chapitre_donnees()
    entrepot_de_donnees()


def page_descriptif_univarie():
    chapitre_descriptif()
    chapitre_univarie()


# --------------------------------------------------------------------------- Affichage de la partie choisie
PAGES = dict(zip(CHAPITRES, [page_accueil, page_donnees, chapitre_qualite, page_descriptif_univarie, chapitre_echantillonnage,
                             chapitre_bivarie, chapitre_multivarie, page_compagnie, page_simulateur, chapitre_conclusion]))
PAGES[st.session_state.chapitre]()

# Quand on change de chapitre, on remonte en haut de la page (sinon Streamlit garde la position du défilement)
components.html(f"<script>/* {st.session_state.chapitre} */ function haut() {{"
                "var p = window.parent.document; var zone = p.querySelector('[data-testid=stMain]');"
                "if (zone) zone.scrollTo(0, 0); window.parent.scrollTo(0, 0); }"
                "haut(); setTimeout(haut, 300); setTimeout(haut, 1000);</script>", height=0)

st.divider()
colonne_avant, _, colonne_apres = st.columns([1, 2, 1])
position = CHAPITRES.index(st.session_state.chapitre)
if position > 0:
    colonne_avant.button(f"← {CHAPITRES[position - 1]}", on_click=aller_a, args=(-1,), width="stretch")
if position < len(CHAPITRES) - 1:
    colonne_apres.button(f"{CHAPITRES[position + 1]} →", on_click=aller_a, args=(1,), type="primary", width="stretch")
st.markdown('<div class="source">Source : Bureau of Transportation Statistics (BTS), via Kaggle « Flight Status Prediction » '
            '(Rob Mulla, CC0). Mêmes chiffres que le notebook et le rapport (reports/resultats.json).</div>',
            unsafe_allow_html=True)

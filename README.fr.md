# ✈️ Qu'est-ce qui distingue les compagnies aériennes au départ d'Atlanta ?

> **Comparaison de 14 compagnies sur 319 335 vols (août 2021 – juillet 2022), effets de composition,
> propagation des retards, et une étude d'échantillonnage qui vérifie quelles conclusions résistent.**

🇬🇧 [English version](README.md) · Auteure : **Salma Bouhmid** (ingénierie des données).

## Question principale et questions secondaires

> Qu'est-ce qui distingue les compagnies aériennes au départ d'Atlanta (août 2021 – juillet 2022),
> quels facteurs observables sont associés à ces différences, et ces conclusions résistent-elles à l'échantillonnage ?

| | Question |
|---|---|
| Q1 | Comment les compagnies diffèrent-elles (ponctualité, gravité, annulations, exploitation) ? |
| Q2 | Ces écarts sont-ils associés à *quand* elles volent (heure, mois) ? Effet de composition ? |
| Q3 | Le classement change-t-il selon la saison ? |
| Q4 | Quand le retard apparaît-il et se propage-t-il d'un vol au suivant (effet domino) ? |
| Q5 | Quelles méthodes d'échantillonnage, à quelle taille, préservent ces conclusions ? |
| Q6 | Quels biais et limites encadrent ces conclusions ? |

## Données

- **Source** : Kaggle [Flight Status Prediction](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022)
  (Rob Mulla, CC0), issu du Bureau of Transportation Statistics (BTS).
- **Population** : tous les vols programmés au départ d'Atlanta du 01/08/2021 au 31/07/2022 :
  **319 335 vols × 61 colonnes**, 365 jours, 14 compagnies. Une ligne = un vol.
- **Pourquoi 12 mois** : l'été seul surestime le retard (26,2 % contre 18,6 % sur l'année) et ne permet pas
  de comparer les compagnies sur toutes les saisons ; le fichier 2022 s'arrête en juillet.

## Principaux résultats

- Les compagnies **régionales** sont les plus ponctuelles (Endeavor : 10 % de vols retardés ou annulés),
  les compagnies **à bas coût** les moins ponctuelles (JetBlue : 37 %), avec des retards plus graves. *Solide.*
- Une partie des écarts est un **effet de composition** : à heure égale, American passe de 17,4 % à 20,9 % de retard ;
  à mois égal, SkyWest recule du 5ᵉ au 11ᵉ rang.
- **Effet domino** : 40,5 % de retard quand le départ précédent du même avion était en retard, contre 15,4 % sinon.
- **Échantillonnage** : aucune méthode n'est la meilleure partout. Les grandes conclusions survivent à toutes
  les méthodes ; les écarts fins entre compagnies proches, non. Pour comparer les compagnies, il faut stratifier
  à part égale (et repondérer pour tout chiffre global) ; les grappes de jours sont les moins fiables.

- **Vu jour par jour**, le retard arrive par **crises courtes** (de 3 % à 64 % selon le jour) : dix événements réels
  (crise de Spirit en août 2021, Southwest en octobre, Omicron à Noël, tempête Izzy le 16 janvier 2022...), chacun
  visible dans les données et documenté par une source (`src/contexte.py`).
- **Nuances** : le milieu du classement n'est pas départageable (28 paires sur 91, test avec correction de Bonferroni) ;
  l'avantage des régionales est surtout celui d'Endeavor ; l'heure de départ est aussi liée au retard que le type de compagnie.

- **Base de données** : les vols sont fusionnés avec 5 tables (compagnies, destinations, calendrier, événements réels,
  météo d'Atlanta heure par heure) dans une base SQLite ; 11 vues SQL (une par question) sont stockées puis testées
  (mêmes chiffres en SQL et en Python).
- **Météo** (fusion Open-Meteo) : 14,1 % de retard un jour sec, 24,5 % un jour de forte pluie ; les deux jours anormaux
  sans source de presse sont expliqués par la pluie et le vent.
- **Modèle** (régression logistique, tous les facteurs ensemble) : à conditions égales, partir le soir multiplie la cote
  du retard par 3,2, un avion déjà en retard par 3,1, JetBlue par 2,7 (vs Delta) ; **l'heure pèse plus que la compagnie**.
  Du meilleur au pire vol possible : de 1,6 % à 81,5 % de risque.
- **Compagnies dans le monde réel** : le code OH « Comair » désigne en réalité PSA Airlines (Comair a fermé en 2012).

*Associations observées, pas des preuves de causalité.*

### Limites techniques connues (assumées)

- **Schéma en étoile** : la table de faits `vols` garde aussi des colonnes descriptives (saison, type de compagnie…) pour
  simplifier les vues ; un schéma strict ne garderait que les clés et les mesures (la base serait plus légère que 163 Mo).
- **Effet domino** : seuls les départs d'Atlanta sont dans le fichier ; le « vol précédent » est le départ précédent d'ATL
  du même avion le même jour (les vols intermédiaires vers ailleurs ne sont pas visibles).
- **Modèle** : régression logistique sans interactions ; AUC 0,69 sur des vols jamais vus (évaluation 70 / 30) :
  il sert à comparer les facteurs, pas à prédire chaque vol.
- **Carte** : par État de destination (le fichier n'a pas les coordonnées des aéroports). Tous les chiffres : [`reports/resultats.json`](reports/resultats.json).

## Méthode

1. **Extraction** des 12 mois (CSV 2021 zippé lu par morceaux + Parquet 2022) — `src/data_loading.py`
2. **Qualité et nettoyage** : motif des manquants (en bloc : vols annulés), biais du survivant corrigé
   par l'indicateur « retardé ou annulé », type de compagnie, classes de gravité — `src/preprocessing.py`
3. **Statistiques descriptives** (centre, dispersion, forme, extrêmes, règle IQR) — `src/analysis.py`
4. **Comparaison des compagnies** : profils, standardisation directe, corrélation globale contre par compagnie
   (effet de type Simpson), saisons, domino — `src/comparison.py`
5. **Échantillonnage** : 6 plans classiques (aléatoire simple, systématique, stratifié par compagnie, par grappes,
   temporel, répété/bootstrap), 7 expériences E1-E7, 100 tirages par méthode,
   tailles de 1 000 à 20 000 vols — `src/sampling.py`, `src/sampling_study.py`
6. **Approfondissement** : jour par jour, jour de la semaine, force des liens (V de Cramér, Spearman, Kruskal-Wallis),
   tests par paires, plusieurs versions de la classification, valeurs extrêmes, contexte réel — `src/approfondissement.py`,
   `src/contexte.py`
7. **Base de données** : fusion en schéma en étoile, 11 vues SQL matérialisées (tables `mv_*`) —
   `sql/01_tables.sql`, `sql/02_vues.sql`, `src/base_de_donnees.py`
8. **Modèle** (data scientist) : régression logistique, rapports de cotes, importance des facteurs, scénarios — `src/modele.py`
9. **Regard métier** (data analyst) : destinations, créneaux, records, météo — `src/analyste.py`
10. **Figures** avec une charte commune (une couleur = un sens ; n, période et source sur chaque graphique) — `src/figures.py`

## Lancer le projet

Pour ouvrir le dashboard : double-cliquer sur `lancer_dashboard.py` (ou `python lancer_dashboard.py`). Il installe ce qui manque, prépare les données au premier lancement, puis ouvre le navigateur.

```bash
pip install -r requirements.txt
python pipeline.py                      # nettoyage + base SQLite (fusion + vues) + resultats.json
                                        # (python pipeline.py --extraire pour tout refaire depuis Kaggle)
python src/sampling_study.py            # étude d'échantillonnage (environ 15 minutes)
python src/figures.py                   # toutes les figures
python outils/construire_notebook.py    # réécrit le notebook à partir des résultats
streamlit run dashboard/app.py          # dashboard
python -m pytest                        # 91 tests
```

Pour tout refaire depuis Kaggle : placer `Combined_Flights_2021.csv.zip` et `Combined_Flights_2022.parquet`
dans `data/raw/` (non versionnés, trop volumineux).

## Structure

```
atlanta-airline-performance/
├── pipeline.py               # extraction → nettoyage → résultats, en une commande
├── src/                      # chargement, nettoyage, analyses, échantillonnage, base SQL, modèle, figures
├── sql/                      # 01_tables.sql (schéma en étoile) et 02_vues.sql (11 vues)
├── data/external/            # météo d'Atlanta (Open-Meteo), jour et heure
├── outils/                   # générateur du notebook
├── notebooks/analysis.ipynb  # le récit complet
├── dashboard/app.py          # dashboard Streamlit raconté en chapitres (prologue + 9 chapitres)
├── reports/                  # resultats.json, figures/, echantillonnage/
├── tests/                    # 91 tests automatiques
└── docs/                     # documentation des données
```

## Technologies

Python · pandas · NumPy · pyarrow · SQLite (SQL, vues, fonctions de fenêtre) · statsmodels · scikit-learn · Matplotlib · Plotly · Streamlit · Jupyter · pytest

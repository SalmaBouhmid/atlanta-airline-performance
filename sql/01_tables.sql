-- =====================================================================================
-- Base de données du projet : vols au départ d'Atlanta (août 2021 – juillet 2022)
-- 01_tables.sql : les tables (le « modèle en étoile » : une table de faits + des dimensions)
--
--   vols            (faits)       une ligne = un vol programmé au départ d'Atlanta
--   compagnies      (dimension)   une ligne = une compagnie : type, maison mère, base, remarque
--   destinations    (dimension)   une ligne = un aéroport de destination
--   calendrier      (dimension)   une ligne = un jour : jour de la semaine, saison, jour férié
--   evenements      (dimension)   une ligne = un jour touché par un événement réel documenté
--   meteo_jour      (externe)     la météo d'Atlanta jour par jour (Open-Meteo)
--   meteo_heure     (externe)     la météo d'Atlanta heure par heure (Open-Meteo)
--
-- Les tables sont remplies par src/base_de_donnees.py (depuis les fichiers du projet),
-- puis 02_vues.sql crée les vues (tables virtuelles), une par question de l'étude.
-- =====================================================================================

DROP TABLE IF EXISTS vols;
DROP TABLE IF EXISTS compagnies;
DROP TABLE IF EXISTS destinations;
DROP TABLE IF EXISTS calendrier;
DROP TABLE IF EXISTS evenements;
DROP TABLE IF EXISTS meteo_jour;
DROP TABLE IF EXISTS meteo_heure;

CREATE TABLE compagnies (
    compagnie          TEXT PRIMARY KEY,   -- nom court utilisé dans tout le projet
    nom_bts            TEXT,               -- nom tel qu'il figure dans le fichier BTS / Kaggle
    code_iata          TEXT,               -- code de la compagnie qui opère le vol
    type_compagnie     TEXT,               -- Majeure, Régionale, Bas coût
    vend_sous_la_marque TEXT,              -- marque(s) commerciale(s) sous laquelle le vol est vendu
    base_principale    TEXT,               -- hub ou base principale (contexte réel)
    remarque           TEXT                -- information du monde réel utile à la lecture des données
);

CREATE TABLE destinations (
    dest               TEXT PRIMARY KEY,   -- code IATA de l'aéroport d'arrivée
    ville              TEXT,
    etat               TEXT,
    distance_miles     REAL
);

CREATE TABLE calendrier (
    date_vol           TEXT PRIMARY KEY,   -- AAAA-MM-JJ
    jour_semaine       TEXT,
    num_jour_semaine   INTEGER,            -- 1 = lundi ... 7 = dimanche
    annee_mois         TEXT,
    saison             TEXT,
    jour_ferie         TEXT                -- nom du jour férié fédéral américain, sinon NULL
);

CREATE TABLE evenements (
    date_vol           TEXT,
    numero             INTEGER,
    nom                TEXT,
    compagnies_concernees TEXT,            -- « toutes » ou liste de compagnies
    source             TEXT
);

CREATE TABLE meteo_jour (
    date_vol           TEXT PRIMARY KEY,
    code_meteo         INTEGER,            -- code WMO (Open-Meteo)
    temperature_max    REAL,
    temperature_min    REAL,
    pluie_mm           REAL,
    neige_cm           REAL,
    heures_de_pluie    REAL,
    vent_max_kmh       REAL,
    rafales_max_kmh    REAL
);

CREATE TABLE meteo_heure (
    date_vol           TEXT,
    heure              INTEGER,            -- 0 à 23, heure locale d'Atlanta
    code_meteo         INTEGER,
    pluie_mm           REAL,
    neige_cm           REAL,
    vent_kmh           REAL,
    rafales_kmh        REAL,
    visibilite_m       REAL,
    nuages_pct         REAL,
    PRIMARY KEY (date_vol, heure)
);

-- La table des vols reprend les colonnes du fichier nettoyé (67 colonnes) ; elle est créée
-- par pandas (to_sql) pour garder exactement les mêmes noms et types que dans le notebook.

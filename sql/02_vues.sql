-- =====================================================================================
-- 02_vues.sql : les vues (tables virtuelles), une par question de l'étude
--
-- Une vue ne stocke pas de données : c'est une requête enregistrée sous un nom.
-- src/base_de_donnees.py les « matérialise » ensuite (CREATE TABLE mv_x AS SELECT * FROM v_x)
-- pour les stocker avant de les manipuler dans Python.
--
-- Rappels : DepDel15 = 1 si le vol part avec 15 min de retard ou plus (NULL si annulé) ;
-- Cancelled = 1 si le vol est annulé ; retard_ou_annule = 1 si en retard OU annulé.
-- =====================================================================================

DROP VIEW IF EXISTS v_ponctualite_compagnie;
DROP VIEW IF EXISTS v_ponctualite_type;
DROP VIEW IF EXISTS v_composition_heure;
DROP VIEW IF EXISTS v_compagnie_saison;
DROP VIEW IF EXISTS v_domino;
DROP VIEW IF EXISTS v_domino_compagnie;
DROP VIEW IF EXISTS v_jour;
DROP VIEW IF EXISTS v_jour_semaine;
DROP VIEW IF EXISTS v_destination;
DROP VIEW IF EXISTS v_creneau;
DROP VIEW IF EXISTS v_depart_meteo;

-- Q1. Comment les compagnies diffèrent-elles ? (fusion vols × compagnies)
CREATE VIEW v_ponctualite_compagnie AS
SELECT
    v.compagnie,
    c.type_compagnie,
    c.vend_sous_la_marque,
    COUNT(*)                                                              AS nb_vols,
    ROUND(100.0 * AVG(v.retard_ou_annule), 1)                             AS pct_retard_ou_annule,
    ROUND(100.0 * AVG(CASE WHEN v.Cancelled = 0 THEN v.DepDel15 END), 1)  AS pct_retard,
    ROUND(100.0 * AVG(v.Cancelled), 1)                                    AS pct_annules,
    ROUND(100.0 * AVG(CASE WHEN v.Cancelled = 0 THEN (v.DepDelayMinutes > 45) END), 1) AS pct_grand_retard
FROM vols v
JOIN compagnies c ON c.compagnie = v.compagnie
GROUP BY v.compagnie, c.type_compagnie, c.vend_sous_la_marque;

CREATE VIEW v_ponctualite_type AS
SELECT
    c.type_compagnie,
    COUNT(*)                                          AS nb_vols,
    ROUND(100.0 * AVG(v.retard_ou_annule), 1)         AS pct_retard_ou_annule
FROM vols v
JOIN compagnies c ON c.compagnie = v.compagnie
GROUP BY c.type_compagnie;

-- Q2. À heure égale ? Standardisation directe : x̄ = Σ w_h · μ_h, avec les poids w_h de tous les vols partis
CREATE VIEW v_composition_heure AS
WITH partis AS (
    SELECT * FROM vols WHERE Cancelled = 0
),
poids AS (
    SELECT tranche_horaire, COUNT(*) * 1.0 / (SELECT COUNT(*) FROM partis) AS w
    FROM partis GROUP BY tranche_horaire
),
taux AS (
    SELECT compagnie, tranche_horaire, AVG(DepDel15) AS mu
    FROM partis GROUP BY compagnie, tranche_horaire
)
SELECT
    t.compagnie,
    ROUND(100.0 * (SELECT AVG(DepDel15) FROM partis p WHERE p.compagnie = t.compagnie), 1) AS pct_retard_brut,
    ROUND(100.0 * SUM(t.mu * p.w) / SUM(p.w), 1)                                         AS pct_retard_standardise
FROM taux t
JOIN poids p ON p.tranche_horaire = t.tranche_horaire
GROUP BY t.compagnie;

-- Q3. Le classement change-t-il avec la saison ?
CREATE VIEW v_compagnie_saison AS
SELECT compagnie, saison, COUNT(*) AS nb_vols_partis, ROUND(100.0 * AVG(DepDel15), 1) AS pct_retard
FROM vols
WHERE Cancelled = 0
GROUP BY compagnie, saison;

-- Q4. Effet domino : le départ précédent du même avion, le même jour (fonction de fenêtre LAG)
CREATE VIEW v_domino AS
WITH ordonnes AS (
    SELECT
        compagnie, tranche_horaire, Cancelled, DepDel15,
        LAG(DepDel15) OVER (PARTITION BY Tail_Number, FlightDate ORDER BY CRSDepTime) AS precedent_en_retard,
        ROW_NUMBER()  OVER (PARTITION BY Tail_Number, FlightDate ORDER BY CRSDepTime) AS rang_du_depart
    FROM vols
    WHERE Tail_Number IS NOT NULL
)
SELECT * FROM ordonnes WHERE Cancelled = 0 AND precedent_en_retard IS NOT NULL;

CREATE VIEW v_domino_compagnie AS
SELECT
    compagnie,
    SUM(precedent_en_retard = 1)                                                AS nb_cas_precedent_en_retard,
    ROUND(100.0 * AVG(CASE WHEN precedent_en_retard = 0 THEN DepDel15 END), 1)  AS pct_si_precedent_a_l_heure,
    ROUND(100.0 * AVG(CASE WHEN precedent_en_retard = 1 THEN DepDel15 END), 1)  AS pct_si_precedent_en_retard
FROM v_domino
GROUP BY compagnie
HAVING SUM(precedent_en_retard = 1) >= 30;

-- Le temps, jour par jour : fusion vols × calendrier × météo × événements
CREATE VIEW v_jour AS
WITH par_jour AS (
    SELECT
        FlightDate                                                            AS date_vol,
        COUNT(*)                                                              AS nb_vols,
        ROUND(100.0 * AVG(CASE WHEN Cancelled = 0 THEN DepDel15 END), 1)      AS pct_retard,
        ROUND(100.0 * AVG(Cancelled), 1)                                      AS pct_annules
    FROM vols GROUP BY FlightDate
),
evts AS (
    SELECT date_vol, GROUP_CONCAT(numero || ' ' || nom, ' ; ') AS evenements
    FROM evenements GROUP BY date_vol
)
SELECT
    c.date_vol, c.jour_semaine, c.saison, c.jour_ferie,
    j.nb_vols, j.pct_retard, j.pct_annules,
    m.pluie_mm, m.neige_cm, m.heures_de_pluie, m.rafales_max_kmh, m.temperature_max,
    e.evenements
FROM calendrier c
LEFT JOIN par_jour   j ON j.date_vol = c.date_vol
LEFT JOIN meteo_jour m ON m.date_vol = c.date_vol
LEFT JOIN evts       e ON e.date_vol = c.date_vol;

CREATE VIEW v_jour_semaine AS
SELECT c.num_jour_semaine, c.jour_semaine,
       COUNT(*) AS nb_vols_partis, ROUND(100.0 * AVG(v.DepDel15), 1) AS pct_retard
FROM vols v
JOIN calendrier c ON c.date_vol = v.FlightDate
WHERE v.Cancelled = 0
GROUP BY c.num_jour_semaine, c.jour_semaine;

-- Nouvelle dimension : les destinations (fusion vols × destinations)
CREATE VIEW v_destination AS
SELECT
    d.dest, d.ville, d.etat, d.distance_miles,
    COUNT(*)                                                              AS nb_vols,
    ROUND(100.0 * AVG(CASE WHEN v.Cancelled = 0 THEN v.DepDel15 END), 1)  AS pct_retard,
    ROUND(100.0 * AVG(v.Cancelled), 1)                                    AS pct_annules
FROM vols v
JOIN destinations d ON d.dest = v.Dest
GROUP BY d.dest, d.ville, d.etat, d.distance_miles
HAVING COUNT(*) >= 300;

-- Le meilleur et le pire créneau : compagnie × tranche horaire × saison (au moins 200 vols partis)
CREATE VIEW v_creneau AS
SELECT compagnie, tranche_horaire, saison,
       COUNT(*) AS nb_vols_partis, ROUND(100.0 * AVG(DepDel15), 1) AS pct_retard
FROM vols
WHERE Cancelled = 0
GROUP BY compagnie, tranche_horaire, saison
HAVING COUNT(*) >= 200;

-- Chaque départ avec la météo de son heure de départ (fusion vols × météo horaire),
-- base du modèle explicatif (data scientist)
CREATE VIEW v_depart_meteo AS
SELECT
    v.FlightDate, v.compagnie, c.type_compagnie, v.tranche_horaire, v.heure_depart_prevue,
    v.jour_semaine, v.saison, v.DepDel15, v.vol_precedent_en_retard,
    mh.pluie_mm   AS pluie_heure_mm,
    mh.rafales_kmh AS rafales_heure_kmh,
    mh.neige_cm   AS neige_heure_cm,
    mj.heures_de_pluie AS heures_de_pluie_jour
FROM vols v
JOIN compagnies c ON c.compagnie = v.compagnie
LEFT JOIN meteo_heure mh ON mh.date_vol = v.FlightDate AND mh.heure = v.heure_depart_prevue
LEFT JOIN meteo_jour  mj ON mj.date_vol = v.FlightDate
WHERE v.Cancelled = 0;

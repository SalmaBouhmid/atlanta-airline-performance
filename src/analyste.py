"""Le regard du data analyst : des résultats concrets, tirés des vues de la base de données.

Destinations, meilleurs et pires créneaux, records de l'année, effet de la météo, et retour
sur les jours que la presse n'expliquait pas (avec la météo cette fois).
"""

import pandas as pd

from base_de_donnees import lire, requete
from contexte import JOURS_NON_EXPLIQUES


def destinations(n=5):
    """Les destinations les plus et les moins en retard (au moins 300 vols)."""
    d = lire("destination").sort_values("pct_retard")
    return {"meilleures": d.head(n), "pires": d.tail(n).iloc[::-1], "nb_destinations": len(d)}


def creneaux(n=3):
    """Compagnie × tranche horaire × saison (au moins 200 vols partis) : les extrêmes."""
    c = lire("creneau").sort_values("pct_retard")
    return {"meilleurs": c.head(n), "pires": c.tail(n).iloc[::-1], "nb_creneaux": len(c)}


def meteo_par_classe():
    """% de retard selon la pluie de la journée (vue v_jour : fusion vols × météo)."""
    j = lire("jour")
    j["pluie"] = pd.cut(j["pluie_mm"], [-0.1, 0.1, 5, 20, 1000],
                        labels=["Sec (0 mm)", "Faible (< 5 mm)", "Moyenne (5-20 mm)", "Forte (> 20 mm)"])
    return j.groupby("pluie", observed=True).agg(nb_jours=("date_vol", "size"), pct_retard=("pct_retard", "mean"),
                                                  pct_annules=("pct_annules", "mean")).round(1)


def correlations_meteo():
    """Corrélation de Spearman entre la météo du jour et le retard du jour (365 jours)."""
    j = lire("jour")
    return j[["pct_retard", "pluie_mm", "heures_de_pluie", "rafales_max_kmh", "neige_cm", "temperature_max"]].corr(
        method="spearman")["pct_retard"].drop("pct_retard").round(2)


def records():
    j = lire("jour")
    vol_max = requete("SELECT FlightDate, compagnie, Dest, DepDelay FROM vols ORDER BY DepDelay DESC LIMIT 1").iloc[0]
    return {
        "jour_le_plus_en_retard": j.loc[j["pct_retard"].idxmax(), ["date_vol", "jour_semaine", "pct_retard"]].to_dict(),
        "jour_le_plus_calme": j.loc[j["pct_retard"].idxmin(), ["date_vol", "jour_semaine", "pct_retard"]].to_dict(),
        "jour_le_plus_annule": j.loc[j["pct_annules"].idxmax(), ["date_vol", "pct_annules", "neige_cm"]].to_dict(),
        "vol_le_plus_en_retard": {"date": vol_max["FlightDate"], "compagnie": vol_max["compagnie"],
                                  "destination": vol_max["Dest"], "minutes": float(vol_max["DepDelay"]),
                                  "heures": round(float(vol_max["DepDelay"]) / 60, 1)},
        "jours_feries": j[j["jour_ferie"].notna()][["date_vol", "jour_ferie", "pct_retard"]].to_dict(orient="records"),
    }


def jours_non_expliques_avec_meteo():
    """Les jours anormaux sans source trouvée : que dit la météo d'Atlanta ces jours-là ?"""
    j = lire("jour").set_index("date_vol")
    mediane = j[["pluie_mm", "heures_de_pluie", "rafales_max_kmh"]].median()
    lignes = []
    for date in JOURS_NON_EXPLIQUES:
        ligne = j.loc[date]
        lignes.append({"date": date, "jour": ligne["jour_semaine"], "pct_retard": ligne["pct_retard"],
                       "pluie_mm": ligne["pluie_mm"], "heures_de_pluie": ligne["heures_de_pluie"],
                       "rafales_max_kmh": ligne["rafales_max_kmh"], "neige_cm": ligne["neige_cm"],
                       "temperature_max": ligne["temperature_max"]})
    return pd.DataFrame(lignes), mediane


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    d = destinations()
    print(d["meilleures"][["dest", "ville", "nb_vols", "pct_retard"]].to_string(), "\n", d["pires"][["dest", "ville", "nb_vols", "pct_retard"]].to_string())
    c = creneaux()
    print(c["meilleurs"].to_string(), "\n", c["pires"].to_string())
    print(meteo_par_classe(), correlations_meteo(), sep="\n")
    print(records())
    tableau, mediane = jours_non_expliques_avec_meteo()
    print(tableau.to_string(), mediane.to_dict(), sep="\n")

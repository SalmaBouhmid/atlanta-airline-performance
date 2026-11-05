"""La conclusion de chaque compagnie, construite à partir des résultats déjà calculés (rien n'est inventé).

Pour chaque compagnie : son rang, sa position (en tête, au milieu, en queue), ce qui la distingue
(annulations, gravité), l'effet de ses horaires, sa meilleure et sa pire saison, ses crises réelles,
ce que dit le modèle à conditions égales, et une phrase de conclusion.
"""

ORDRE_SAISONS = ["Automne", "Hiver", "Printemps", "Été"]


def court(compagnie):
    return "Endeavor" if compagnie.startswith("Endeavor") else compagnie


def fr(nombre, decimales=1):
    return f"{nombre:,.{decimales}f}".replace(",", " ").replace(".", ",")


def conclusions_par_compagnie(resultats):
    """Renvoie {compagnie: {rang, position, constats: [...], conclusion}} à partir du dictionnaire des résultats."""
    cie = resultats["compagnies"]
    ind = resultats["indicateurs"]
    grand_retard_global = resultats["classes_retard"]["Grand retard (> 45 min)"]
    paires = resultats["approfondissement"]["paires_compagnies"]["non_distinctes"]
    evenements = resultats["approfondissement"]["evenements"]
    rapports = (resultats.get("equipe") or {}).get("modele", {}).get("rapports_de_cotes", {})
    contexte = (resultats.get("equipe") or {}).get("compagnies_contexte", {})
    ordre = sorted(cie, key=lambda c: cie[c]["pct_retard_ou_annule"])

    sortie = {}
    for rang, c in enumerate(ordre, 1):
        p = cie[c]
        constats = []
        if rang <= 3:
            position = "parmi les plus ponctuelles"
        elif rang >= len(ordre) - 3:
            position = "parmi les moins ponctuelles"
        else:
            position = "au milieu du classement"
        # Non départageable de…
        voisins = sorted({court(x["compagnie_b"]) if x["compagnie_a"] == c else court(x["compagnie_a"])
                          for x in paires if c in (x["compagnie_a"], x["compagnie_b"])})
        if voisins:
            constats.append("pas séparable statistiquement de " + ", ".join(voisins))
        # Annulations et gravité
        if p["pct_annules"] >= 2 * ind["pct_annules"]:
            constats.append(f"annule beaucoup ({fr(p['pct_annules'])} % contre {fr(ind['pct_annules'])} % en moyenne)")
        elif p["pct_annules"] <= ind["pct_annules"] / 2:
            constats.append(f"annule très peu ({fr(p['pct_annules'])} %)")
        if p["pct_grand_retard"] >= 1.5 * grand_retard_global:
            constats.append(f"retards graves fréquents ({fr(p['pct_grand_retard'])} % de plus de 45 min, contre {fr(grand_retard_global)} %)")
        # Effet des horaires et du calendrier (standardisation)
        for cle, quoi in [("composition_tranche_horaire", "heure égale"), ("composition_mois", "mois égal")]:
            e = resultats[cle].get(c, {}).get("ecart_points")
            if e is not None and abs(e) >= 2:
                sens = "ses horaires la flattent" if e > 0 else "ses horaires la pénalisent"
                if quoi == "mois égal":
                    sens = "son calendrier la flatte" if e > 0 else "son calendrier la pénalise"
                constats.append(f"à {quoi}, {fr(abs(e))} points {'de plus' if e > 0 else 'de moins'} : {sens}")
        # Saisons
        s = resultats["retard_par_saison_pct"].get(c)
        if s:
            meilleure, pire = min(s, key=s.get), max(s, key=s.get)
            constats.append(f"meilleure saison : {meilleure.lower()} ({fr(s[meilleure])} %), pire : {pire.lower()} ({fr(s[pire])} %)")
        # Crises réelles propres à la compagnie
        crises = [e["nom"] for e in evenements if e["compagnies"] != "toutes" and court(c) in e["compagnies"]]
        if crises:
            constats.append("crise(s) de l'année : " + ", ".join(crises))
        # Effet domino
        d = resultats["domino_par_compagnie"].get(c)
        if d:
            constats.append(f"effet domino : {fr(d['pct_si_precedent_a_l_heure'])} % → {fr(d['pct_si_precedent_en_retard'])} %")
        # Modèle, à conditions égales (référence : Delta)
        r = rapports.get(f"Compagnie : {c}")
        if c == "Delta":
            constats.append("référence du modèle")
        elif r:
            if r["ic_bas"] <= 1 <= r["ic_haut"]:
                constats.append("à conditions égales, pas différente de Delta (modèle)")
            else:
                constats.append(f"à conditions égales, cote du retard × {fr(r['rapport_de_cotes'], 2)} par rapport à Delta (modèle)")

        phrase = f"{court(c)} ({p['type'].lower()}) est {position} : {fr(p['pct_retard_ou_annule'])} % de vols retardés ou annulés, rang {rang} sur {len(ordre)}."
        if voisins:
            phrase += " Son rang exact n'est pas départageable."
        if c.startswith("Endeavor"):
            phrase += " C'est elle qui fait l'avantage des régionales."
        if r and not (r["ic_bas"] <= 1 <= r["ic_haut"]) and r["rapport_de_cotes"] > 1.5:
            phrase += " L'écart tient même à conditions égales."
        elif r and not (r["ic_bas"] <= 1 <= r["ic_haut"]) and r["rapport_de_cotes"] < 0.8:
            phrase += " Même à conditions égales, moins de retard que Delta."
        sortie[c] = {"rang": rang, "type": p["type"], "pct_retard_ou_annule": p["pct_retard_ou_annule"], "position": position,
                     "constats": constats, "contexte": contexte.get(c, {}).get("remarque", ""), "conclusion": phrase}
    return sortie

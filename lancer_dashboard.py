"""Lance le dashboard du projet (double-clic sur ce fichier, ou : python lancer_dashboard.py).

Ce lanceur remplace l'ancien fichier .bat (bloqué par les messageries, car les .bat peuvent exécuter n'importe quoi).
Il fonctionne sous Windows, Mac et Linux, avec le Python qui l'exécute :
  1. vérifie que les bibliothèques nécessaires sont installées (et propose de les installer) ;
  2. crée les données nettoyées, la base SQL et les résultats s'ils manquent (environ 5 minutes la première fois) ;
  3. ouvre le dashboard dans le navigateur.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
BIBLIOTHEQUES = ["streamlit", "pandas", "numpy", "pyarrow", "plotly", "scipy", "sklearn", "statsmodels", "matplotlib"]


def attendre_et_quitter(code=0):
    input("\nAppuyez sur Entrée pour fermer cette fenêtre…")
    sys.exit(code)


def main():
    print("=" * 64)
    print("  Dashboard : compagnies aériennes au départ d'Atlanta")
    print("=" * 64)
    print(f"Python utilisé : {sys.executable}")

    manquantes = [b for b in BIBLIOTHEQUES if importlib.util.find_spec(b) is None]
    if manquantes:
        print(f"\nBibliothèques manquantes : {', '.join(manquantes)}")
        if input("Les installer maintenant avec pip (requirements.txt) ? [o/n] ").strip().lower() != "o":
            print("Installation annulée : installez-les avec  pip install -r requirements.txt")
            attendre_et_quitter(1)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(RACINE / "requirements.txt")], check=False)

    if not (RACINE / "reports" / "resultats.json").exists() or not (RACINE / "data" / "atlanta_vols.sqlite").exists():
        print("\nPremier lancement : création des données nettoyées, de la base SQL et des résultats (environ 5 minutes)…")
        resultat = subprocess.run([sys.executable, str(RACINE / "pipeline.py")], cwd=RACINE)
        if resultat.returncode != 0:
            print("[ERREUR] La préparation des données a échoué : lisez le message ci-dessus.")
            attendre_et_quitter(1)

    print("\nLe navigateur va s'ouvrir (adresse « Local URL », en général http://localhost:8501).")
    print("Pour arrêter le dashboard : fermez cette fenêtre (ou Ctrl+C).\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(RACINE / "dashboard" / "app.py")], cwd=RACINE)
    attendre_et_quitter()


if __name__ == "__main__":
    main()

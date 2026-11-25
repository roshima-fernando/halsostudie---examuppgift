## halsostudie---examuppgift

Analys av hälsodata i Jupyter Notebook (`health_study.ipynb`).

- Python-version: 3.14.7
- Del 1 finns i branchen `del1`. Del 2 finns i branchen `del2`, där notebooken innehåller både Del 1 och Del 2.

## Projektstruktur

- `health_study.ipynb` – notebooken med analysen
- `data/health_study_dataset.csv` – datasetet
- `src/health_functions.py` – funktioner för statistik, tester och regression
- `src/health_analyzer.py` – klassen `HealthAnalyzer`
- `requirements.txt` – paket som behövs

## Kom igång

```bash
# Klona projektet och byt till branchen del2
git clone https://github.com/roshima-fernando/halsostudie---examuppgift.git
cd halsostudie---examuppgift
git checkout del2

# Skapa en virtuell miljö
python -m venv .venv

# Aktivera den virtuella miljön
# Windows (Git Bash)
source .venv/Scripts/activate

# Windows (PowerShell)
.venv\Scripts\Activate

# macOS/Linux
source .venv/bin/activate

# Installera beroenden
python -m pip install -r requirements.txt
```
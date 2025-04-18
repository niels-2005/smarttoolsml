# DVC + DagsHub Setup

Dieses Repository zeigt, wie man ein DVC-Tracking-System mit DagsHub einrichtet.

## Voraussetzungen

- Git
- DVC
- DagsHub Account
- Repository auf DagsHub erstellt

## Setup-Schritte

### 1. Repository initialisieren und verbinden

```bash
git init
git remote add origin https://dagshub.com/<username>/<repo-name>.git
```
### 2. DVC initialisieren
```bash
dvc init
```
### 3. Daten hinzufügen
```bash
mkdir data
cp <deine-datenquelle> data/data.csv
dvc add data/data.csv
```
### 4. DVC Remote konfigurieren (DagsHub als Storage)
```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<username>/<repo-name>.s3
dvc remote modify origin --local access_key_id <dein_token>
dvc remote modify origin --local secret_access_key <dein_token>
```
Hinweis: <dein_token> findest du unter Account Settings → Access Tokens auf DagsHub.

### 5. Daten hochladen
```bash
dvc push -r origin
```
### 6. Daten herunterladen
```bash
dvc pull -r origin
```
### 7. Git Änderungen committen
```bash
git add data/data.csv.dvc .gitignore dvc.yaml dvc.lock
git commit -m "Add data file with DVC tracking"
git push origin main
```

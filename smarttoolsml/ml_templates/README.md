# Machine Learning Metriken - README

In Machine Learning sind verschiedene Metriken entscheidend, um zu bewerten, wie gut ein Modell arbeitet. Hier sind die häufigsten Metriken und wann du sie verwenden solltest.

## Metriken

### 1. Accuracy
- **Beschreibung**: Misst, wie oft das Modell richtig vorhersagt, sowohl für positive als auch negative Klassen.
- **Wann verwenden**: Wenn deine Daten **ausgewogen** sind und es darauf ankommt, **allgemein richtig zu liegen**.
  
#### Anwendungsfälle:
- Handschriftenerkennung (Zahlen von 0 bis 9 kommen gleich oft vor).
- Bildklassifikation von Haustieren (Hunde und Katzen in gleicher Anzahl).
- Spam-Erkennung (gleiche Menge an Spam und Nicht-Spam-E-Mails).

---

### 2. Precision
- **Beschreibung**: Misst, wie oft die positiven Vorhersagen auch tatsächlich positiv sind.
- **Wann verwenden**: Wenn es wichtig ist, **falsch-positive Vorhersagen zu vermeiden**.
  
#### Anwendungsfälle:
- **Spam-Erkennung**: Verhindern, dass legitime E-Mails als Spam markiert werden.
- **Medizinische Tests**: Vermeiden, dass gesunde Personen fälschlicherweise als krank diagnostiziert werden.
- **Malware-Erkennung**: Verhindern, dass harmlose Programme als schädlich eingestuft werden.

---

### 3. Recall
- **Beschreibung**: Misst, wie viele der tatsächlichen positiven Fälle vom Modell richtig erkannt wurden.
- **Wann verwenden**: Wenn es wichtig ist, **alle positiven Fälle zu finden**, auch wenn es falsch-positive Vorhersagen gibt.
  
#### Anwendungsfälle:
- **Krebsdiagnose**: Alle möglichen Krebsfälle müssen erkannt werden, auch wenn manche gesunde Patienten fälschlicherweise als krank erkannt werden.
- **Betrugserkennung**: Alle betrügerischen Transaktionen erkennen, auch wenn einige legitime Transaktionen als Betrug gewertet werden.
- **Phishing-Erkennung**: Alle Phishing-E-Mails erkennen, selbst auf Kosten einiger Fehlalarme.

---

### 4. F1-Score
- **Beschreibung**: Die **harmonische Mitte** von Precision und Recall, die ein Gleichgewicht zwischen beiden misst.
- **Wann verwenden**: Wenn du ein **ausgewogenes Verhältnis** zwischen Precision und Recall brauchst, besonders bei **unausgewogenen Daten**.
  
#### Anwendungsfälle:
- **Betrugserkennung**: Du möchtest sowohl alle Betrugsfälle finden (Recall) als auch nicht zu viele Fehlalarme (Precision) haben.
- **Tumorerkennung in medizinischen Bildern**: Ein Gleichgewicht zwischen möglichst vielen erkannten Tumoren (Recall) und wenigen falsch-positiven Diagnosen (Precision).
- **Maschinendefekte erkennen**: Alle Defekte finden, aber unnötige Alarme minimieren.

---

## Fazit

- **Accuracy**: Verwenden bei **ausgewogenen Daten** und wenn **alle Vorhersagen zählen**.
- **Precision**: Verwenden, wenn **falsch-positive Fehler vermieden** werden sollen.
- **Recall**: Verwenden, wenn es wichtig ist, **alle positiven Fälle zu erkennen**.
- **F1-Score**: Verwenden bei **unausgewogenen Daten** oder wenn ein **Gleichgewicht zwischen Precision und Recall** wichtig ist.

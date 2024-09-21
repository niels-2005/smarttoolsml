# Machine Learning Metriken

In Machine Learning sind verschiedene Metriken entscheidend, um zu bewerten, wie gut ein Modell arbeitet. Hier sind die häufigsten Metriken und wann du sie verwenden solltest.

## Klassifizierungs Metriken

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

### 5. Specificity (True Negative Rate)
- **Beschreibung**: Misst den Anteil der **tatsächlichen negativen Fälle**, die korrekt als negativ erkannt werden. Es ist das Gegenstück zu Recall (der sich auf die positiven Klassen konzentriert).
- **Wann verwenden**: Wenn es wichtig ist, **falsch-positive Vorhersagen** zu minimieren und die negativen Klassen korrekt zu erkennen.
  
#### Anwendungsfälle:
- **Kriminalitätsvorhersage**: Sicherstellen, dass harmlose Personen nicht fälschlicherweise als Verdächtige identifiziert werden.
- **Medizinische Tests**: Sicherstellen, dass gesunde Personen als gesund erkannt werden.
- **Identitätsprüfung**: Sicherstellen, dass rechtmäßige Benutzer nicht fälschlicherweise als Betrüger markiert werden.

---

### 6. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- **Beschreibung**: Misst die **Trennfähigkeit des Modells**, indem der Trade-off zwischen True Positive Rate (Recall) und False Positive Rate (1 - Specificity) für alle möglichen Schwellenwerte betrachtet wird.
- **Wann verwenden**: Verwende diese Metrik, wenn du ein **probabilistisches Klassifikationsmodell** hast und die **generelle Klassifikationseffizienz über verschiedene Schwellenwerte** vergleichen möchtest.
  
#### Anwendungsfälle:
- **Kreditkartenbetrugserkennung**: Bewerten, wie gut ein Modell betrügerische Transaktionen von legitimen unterscheiden kann.
- **Krankheitsvorhersage**: Messen, wie gut das Modell kranke Patienten von gesunden trennt.
- **Phishing-Website-Erkennung**: Wie gut das Modell zwischen legitimen und schädlichen Websites unterscheidet.

---

### 7. Log-Loss (Logarithmic Loss)
- **Beschreibung**: Misst die **Unsicherheit des Modells bei probabilistischen Vorhersagen**. Es bestraft falsche Klassifizierungen und gibt zusätzlich einen Anreiz, die korrekten Klassen mit hoher Wahrscheinlichkeit vorherzusagen.
- **Wann verwenden**: Wenn du möchtest, dass das Modell **präzise Wahrscheinlichkeiten** ausgibt und nicht nur die richtige Klasse vorhersagt.
  
#### Anwendungsfälle:
- **Wahrscheinlichkeitsvorhersagen** für Wettermodelle: Das Modell soll nicht nur sagen, ob es regnet, sondern auch wie sicher es dabei ist.
- **Stimmungsanalyse**: Bewerten, wie sicher das Modell in der Klassifikation von positiver, neutraler oder negativer Stimmung ist.
- **Kundenabwanderung (Churn Prediction)**: Vorhersage, wie wahrscheinlich ein Kunde abwandern wird, anstatt nur Ja/Nein zu sagen.

---

### 8. Matthews Correlation Coefficient (MCC)
- **Beschreibung**: Ein robustes Maß, das den **Zusammenhang zwischen tatsächlichen und vorhergesagten Klassen** quantifiziert, selbst bei **unausgewogenen Klassen**. Werte liegen zwischen -1 (schlechte Vorhersage) und 1 (perfekte Vorhersage).
- **Wann verwenden**: Verwende MCC, wenn du eine **ausgewogene Bewertung der Modellleistung** für **alle Klassen** möchtest, insbesondere bei **unausgewogenen Daten**.
  
#### Anwendungsfälle:
- **DNA-Sequenzanalyse**: Klassifikation von Sequenzen, bei denen die Klassen extrem unausgewogen sind.
- **Seltene Krankheitsdiagnosen**: Bewerten, wie gut das Modell bei Krankheiten mit wenigen Fällen arbeitet.
- **Betrugserkennung**: MCC ist nützlich, um bei stark unausgewogenen Klassen zu bewerten, wie gut betrügerische Transaktionen erkannt werden.

---

### 9. Balanced Accuracy
- **Beschreibung**: Ein Mittelwert aus der **True Positive Rate (Recall)** und der **True Negative Rate (Specificity)**, um eine faire Bewertung für **unausgewogene Datensätze** zu geben.
- **Wann verwenden**: Wenn du mit **unausgewogenen Datensätzen** arbeitest und sowohl die **positiven als auch die negativen Klassen** gleich gewichtet bewerten möchtest.
  
#### Anwendungsfälle:
- **Krankheitserkennung**: Bei seltenen Krankheiten, bei denen sowohl die Erkennung der Krankheit als auch die korrekte Identifizierung gesunder Personen wichtig ist.
- **Kreditrisikomodelle**: Klassifikation von zahlungsunfähigen und zahlungsfähigen Kreditnehmern bei unausgewogenen Daten.
- **Seltene Anomalien**: Erkennen von seltenen Ereignissen in großen, überwiegend normalen Datensätzen (z.B. Maschinenausfälle).

---

### 10. Hamming Loss
- **Beschreibung**: Misst den **Anteil der falsch klassifizierten Instanzen**, d.h. wie viele Vorhersagen nicht mit den tatsächlichen Werten übereinstimmen.
- **Wann verwenden**: Bei **multilabel Klassifikation**, wo es mehrere Klassen pro Instanz gibt, die korrekt vorhergesagt werden müssen.
  
#### Anwendungsfälle:
- **Textklassifikation**: Mehrere Kategorien für Dokumente gleichzeitig vorhersagen, z.B. Nachrichtenkategorien.
- **Bildtagging**: Bilder gleichzeitig in mehrere Kategorien einordnen, wie "Landschaft", "Personen", "Tiere".
- **Musikgenre-Vorhersage**: Ein Song kann mehreren Genres zugeordnet werden, und Hamming Loss misst, wie viele dieser Vorhersagen falsch waren.

---

## Fazit

- **Accuracy**: Verwenden bei **ausgewogenen Daten** und wenn **alle Vorhersagen zählen**.
- **Precision**: Verwenden, wenn **falsch-positive Fehler vermieden** werden sollen.
- **Recall**: Verwenden, wenn es wichtig ist, **alle positiven Fälle zu erkennen**.
- **F1-Score**: Verwenden bei **unausgewogenen Daten** oder wenn ein **Gleichgewicht zwischen Precision und Recall** wichtig ist.
- **Specificity**: Verwenden, wenn es wichtig ist, **falsch-positive Fehler** zu vermeiden und die negativen Klassen korrekt zu erkennen.
- **ROC-AUC**: Verwenden, wenn du **verschiedene Modelle** bewerten möchtest, die Wahrscheinlichkeiten vorhersagen.
- **Log-Loss**: Verwenden, wenn du probabilistische Vorhersagen bewerten und die **Unsicherheit** des Modells messen möchtest.
- **MCC**: Verwenden bei **unausgewogenen Daten**, wenn du eine **ausgewogene Bewertung** der Modellleistung benötigst.
- **Balanced Accuracy**: Verwenden bei **unausgewogenen Daten**, um sowohl positive als auch negative Klassen fair zu bewerten.
- **Hamming Loss**: Verwenden bei **multilabel Klassifikation**, um den Anteil falsch klassifizierter Instanzen zu messen.
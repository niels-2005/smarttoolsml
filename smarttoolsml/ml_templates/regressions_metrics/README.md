# Regression Metriken - README

In Regression werden Modelle verwendet, um kontinuierliche Werte vorherzusagen. Hier sind die gängigsten Metriken zur Bewertung von Regressionsmodellen, wann du sie verwenden solltest und typische Anwendungsfälle.

## Metriken

### 1. Mean Squared Error (MSE)
- **Beschreibung**: Misst den **durchschnittlichen quadratischen Fehler** zwischen den vorhergesagten und tatsächlichen Werten. Große Fehler werden stärker gewichtet, da sie quadriert werden.
- **Wann verwenden**: Wenn es wichtig ist, **größere Fehler stärker zu bestrafen** und du eine Metrik suchst, die größere Abweichungen stärker in den Fokus rückt.
  
#### Anwendungsfälle:
- **Vorhersage von Hauspreisen**: Wenn große Vorhersagefehler bei sehr teuren Häusern besonders problematisch sind.
- **Temperaturvorhersage**: Kleine Vorhersagefehler sind okay, aber große Fehler sollten stärker gewichtet werden.
- **Finanzprognosen**: Vorhersage von Aktienkursen, bei denen große Abweichungen starke Auswirkungen haben können.

---

### 2. Mean Absolute Error (MAE)
- **Beschreibung**: Misst den **durchschnittlichen absoluten Fehler** zwischen den Vorhersagen und den tatsächlichen Werten. Alle Fehler werden gleich behandelt, ohne sie zu quadrieren.
- **Wann verwenden**: Wenn du eine **robustere Metrik** benötigst, bei der **große Fehler nicht übermäßig bestraft** werden, und du einen intuitiveren Mittelwert des Fehlers möchtest.
  
#### Anwendungsfälle:
- **Vorhersage des Einkommens**: Hier könnten große Abweichungen toleriert werden, solange der mittlere Fehler akzeptabel ist.
- **Lagerbestandsvorhersage**: Kleinere Fehler in der Vorhersage sind weniger kritisch.
- **Energieverbrauchsprognosen**: Für den Energieverbrauch ist der durchschnittliche Fehler oft wichtiger als seltene große Abweichungen.

---

### 3. Root Mean Squared Error (RMSE)
- **Beschreibung**: Die **Wurzel des MSE**, um die Fehler in der gleichen Einheit wie die Zielvariable zu haben. Wie MSE bestraft auch RMSE größere Fehler stärker.
- **Wann verwenden**: Wenn du die **Einheiten des Fehlers** verstehen möchtest (gleiche Einheit wie das zu erwartende Ergebnis) und **große Fehler stark ins Gewicht fallen** sollen.
  
#### Anwendungsfälle:
- **Wettervorhersage**: Die Fehlergröße sollte in der gleichen Einheit wie die Temperatur gemessen werden.
- **Prognose von Verkaufszahlen**: Die Einheitlichkeit der Fehlermessung ist hier nützlich, um Geschäftsentscheidungen zu treffen.
- **Verkehrsflussvorhersage**: Abweichungen in der Vorhersage von Fahrzeiten sollten in Minuten oder Stunden gemessen werden.

---

### 4. R² (Bestimmtheitsmaß)
- **Beschreibung**: Misst den Anteil der Varianz der Zielvariable, der durch das Modell erklärt wird. Werte liegen zwischen 0 und 1, wobei 1 bedeutet, dass das Modell alle Varianz perfekt erklärt.
- **Wann verwenden**: Wenn du wissen möchtest, **wie gut dein Modell die Daten erklärt** und die Leistung im Vergleich zu einem einfachen Durchschnittsmodell bewerten möchtest.
  
#### Anwendungsfälle:
- **Modellierung von Wohnungspreisen**: Wie gut erklärt das Modell die Variation der Preise?
- **Vorhersage der Kundenzufriedenheit**: Zeigt, wie stark das Modell die Streuung in den Zufriedenheitswerten erfasst.
- **Verkaufsvorhersage**: Gibt einen allgemeinen Überblick darüber, wie gut das Modell den Umsatz vorhersagen kann.

---

### 5. Mean Squared Logarithmic Error (MSLE)
- **Beschreibung**: Misst den quadratischen Fehler, aber **logarithmiert die Werte** vorher. Dies hilft, relative Unterschiede zu betonen und extreme Werte zu dämpfen.
- **Wann verwenden**: Wenn es **wichtig ist, relative Fehler** zu betonen und **extrem große Fehler** weniger stark zu bestrafen, besonders wenn die Zielwerte variieren und stark skaliert sind.
  
#### Anwendungsfälle:
- **Wachstumsprognosen für Startups**: Extreme Werte sind weniger relevant, aber prozentuale Abweichungen bei kleineren Werten sind wichtiger.
- **Vorhersage von Bevölkerungswachstum**: Relativ kleinere Wachstumsfehler sollten stärker ins Gewicht fallen als sehr große Fehler.
- **Vorhersage von Krankheitserregern**: Bei exponentiellem Wachstum sind relative Fehler wichtiger.

---

### 6. Huber Loss
- **Beschreibung**: Mischt zwischen **MSE und MAE**, um robuste Ergebnisse zu liefern. Kleine Fehler werden wie bei MSE behandelt, aber große Fehler wie bei MAE, um den Einfluss von Ausreißern zu verringern.
- **Wann verwenden**: Wenn du eine **robustere Metrik** suchst, die **große Fehler nicht so stark** wie MSE gewichtet, aber dennoch Fehler gut misst.
  
#### Anwendungsfälle:
- **Vorhersage von Aktienkursen**: Ausreißer bei extremen Marktbewegungen sollen nicht die gesamte Prognose verzerren.
- **Verkehrsflussvorhersage**: Große Abweichungen bei Ausreißern (z.B. bei extremen Verkehrsereignissen) sollen abgefedert werden.
- **Vorhersage des Stromverbrauchs**: Gelegentliche starke Schwankungen im Stromverbrauch sollten das Ergebnis nicht übermäßig beeinflussen.

---

## Fazit

- **MSE**: Verwenden, wenn große Fehler stark bestraft werden sollen.
- **MAE**: Verwenden, wenn alle Fehler gleich wichtig sind.
- **RMSE**: Verwenden, wenn du Fehler in der gleichen Einheit wie die Vorhersagen verstehen möchtest.
- **R²**: Verwenden, um zu sehen, wie gut das Modell die Varianz der Daten erklärt.
- **MSLE**: Verwenden, wenn relative Fehler und kleine Werte wichtiger sind.
- **Huber Loss**: Verwenden, wenn du eine robuste Metrik suchst, die sowohl kleine als auch große Fehler gut handhabt.

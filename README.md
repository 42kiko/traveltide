![TravelTide Logo](img/page1.png)

# 🏆 TravelTide Kundensegmentierung

Diese vollständig überarbeitete Version von TravelTide segmentiert die Kunden der fiktiven Reisesuchmaschine anhand ihres Buchungs‑ und Surfverhaltens, sodass personalisierte Incentives gezielt eingesetzt werden können.

## 🎯 Ziele

- **Segmentierung:** Identifikation von Nutzergruppen auf Basis von Buchungen, Interaktionen und Rabatten.
- **Personalisierung:** Ableitung von passgenauen Vorteilen (Perks) pro Segment.
- **Optimierung:** Datengetriebene Empfehlungen für Marketing‑ und Loyalitätsprogramme.

## 🚀 Methodik

1. **Datenaufbereitung:** Das Rohdataset (`data/raw/base-data.csv`) wird eingelesen und mithilfe der Funktion `get_base()` aus `src/setup.py` vorverarbeitet. Datumsfelder werden in `datetime` konvertiert.
2. **Feature Engineering:** Mit `engineer_features()` werden die Einzelereignisse pro Nutzer aggregiert und demografische Merkmale, Buchungszahlen, Sitzungsdauer, Rabatte u. v. m. erzeugt.
3. **Dimensionalitätsreduktion (PCA):** Die numerischen Features werden skaliert und über eine Hauptkomponentenanalyse reduziert, sodass 95 % der Varianz erhalten bleiben.
4. **Clustering (KMeans):** Die optimale Anzahl von Clustern wird mittels Silhouette‑Score ermittelt. In dieser Analyse ergeben sich drei Segmente.
5. **Analyse & Persona‑Definition:** Für jedes Cluster werden Kennzahlen wie durchschnittliche Buchungen, Umsatzanteil und Stornoquote berechnet, daraus Personas abgeleitet und passende Perks vorgeschlagen.

## 👥 Cluster‑Personas & Perks

| **Cluster** | **Persona‑Name**          | **Charakteristik**                                                      | **Vorgeschlagener Perk**               |
|------------:|---------------------------|-------------------------------------------------------------------------|----------------------------------------|
| **0**       | **Vielflieger**           | Viele Flüge, moderater Rabattanteil, sehr niedrige Stornoquote, hoher Umsatzanteil | ✈️  Kostenlose Sitzplatz‑ oder Lounge‑Upgrades |
| **1**       | **Premium‑Stornierer**     | Viele Flüge und Hotels, hoher Umsatzanteil, hohe Stornoquote, nutzt Rabatte aktiv   | 🛡️  Flexible Buchungen ohne Stornogebühren |
| **2**       | **Gelegenheits‑Reisende**  | Wenige Buchungen, geringer Umsatzanteil, kaum Stornos, geringer Rabattanteil        | 🎁  Willkommensrabatt für nächste Buchung |

## 📊 Wichtige Kennzahlen

| Cluster | Nutzer | Ø Flüge | Ø Hotels | Ø Umsatz (USD) | Umsatzanteil (%) | Stornoquote (%) | Ø Rabatt (USD) |
|-------:|-------:|-------:|---------:|--------------:|-----------------:|----------------:|---------------:|
| 0      | 2 680  |   3.45 |     0.00 |       1 554.30 |            55.9 |            0.02 |           0.16 |
| 1      |   595  |   3.57 |     0.00 |       3 441.70 |            27.5 |           16.79 |           0.28 |
| 2      | 2 723  |   1.05 |     0.00 |         457.11 |            16.7 |            0.01 |           0.17 |

*Hinweis:* Da im verfügbaren Datensatz keine Hotel‑Umsätze enthalten sind, bezieht sich der Umsatz ausschließlich auf Flugbuchungen. Die Stornoquote ist der Durchschnitt der individuellen Stornoquoten pro Nutzer.

## 📈 Analyse & Nutzung

Das Jupyter‑Notebook unter `notebooks/analysis.ipynb` führt die gesamte Analyse Schritt für Schritt aus: Daten laden, Features aggregieren, PCA, Clustering und Auswertung. Über die Python‑Module im Ordner `src/` kann die Logik in eigenen Projekten wiederverwendet werden.

### Installation

Um das Projekt lokal auszuführen, sollten zunächst die benötigten Python‑Pakete installiert werden. Wechsele dazu ins Projektverzeichnis und führe folgenden Befehl aus:

```bash
pip install -r requirements.txt
```

Danach kann das Notebook geöffnet oder die Module direkt in eigenen Skripten importiert werden.

## 📑 Bericht

Im Ordner `report/` befindet sich eine CSV-Datei mit der Cluster‑Zusammenfassung (`cluster_summary.csv`). Die beigefügte PDF‑Präsentation fasst die Ergebnisse graphisch zusammen und entspricht inhaltlich den hier berechneten Clustern.

---

👤 **Autor:** 42KIKO (Refactoring durch ChatGPT)

📅 **Datum:** 3. Oktober 2025
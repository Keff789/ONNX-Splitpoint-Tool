# v2.80.4 – Qualitätsbefunde und Abnahmegrenzen

Stand: 12. September 2026. Diese Version repariert Softwareübergänge und behält die vorab festgelegte Modellverarbeitung und Qualitätsmethode bei. Sie soll keinen bestehenden Quality-FAIL durch Umbenennung, neue Grenzwerte oder eine günstiger gewählte Teilmenge in PASS verwandeln.

## Vorhandene Hailo8-Diagnose

Die tatsächliche Q2-Terminalausgabe zu `hailo8_mobilenet_runtime_8FehkBqO` meldet:

| Variante, gleiche 16 IDs | Top-1 | Top-5 |
|---|---:|---:|
| Original-ONNX | 13/16 | 14/16 |
| Compiler-ONNX | 13/16 | 14/16 |
| CPU-HEF auf Hailo8 | 8/16 | 14/16 |
| Privates GPU-HEF auf Hailo8 | 10/16 | 15/16 |

G3 ist technisch PASS; die Quality des privaten GPU-HEFs ist nicht gegen den Kampagnengrenzwert ausgewertet. CPU-/GPU-HEFs haben unterschiedliche konkrete Identitäten. Kein automatischer Tausch, keine Veröffentlichung des privaten GPU-HEFs in den Produktivcache und keine Übertragung einer fremden Qualitätsbewertung.

Der Collector meldet auf dieser Probe Original-/Compiler-ONNX numerisch gleich und tatsächlich erfasste FLOAT32-VStreaminputs gleich. Input-QuantInfo ist gleich (`qp_scale=0.01872340589761734`, `qp_zp=114`). **Interne UINT8-Puffer wurden nicht beobachtet.** Full-Precision-/Quantized-HAR-Emulation sind `not_available`; damit ist keine allgemeine interne Verlustursache nachgewiesen.

Der Plan nennt etwa 520,2 s für den alten CPU- und 528,8 s für den privaten GPU-Build. Daraus folgt kein Bauzeitgewinn. Die Q2-Reportfixture ist ausdrücklich aus der Terminalausgabe abgeleitet. Sie ersetzt weder Q1-Originaldateien noch Roharrays oder einen unabhängigen Recount gegen die Original-Labels; Verfügbarkeit und bytegebundene Q1/Q3-Prüfung stehen im Quellenindex des Lieferpakets.

Die historische Hailo10-Probe war ein anderer Test: beide HEFs 12/16 Top-1 und 15/16 Top-5. Diese Werte gehören nicht in die Hailo8-Tabelle. Initiale `plan.json`-Felder wie `runtime_status=not_run` bleiben unverändert als Planung erhalten; der terminale Vergleich ist die Ergebnisquelle.

## Abgeschlossene größere Qualitätsbefunde

Der .4-Plan benennt MobileNet-CPU-Referenz 73,70 % Top-1, Hailo8 Full 60,62 %, Hailo10H Full 59,42 % und DeepX Mean/Std Full 71,72 % auf je 5.000 Bildern. Diese bisherigen Full-Artefakte sind als FAIL ausgewertet. Die Planwerte werden nicht als neu gemessene .4-Ergebnisse ausgegeben; ein unabhängiger Q5-Originalabgleich wird nur bei tatsächlich verfügbarer Originalquelle behauptet. Die private H8-Fixed16-Probe ersetzt diese B5000-Befunde nicht.

Der hier vollständig nachgesammelte YOLO11l/Hailo10H-Lauf ist unter seinem zentralen Vertrag abgeschlossen:

| Variante | AP50:95 (0–100) | Δ zur CPU | Quality |
|---|---:|---:|---|
| CPU | 46,4781 | — | Referenz |
| TensorRT Full | 46,4609 | −0,0172 pp | PASS |
| Hailo10H Full | 45,0829 | −1,3953 pp | FAIL |
| Hailo10H → TRT, b062 | 45,3656 | −1,1125 pp | FAIL |

Vollständige Vorhersagen auf denselben 5.000 IDs und Fingerprints wurden geprüft, AP wurde mit dem bestehenden Evaluator reproduziert. Die Hailo-Punktentscheidungen verfehlen bereits die unveränderte 1-pp-Marge; der dokumentierte frühe FAIL ohne Bootstrap ist abgeschlossen und erzeugt kein fingiertes CI. Der ursprüngliche zusätzliche offizielle COCO-Bericht war nicht verfügbar; eine Ergänzung kann vorhandene Vorhersagen verwenden, ohne erneuten Build oder Inferenz.

Historischer Abgleich: YOLOv7/Hailo8 Opt1 41,478 AP, Opt2 laut damaliger KB 40,984 AP. Der vollständige Opt2-Rohbericht wurde im Results-Archiv nicht gefunden. Die bestehende Projektpolicy bleibt **balanced / Opt1 / B500 / Batch8**; der zwischenzeitliche Opt2/B1024- und Opt3-Vorschlag wurde verworfen. Es wird kein pauschaler Optimierungssweep ausgeliefert.

## Was die neue Software abnimmt

- Gespeicherte H8-Manifestauswahl bleibt im GUI-/Profil-Roundtrip erhalten: nicht gesetzt lässt explizites Environment-Opt-in zu, explizit leer bedeutet kein Overlay.
- CPU prüft kein Overlay. Ein gültiger Cache-HIT benötigt keine GPU-/SDK-/Assemblerprüfung; der fehlende Kontext wird erst bei einem benötigten Kaltbuild relevant.
- Cancel behält fertige Ergebnisse; Service-Closed gilt nur mit belegtem passenden Run-Cancel als Abbruchfolge. Dokumentiertes Q5-Soll: 123 terminal, 63 ausgewertet (43 PASS / 18 FAIL / 2 INCONCLUSIVE), 60 abgebrochen.
- Debuglimits: zentrale Requests 512 MiB gesamt / 64 MiB pro Datei, freigegebene strukturierte Ergebnisse/Indizes 256 MiB pro Datei. Pfad-, Symlink-, Hash- und atomare Publikationsregeln bleiben bestehen.

Neue Softwaretests sind keine neuen H8-Gerätetests. Die Zielabnahme startet aus der frisch installierten .4-Version: vorhandene Artefakte in frischen Prozessen wiederverwenden und einen begrenzten normalen CLS-/DET-Workflow bis Referenz, Consumer, Native und Bericht ausführen. Ein echter H8-Cold-Build ist nur bei weiterhin benötigtem bestätigtem MISS zulässig; kein Force oder Löschen eines warmen Artefakts für ein Testhäkchen.

Native-Energie bleibt separat mit physischem Scope **FS**, Fenster **command**, Dauer und Replikaten. **1 s × 3** Screening ist weder **30 s × 3** noch **60 s × 3**; Final Quality mit 5.000 Bildern/5.000 angeforderten Bootstraps ändert kein Energiezeitfenster. Hailo10/YOLO26-Endpunktprobleme und passende Jetson-Power-/Clock-/TPC-Belege bleiben eigene begründete Diagnosen, keine automatische Numerik- oder Systemkonfigurationsänderung in diesem Release.

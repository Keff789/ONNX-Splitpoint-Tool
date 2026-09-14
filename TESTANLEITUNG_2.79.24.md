# Testanleitung 2.79.24

Build-ID: `v2.79.24-targeted-run-repairs`

Basis ist 2.79.23 und der reale Run `complete_set_20260905_164805`.
Diese Version korrigiert konkrete Ablauf- und Darstellungsfehler. Sie erklärt
vorhandene Accuracyverluste nicht für behoben. Keine zusätzlichen Datenbanken,
Cachehierarchien, Modellumbauten oder gelockerten Quality-Grenzen.

## 1. Installation und Offline-Abnahme

Tool schließen, vollständiges Lieferpaket nach `~/Downloads` herunterladen:

```bash
cd ~/Downloads
unzip -n ONNX-Splitpoint-Tool_v2.79.24_COMPLETE_DELIVERY_BUNDLE.zip
cd ONNX-Splitpoint-Tool_v2.79.24_COMPLETE_DELIVERY_BUNDLE
bash install_v27924_and_collect_acceptance.sh
```

Das Skript erwartet die bestehende Installation `~/ONNX-Splitpoint-Tool` mit
`.venv`. Einen abweichenden vorhandenen Installationspfad mit `TOOL=/pfad`
voranstellen. Es prüft das exakte Quell-ZIP, aktualisiert die Quellen über den
bisherigen Updater und führt die Offline-Abnahme aus. Eigene Profile,
tool-lokale `artifact_store`/`build_evidence` und die Umgebung bleiben erhalten.
Zentrale Bestände unter `~/.onnx_splitpoint_tool` liegen außerhalb des Updates.
Es startet keine realen Compiler, SSH-Messungen oder EvaluationRuns.

Erwartet: `INSTALL_ACCEPTANCE=PASS`. Das ausgegebene `EVIDENCE_ZIP` enthält
den vollständigen Installations- und Testverlauf. Bei Fehler dieses ZIP
zurückschicken; die Installation nicht durch Umgehen der Prüfungen erzwingen.

Die Offline-Abnahme kann später wiederholt werden:

```bash
cd ~/ONNX-Splitpoint-Tool
bash scripts/run_v27924_small_acceptance.sh
```

## 2. Kleine Geräteabnahmen in dieser Reihenfolge

Die Abnahmeprofile im Lieferpaket begrenzen Modell-/Boundaryauswahl und halten
die bestehende 500-Bilder-Quality-Policy bei. Sie sind Kontrollläufe, keine
finale Vergleichskampagne. Die ausgewählten vorhandenen Artefakte zunächst im
warmen Preflight prüfen. Ein fehlendes Artefakt ist ein sichtbarer Stopp,
keine Aufforderung zu einem unbemerkten Cold Build.

| Abnahme | Umfang | Technisches Bestehenskriterium |
|---|---|---|
| A | MobileNet b027, H8/H10 plus jeweilige Full- und TRT-Full-Baseline | Full-HEFs erreichbar, Receipt-Promotion gültig, vollständige drei Wiederholungen, Abschlussindex ohne Symlink-Ausnahme |
| B | YOLO11l b003, H8/H10 plus Full-Baselines | H8-Detection-Abschluss inklusive gültigem Oracle-Artefakt, echte abgeschlossene Frames und konsistente FPS; Accuracy getrennt bewerten |
| C | ResNet50 b060 und RegNet b052 auf den drei Hosts | Gebundene Artefakte erreichbar und bis zum Ende erhalten; im Workflow `RETENTION_DEFERRED` statt Namespace-Eviction |

Die volle Sieben-Modell-Retentionabfolge wird offline mit vorhandenen Dateien
reproduziert. C prüft auf Hardware die konkrete Erreichbarkeit. Da der alte
Run die ResNet-Namespaces bereits gelöscht hat, kann C korrekt im warmen
Preflight stoppen. Erst dann genau die fehlenden Artefakte inventarisieren
und gegebenenfalls gezielt wiederherstellen oder neu bauen. Eine Änderung
am Pfad-/Hashgate stellt gelöschte Dateien nicht wieder her.

Für die zwei späten YOLO26-Boundaries zusätzlich kurz lesend prüfen:
`yolo26m/b398/H8` und `yolo26s/b364/H8` müssen den passenden gespeicherten
`COMPILE_INFEASIBLE`-Befund zeigen und den Compiler überspringen. Ein
`TRANSIENT_INFRASTRUCTURE`-Befund darf keinen Split dauerhaft sperren. Die
H10-Pfade sind davon getrennt; ihre Layoutkorrektur benötigt einen eigenen
kurzen Lauf mit vorhandenem HEF und exakt passendem Part2-Vertrag.
Die korrigierte Rank-3-Bridge kann eine neue TensorRT-Part2-Engine benötigen.
Das muss als gezielter TRT-Cold-Build erscheinen; der vorhandene Hailo-HEF
soll dabei wiederverwendet werden.

## 3. Quality und Energie getrennt abnehmen

- Quality bleibt auf denselben 500 Bildern und derselben CPU-Referenz.
  Fehlende Ergebnisse, `fail` und `inconclusive` bleiben sichtbar. Ein
  funktionierender Runner kann weiterhin eine ungeeignete quantisierte
  Variante messen. Insbesondere MobileNet b135 und YOLO11/26 nicht allein
  anhand gültiger FPS freigeben.
- Für Energie 30 Sekunden Hotloop und drei Wiederholungen verwenden;
  reguläre Roh-Parquets im Debug-Pack einschließen. Einen repräsentativen
  Fall zusätzlich mit 60 Sekunden prüfen. Start-/Ladeaufwand bleibt im
  vorhandenen Command-Fenster enthalten und wird nicht rechnerisch versteckt.
- Prüfen: FS-Kalibrierung angewandt, gültige Tracefenster, echte Abschlusszähler,
  keine verlorenen Samples in ausgewählten Wiederholungen; `P × t = E` und
  `E / tatsächliche Frames = J/Frame`. Keine 1-s-Werte als Dauerbetriebsenergie
  interpretieren. Für absolute Leistungsgenauigkeit bleibt eine externe
  Referenzmessung erforderlich.
- Alle Jetsons vor dem Vergleich auf dieselbe bewusst gewählte CPU/GPU-
  Taktpolicy bringen. Im untersuchten Run waren H8/H10 fest getaktet und der
  DeepX-Host dynamisch. Das Tool schaltet diese Einstellungen nicht automatisch.
- DeepX-Cold-Builds erst nach separater Prüfung der Compilerumgebung zulassen:
  die protokollierte sm_61-GPU wird vom verwendeten Torch-Build nicht unterstützt.
  Bestehende kompatible DXNNs können weiterhin genutzt werden.

## 4. Nachtlauf

Erst nach erfolgreicher technischer Abnahme mit einem kleinen Multimodelllauf
beginnen. Unaufgelöste Quality-Fehler bleiben Diagnosefälle; sie liefern keine
finale wissenschaftliche Freigabe. Der Status der Full-TRT-Referenz steht nun
separat von der globalen Quality-Entscheidung. Fehlende erforderliche Zellen
dürfen kein globales PASS ergeben.

```text
REAL_HARDWARE_ACCEPTANCE=NOT_RUN
REAL_ACCURACY_RECOVERY=NOT_RUN
REAL_30S_60S_ENERGY_COMPARISON=NOT_RUN
REAL_MULTI_MODEL_NIGHT_RUN=NOT_RUN
```

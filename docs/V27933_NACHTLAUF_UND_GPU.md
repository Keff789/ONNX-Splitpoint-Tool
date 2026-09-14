# v2.79.33: nächster Vorlauf und getrennte GPU-Diagnose

Stand: 9. September 2026. Dieser Ablauf dient der Artefaktvorbereitung und
Integration. Er ist kein Hold-out-, B5000-, Energie- oder Finalexperiment.

## Vor dem nächsten normalen Workflow

Den laufenden Workflow zuerst vollständig beenden. Alte Einstellungsdialoge
schließen und die aktualisierte GUI frisch öffnen. Historische Run-YAMLs und
bestehende Compilerartefakte bleiben unverändert.

1. Unter **Tool Config → Run modes → final → Edit selected…** Hailo Force und
   DeepX Force bewusst ausschalten, speichern und frisch laden. Kein Reset der
   Modusdefaults. Die bisherige Registry enthält an dieser Stelle echte `true`
   Werte; das Update korrigiert keine persönlichen Einstellungen automatisch.
2. Im tatsächlich verwendeten Evaluationsprofil explizit
   `deepx_build.classification_preprocessing: imagenet_mean_std` auswählen.
   Diese Profileinstellung hat eine andere Quelle als die zentralen Forcewerte.
3. In der Startsummary die effektiven Werte prüfen: beide Forcewerte AUS,
   `reuse_and_build_missing`, Hailo B500/Opt1/Batch8, DeepX B500/EMA/Opt0 sowie
   Hailo `cache_integrity: relaxed`. Mean/Std und seine Quelle müssen sichtbar
   sein. Keine gleichzeitige Änderung der Recipe, Splitpunkte oder Parallelität.
4. Die bereits vereinbarte YOLOv7-Modellquelle und die gewählten Splitpunkte
   anhand des neuen Effektivprofils prüfen. Kein alternatives Modell allein
   wegen eines ähnlichen Dateinamens einsetzen.
5. Zunächst einen kleinen normalen Lauf mit einem bereits kompatiblen Artefakt
   ausführen. Den gleichen Request ein zweites Mal ausführen und den tatsächlich
   verwendeten Vertrag vergleichen. Erst danach den größeren Nachtlauf starten.

Ein Resume bleibt an seinen alten Profilsnapshot gebunden. Enthält dieser Force,
ist eine neue bewusste Bestätigung erforderlich. Für einen Vorlauf mit geändertem
Force-/MeanStd-Vertrag einen neuen Run anlegen; den alten Snapshot nicht umschreiben.

## Was einen Wiederverwendungsnachweis ausmacht

Eine Preflight-Zeile `HIT` genügt nicht. Im tatsächlichen Buildpfad müssen ein
positiver Receipt-/Artefaktcheck und Wiederverwendung protokolliert sein. Für
DeepX Full sind dies beispielsweise `deepx_build_status: ready_reused` und
`deepx_cache_outcome: HIT` im `deepx_artifact_status.json` sowie die gebundene
Cache-Receipt. Bei Hailo bleiben HEF und Receipt an denselben Buildvertrag gebunden;
es darf kein Übersetzen/Optimieren/Kompilieren beziehungsweise DFC-Kindprozess
gestartet werden. Vorher und nachher die vorhandenen Receipt- und Artefaktmetadaten
vergleichen, keine neuen Cacheidentitäten einführen.

Ein bisher nur für `current_scale_only` vorhandenes DeepX-Klassifikationsartefakt
erfüllt den neuen Mean/Std-Vertrag nicht. Der erste entsprechende Mean/Std-Build
ist ein erwarteter Neubau, ohne die Hailo-Integrität auf strict umzustellen.
Eine exakte gespeicherte `COMPILE_INFEASIBLE`-Entscheidung muss als bekannte
negative Evidence ohne erneuten Compilerdispatch erscheinen. Andere gültige
Modelle/Backends bleiben ausführbar. Ein geänderter Quell-/Recipe-/Endpointvertrag
ist kein identischer negativer Request.

Die Release-Regression `test_v27933_runtime_reuse.py` prüft echte lokale
Cache-, Manifest-, Receipt- und Buildpfade mit temporären synthetischen Artefakten.
Nur die Vendor-Kompilierung wird simuliert. Sie deckt zwei identische Hailo- und
DeepX-Anfragen ohne Compilerdispatch, Mean/Std als getrennten Neubau sowie negative
Evidence mit unabhängigem Backend ab. Ein neuer Smartmirror2-Hardwarelauf gegen
seine bestehenden HEF/DXNN-Dateien bleibt davon getrennt und muss durch dessen
Laufprotokoll belegt werden.

## Retention: sieben Namespaces bewusst erhalten

Die neun gesicherten Retention-JSONs des BiggerSet-Vorlaufs zeigen jeweils
`managed_count: 7`, `max_namespaces: 6`, `retention_deferred: true` und
`removed: []`. Während dieses aktiven Workflows wurden an diesen Stellen keine
Namespaces gelöscht. Drei Zeilen enthalten zusätzlich ältere `prior_eviction`
Einträge für ResNet50. Sie belegen frühere LRU-Löschungen; sie sind keine Löschungen
dieses Vorlaufs und erklären nicht die separat belegten Force-Neubauten.

Der bestehende Code schützt alle vorhandenen Namespaces für die gesamte aktive
EvaluationWorkflow-Lease. Außerhalb eines aktiven Workflows gilt weiterhin die
bestehende Policy: aktueller/aktiver Bestand bleibt geschützt; nur eindeutig
eigene, durch Marker und Receipts gültige inaktive Namespaces kommen als
LRU-Löschkandidaten infrage. Diese Version ändert das Pruning nicht.

Für den bekannten Bestand von sieben Namespaces kann das vorhandene Softlimit
bewusst auf sieben gesetzt werden. Der Default von 20 GiB für den nicht aktuellen
Bestand bleibt erhalten. Bei zusätzlicher neuer Namespaceidentität muss die
gewählte Zahl entsprechend dem geplanten Bestand erhöht werden; sieben reserviert
keinen unbegrenzten zukünftigen Platz. Beispiel für genau diese sieben, in der
Shell, aus der anschließend der normale Workflow/GUI gestartet wird:

```bash
export ONNX_SPLITPOINT_REMOTE_TRT_CACHE_MAX_NAMESPACES=7
```

Dies wirkt im gestarteten Toolprozess und ist keine Änderung historischer
Profile. Die vorhandene physische Kapazitäts-/Inodeprüfung bleibt aktiv. Im
neuen Run `run_meta.json → trt_engine_cache.retention` und die
`[remote][cache][retention]`-Zeilen prüfen: gewähltes Limit, `protected`, `removed`,
`prior_eviction` und den jeweiligen Grund. Jede tatsächliche Löschung wird
gesondert ausgewiesen. Keine Cacheverzeichnisse oder Lockdateien manuell löschen.

## G0: vorhandene DFC-Umgebungen auf GPU rechnen lassen

G0 bleibt eine explizite Diagnose außerhalb des Nachtlaufs. Reguläre Tooljobs
zuerst beenden. Der folgende Einstieg hält für die ganze Probe denselben
Workflow-/Plattformlock exklusiv wie die produktiven Abläufe:

```bash
(
  set -e
  cd "$HOME/ONNX-Splitpoint-Tool"
  bash scripts/run_hailo_gpu_compute_v27933.sh
)
```

Die vorhandenen Standard-Venvs sind
`~/.onnx_splitpoint_tool/hailo/venv_hailo8` und
`~/.onnx_splitpoint_tool/hailo/venv_hailo10`. Bei abweichenden tatsächlich
konfigurierten Pfaden die expliziten Optionen `--venv-hailo8` und
`--venv-hailo10` verwenden. Fehlende Venvs werden als Setupfehler gemeldet.
Es wird nichts installiert und kein anderes Python ersatzweise ausgewählt.

DFC 8 und danach 10H laufen streng nacheinander. Pro Venv sind SDK-Import,
GPU-Matmul, GPU-Conv2D, XLA-Matmul/Sin/Exp und XLA-Conv2D/Sin erforderlich.
Soft-Device-Placement ist aus. Jeder Output muss ein GPU-Tensor sein, wird
synchronisiert und gegen eine unabhängige NumPy-Referenz geprüft. Maximal 180 s
pro Venv, danach begrenzte TERM/KILL-Bereinigung der eigenen Prozessgruppe.
Bei belegtem Workflowlock startet kein Kind und entsteht kein leeres Evidence-ZIP.

`CUDA_VISIBLE_DEVICES` und `ONNX_SPLITPOINT_HAILO_ALLOW_GPU=1` gelten nur für die
Diagnosekinder. Die reguläre Hailo-CPU-Policy sowie CUDA-/XLA-Pfade bleiben
unverändert. Parent-Umgebung, Pakete, Registry, Modelle und Produktionscache
werden nicht geändert. Es gibt keinen pip-/sudo-/SSH-/Toolkit-Schritt.

`GPU_SMOKE_STATUS=compute_pass` belegt ausschließlich diese synthetischen
Rechenoperationen in den protokollierten Versionen. `MODEL_BUILD=NOT_RUN` und
`MODEL_ACCEPTANCE=NOT_EVALUATED_BY_DIAGNOSTIC` bleiben ausdrücklich gesetzt.
Die mitgelieferten 19 ursprünglichen Helfertests und zusätzlichen
Integrationsprüfungen verwenden simuliertes TensorFlow; sie sind keine echte
GPU-Abnahme. Für G0 liegt im vorliegenden Bauauftrag kein reales Ergebnis-ZIP vor.

## G1: einzelner Modellbuild nach erfolgreichem G0

G1 ist ein anschließendes, separates Hardwareexperiment. G0 führt es nicht
automatisch aus. Vor einem Start sind der bestehende CPU-Build und dessen
vollständiger gebundener Eingabevertrag konkret auszuwählen. Geeigneter kleiner
Kandidat: vorhandenes MobileNet Full für Hailo 10H.

| Gegenstand | Festzuhaltender Vertrag |
|---|---|
| Eingang | vorhandene exakte ONNX-Quelle, Endpoint, Preprocessing und B500-Kalibrationsmanifest |
| Recipe | unverändert Opt1/Batch8 und dieselbe DFC-Version, kein zusätzlicher Sweep |
| Referenz | bestehende CPU-HEF und Receipt unverändert aufbewahren |
| GPU-Ausgabe | eigener Diagnoseordner außerhalb des Produktionscache; keine Veröffentlichung als Ersatz-HEF |
| Gerät | tatsächliche TensorFlow-Gerätewahl, CUDA/cuDNN/XLA-Versionen und Optimierungsphasen protokollieren |
| Laufkontrolle | vorhandener Workflow-/Plattformlock, genau ein Build, begrenzte Laufzeit und Prozessbereinigung |
| Vergleich | vorher festgelegte kleine Eingabemenge; Inputbindung, Outputs/Logits und Top-k gegen CPU-HEF und Floatreferenz |

Erfolgreiche HEF-Erzeugung, Phasenlaufzeiten und der kleine numerische Vergleich
sind getrennt zu dokumentieren. Zwischenartefakte nur gezielt in diesem
Diagnoseordner behalten. Ein geänderter Optimierungspfad ist eine Recipeänderung
und kein reiner Geschwindigkeitsvergleich. Byteidentische HEFs werden nicht
vorausgesetzt; Modellqualitätsmargen werden nicht gelockert. Ohne diese Evidence
bleibt CPU der Default. Ein G0-PASS ersetzt weder G1 noch B500/B5000-Qualität.

## Getrennt offene Qualitätsbefunde

Der MobileNet-Hailo-Verlust von ungefähr 13–14 Prozentpunkten im bestehenden
B5000-Befund bleibt ein negatives Qualitätsergebnis. Zuerst Input-/Logit-/Top-k-
Bindung der vorhandenen HEF prüfen, keine doppelte Mean/Std-Korrektur. Bestätigt
die Quantisierungsemulation den Verlust, ist ein legitimes negatives Ergebnis
möglich; kein Kalibrationssweep als Ersatz für diese Diagnose.

Hailo 10H/YOLO26 am AP6b-Übergang bleibt eine Raw-Boundary-Diagnose. Kein vermuteter
Decoderfix. Für DeepX zuerst den normalen kleinen Mean/Std-Pfad, anschließend den
vereinbarten MobileNet-Paarscope ausführen. Identische R1/R2-Artefaktbuilds ohne
geänderten Vertrag werden nicht erneut angefordert.

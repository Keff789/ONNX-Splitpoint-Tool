# Testanleitung v2.80.3

Build-ID: `v2.80.3-build-readiness-native-not-started-debug-export`.
Basis: geliefertes 2.80.2-Source mit SHA
`2f186954c27f2b28520c46cea1c86c83a200c4d7d4821e4b59bebaa7ec1eb465`.

## Installationsabnahme

GUI und laufende Jobs beenden. Als normaler Eigentümer ohne sudo:

```bash
(
  set -e
  cd -- "$HOME/Downloads"
  unzip -n ONNX-Splitpoint-Tool_v2.80.3_COMPLETE_DELIVERY_BUNDLE.zip
  cd -- ONNX-Splitpoint-Tool_v2.80.3_COMPLETE_DELIVERY_BUNDLE
  bash ./install_v2803_and_collect_acceptance.sh
  bash ./run_v2803_short_tests.sh
)
```

Erwartet: `INSTALL_ACCEPTANCE=PASS` und `SHORT_TESTS=PASS`.
Beide ausgegebenen `EVIDENCE_ZIP` aufbewahren und zur Auswertung zurückgeben.
Der Installer aktualisiert das Tool und seine Entry Points in der vorhandenen
Tool-Venv. Er installiert keine Vendor-, CUDA-, Torch- oder TensorFlowumgebung.
Profile, Registry, Kalibrierungen, vorhandene Artefakte und frühere Runs bleiben erhalten.

Alle ausgewählten Pflichtfälle müssen tatsächlich ausgeführt werden. Fehlende
Abhängigkeiten sind `environment_blocked`; Skips, Xfails und Fehler sind kein PASS.
Finale Testzahlen und JUnit-Identitäten stehen im äußeren `PRUEFBERICHT_2.80.3.md`.

## Was die Softwareabnahme belegt

Die positiven CPU-Referenzfälle nutzen den produktiven Generator, den wirklichen
Managementlauncher, einen frischen Suiteprozess und reale ONNX-Runtime-Inferenz.
Kleine synthetische Modelle und eigene Diagnosebilder sind kontrollierte
Softwareeingaben. Sie beweisen keinen Acceleratorlauf. Ein synthetischer Kandidat
ohne echte Producer-Attestation bleibt vom strengen Native-Loader abgewiesen.

Kontrollierte Grenzen sind externe Vendorcompiler, SSH/Transport,
Acceleratoraufrufe und gezielt ausgelöste Betriebssystemfehler. Die beanspruchten
Reader, Workflowstufen, Publikationen und ZIP-Operationen laufen real.
Neue physische Hardwareausführung in dieser Lieferumgebung: **NOT_RUN**.

## Kurze Zielsystemkette vor weiteren Nächten

Nach G1 zuerst MobileNetV3 und YOLO11l mit vorab festgelegten, vorhandenen Fällen
prüfen. Das ungeprüfte Hailo8-b064-Artefakt gehört nicht in die positive
Reusekontrolle. Die Auswahl wird vor neuen Qualitätswerten festgehalten.
16 feste Validierungsbilder pro Modell erhalten einen eigenen Dataset-/
Referenzvertrag. Die Kalibrierung bleibt B500; die 16er-Referenz darf nicht
als B500- oder B5000-Referenz gespeichert werden.

Den normalen Referenz-/Consumerpfad verwenden. **Kein globales
`cache_verify_only`:** dieser Modus unterdrückt die CPU-Referenzerzeugung und
ersetzt G2/G3 nicht. Mit passenden Artefakten werden null Compilerstarts erwartet.
Eine erwartete Null ist keine beobachtete Null. Unbekannte Dispatchzähler bleiben
unbekannt; einen unerwarteten MISS vor einem weiteren Build begründen.

G2 verlangt echte Classification-/Detectionreferenzen, zentrale Auswertung
und weiterhin gesperrte falsche Modell-/Source-/Datasetbindungen.
G3 verlangt zusätzlich bindbare Producer-/Qualityevidenz und einen exemplarisch
ausgeführten Native-/TensorRT-Full-Zweig mit Bericht. Ein korrekt berechnetes
Quality-FAIL oder INCONCLUSIVE ist kein Softwarefehler und wird nicht zu PASS.
Unabhängige ausführbare Vendor-Full-Rohmessungen bleiben erhalten.

Die ergänzende Anleitung docs/V2803_REFERENCE_AND_DIAGNOSTIC_ACCEPTANCE.md
und scripts/reference_workflow_gate_v2803.py beschreiben die konkreten
Normalworkflow- und Nachprüfschritte. Nur beobachtete Gates gelten als bestanden;
ein vorbereiteter Auftrag ist noch keine Abnahme.

## Fehlerpfad und Debugexport

Ein joblokaler Buildblocker muss neben einem unabhängigen gültigen Fall mit
eigener Identität und Originalursache erscheinen. `COMPILE_INFEASIBLE` bleibt
passende negative Compileevidenz; `TRANSIENT_INFRASTRUCTURE`, fehlende lokale
Compilerkomponenten und unbekannte Exceptions bleiben davon getrennt.
Keine neue Force-, Fallback- oder negative Cachepolicy wird eingeführt.

Die unveränderte Nacht unter 2.80.1 hat 84 logische Fälle: 21 gemessene Vendor-Full-
Fälle, 42 ungestartete Splits und 21 ungestartete TensorRT-Full-Fälle.
Negative Zeilen bleiben negativ. Echte Teilreplikate behalten ihre Statistik;
Vorabblockaden haben keine gemessenen Null-FPS, Null-Joule oder Bootstrapmediane.

Große synthetische Index-/Resultatdateien entstehen nur zur Testlaufzeit.
Ein >32-MiB-Index ist ohne Parsing nicht geprüft, nicht als korrupt nachgewiesen.
14 sicher entdeckte Referenzdiagnosen können dennoch bytegleich archiviert werden.
Originalabdeckung und kompakte Ableitungen bleiben getrennte Zähler.
Fehlende Originale aus dem alten Debug-Pack können nicht rekonstruiert werden.

## MobileNet, YOLO26 und Konfiguration

MobileNet Full/b056/b135 mit denselben vorab festgelegten Developmentbildern
vergleichen. Jeder P1-Output braucht seine eigene Float-Part1-Referenz;
unterschiedliche Boundaries werden nicht direkt auf Tensorgleichheit geprüft.
Gemeinsame Endausgaben können bildweise verglichen werden.
PARSED-/Quantized-HAR nur bei tatsächlicher Verfügbarkeit und passender Bindung
nutzen, sonst `not_available`. Genericdiagnosen sind keine zentrale Qualityfreigabe.

YOLO26-b398/b364 bleiben im negativen Bestand. Für eine P2-Kontrolle exakt
denselben erfassten Übergangstensor in TensorRT-P2 und Float-P2 verwenden.
Kein Sigmoid, Clipping, zweites NMS oder Compilerneubau auf Verdacht.

Hailo8-CPU für notwendige Kaltbuilds wird bewusst ausgewählt; Hailo10H kann GPU
behalten. Fehlende Voraussetzungen bei angeforderter Hailo8-GPU bleiben blockiert.
Passende Cachetreffer bleiben verwendbar. GPU-Präferenz und
`gpu_execution_unproven` belegen keine GPU-Ausführung.

Validierungs-/Bootstrapumfang, Native-Energiedauer, Replikate, Scope und Fenster
werden aus effektiven beziehungsweise archivierten Daten berichtet.
Die alte Nacht mit 1 s × 3 bleibt Screening; sie wird nicht zu 60 s × 3.
Ein gesondert gewählter Zwischentest mit 30 s × 3 ist ebenfalls separat zu
dokumentieren. Physischer Scope und Fenster bleiben FS/command, sofern genau
so gebunden; die Dauer allein erzeugt keine wissenschaftliche Freigabe.
Ein Profil-Label ist keine wissenschaftliche Freigabe. Rohenergie kann bei
fehlender Qualitybindung unqualifiziert erhalten bleiben; Generic-Energie bleibt aus.

MAXN_SUPER/0, TPC-Maske und EMC je Setup getrennt dokumentieren.
Keine automatische Takt-/TPC-/nvpmodel-/Treiberänderung und keine unbelegte
Kernanzahl aus einer Maske ableiten. B500, danach die getrennte 5.000-Bilder-Abnahme (B5000) und finale Energie-/
Rankingclaims benötigen ihre eigenen vorab festgelegten Verträge.

## Ergebnisnachweise

Das kompakte Software-Evidence-ZIP enthält Berichte, Logs, JSON/CSV/XML und
kleine Regressionseingaben. Modelle, HEFs, DXNNs, Engines, Bilder und Roh-Tensoren
gehören nicht ins Ergebnis-Git. Das vollständige Lieferbundle kann historische
Offline-Testtensoren enthalten; dafür nur das ausdrücklich kompakte Evidence-ZIP
als Ergebnisauswahl verwenden.

Die Dauerstufen bleiben getrennte Messaufträge: **1 s × 3** ist das vorhandene
Screening. **30 s × 3** kann ein ausdrücklich konfigurierter Diagnosezwischenschritt
sein; daraus folgt keine neue Standardpolicy. **60 s × 3** bezeichnet nur einen
entsprechend gestarteten neuen Finalauftrag im Umfang **FS/command**.
Die **5.000** Bilder umfassende Qualitätsabnahme bleibt eigenständig; Profilname
und Diagnosezwischenschritt erteilen keine wissenschaftliche Freigabe.

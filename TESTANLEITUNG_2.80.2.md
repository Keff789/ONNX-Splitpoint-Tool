# Testanleitung v2.80.2

Version `2.80.2`, Build-ID `v2.80.2-model-binding-native-quality-reporting`.
Basis ist das finale v2.80.1-Sourcearchiv mit SHA256
`716f4b1ca364b07f1294b9b735b19ac9015ccabf15d78637d4114bb75b0c5f30`.

## Installation

GUI und laufende Jobs regulär beenden. Als Eigentümer ohne sudo ausführen:

```bash
(
  set -e
  cd -- "$HOME/Downloads"
  unzip -n ONNX-Splitpoint-Tool_v2.80.2_COMPLETE_DELIVERY_BUNDLE.zip
  cd -- ONNX-Splitpoint-Tool_v2.80.2_COMPLETE_DELIVERY_BUNDLE
  bash ./install_v2802_and_collect_acceptance.sh
  bash ./run_v2802_short_tests.sh
)
```

Erwartet: `INSTALL_ACCEPTANCE=PASS`, `SHORT_TESTS=PASS` und jeweils ein
`EVIDENCE_ZIP`. Bei Fehler beide verfügbaren Nachweise behalten; das Update kann
bereits erfolgt sein. Die vorhandene Tool-Venv wird weiterverwendet. Benutzerprofile,
Registry, Kalibration, Modellcaches und bisherige Ergebnisse bleiben erhalten.
Der Installer installiert keine Vendor-/Frameworkpakete und startet keine Hardwarejobs.

## Was die Softwareabnahme prüft

Der wirkliche BenchmarkSet-Generator erzeugt kleine Classification- und Detection-
Suites mit seiner produktiven Metadatenstruktur: logische Identität in `model_name`,
Modellpfad in `model`. Echte ONNX-Runtime-CPU-Inferenz durchläuft Managementlauncher,
Suite, Referenzpublikation und zentrale Auswertung. Eine manuell ergänzte Modell-ID
ist keine Voraussetzung. Falsche Modell- und Sourcebindungen bleiben blockiert,
auch bei Wiederverwendung einer bereits vorhandenen Referenz.

Native-TensorRT-Full-Fälle ohne vorgelagerte Qualitybindung bleiben geplante,
konkret blockierte Zeilen mit null Messversuchen. Der wirkliche Blockiergrund bleibt
erhalten. Erfolgreiche Vendor-Full-Messungen werden dadurch nicht zum Transferfehler.
Die GUI liest Modellzahlen aus dem aktuellen Dashboardformat und zeigt fehlende
Zahlen als nicht verfügbar an. Echte Quality-FAIL/INCONCLUSIVE bleiben negativ.

Die verbindliche v2.80.1-Auswahl wird um die neuen Regressionen erweitert;
zeitgebundene Releaseprüfungen werden mit denselben Prüfpflichten fortgeführt.
Skips oder Xfails ergeben keine vollständige Abnahme. Fehlende ONNX-/ORT-
Testabhängigkeiten blockieren das Gate ehrlich. Finale Fallzahlen, JUnit-Dateien
und Sourcehash stehen im beigefügten Prüfbericht.

## Nächster Zielsystemtest

Neue Zielsystemausführung ist in diesem Lieferpaket `NOT_RUN`. Zuerst Installation
und Kurztests prüfen. Danach eine kleine neue Suite für je eine Classification-
und Detectionquelle mit vorab festgelegten Diagnosebildern verwenden. Die normale
Management-CPU-Referenz muss ohne erfundene physische Setup-ID entstehen und vom
zentralen Consumer gelesen werden. Der reine `cache_verify_only`-Canary ist dafür
ungeeignet, weil er CPU-Referenzjobs absichtlich deaktiviert.

Erst anschließend den begrenzten normalen Workflow mit bekannten passenden
Artefakten prüfen: zentrale Qualitybindung, mindestens ein konstruktiver Native-
Hailopfad und DeepX Full mit `imagenet_mean_std`. Case-IDs vor dem Lauf festhalten.
Erwartete und tatsächlich gestartete Compilerjobs getrennt bilanzieren.
Kein unveränderter Complete-Set-Großlauf und kein erneuter privater GPU-Kaltbuild
zur Beschaffung bereits vorliegender Fehlerevidenz.

Eine frische gepaarte B500-Qualitätsauswertung beginnt anschließend mit MobileNet.
Technisch berechnete schlechte Qualität bleibt FAIL/INCONCLUSIVE; fehlende
Referenzen sind technische Blockierungen. Die spätere 5.000-Bilder-Kampagne braucht
ihre eigene Abnahme. Vorhandene 1 s × 3 Energiescreenings bleiben Ergebnisse;
30 s × 3 und 60 s × 3 sind getrennte spätere Messverträge mit physischem
Scope FS/command. Rohenergie ohne Qualitybindung wird nicht wissenschaftlich freigegeben.

## Unveränderte Regeln und bekannte Grenzen

Produktives Force bleibt AUS/gesperrt, `relaxed` und kompatible Artefaktidentitäten
bleiben unverändert. Hailo B500/Opt1/Batch8 sowie DeepX EMA/Opt0/Mean-Std bleiben
erhalten. Keine neue Cache-/Hash-/Seal-/Registry-Art und keine Compilerparallelität.
CPU-Referenzen gehören nicht in Performance-, Energie-, Ranking- oder Paretopopulationen.

Hailo8/YOLO26 `COMPILE_INFEASIBLE` bleibt negative Recipe-Evidenz;
`TRANSIENT_INFRASTRUCTURE` bleibt davon getrennt. Die bekannten ungültigen
Hailo10H/YOLO26-Detectionendpunkte sind durch diese drei Korrekturen nicht repariert.
Kein Sigmoid/Clipping oder identischer Neubau auf Verdacht. Frühere Runartefakte
werden nicht umgeschrieben. Remote-Closure, präzise CPU-Fehlerweitergabe und kleine
Status-/Stdout-Dateien im Debugexport werden als funktionierende Regressionen erhalten.

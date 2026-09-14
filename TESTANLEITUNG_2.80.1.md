# Testanleitung v2.80.1

Release: `2.80.1`, Build-ID: `v2.80.1-cpu-reference-remote-closure-debugexport`.
Basis: unverändertes Sourcearchiv v2.80 mit SHA256
`4d25750c84f85d31c9eea03dc56ffd80ea7a317c67067b4d05daeacfe90193af`.

## Installation und Softwareabnahme

GUI und laufende Jobs regulär beenden. Als Eigentümer der Installation ohne sudo:

```bash
(
  set -e
  cd -- "$HOME/Downloads"
  unzip -n ONNX-Splitpoint-Tool_v2.80.1_COMPLETE_DELIVERY_BUNDLE.zip
  cd -- ONNX-Splitpoint-Tool_v2.80.1_COMPLETE_DELIVERY_BUNDLE
  bash ./install_v2801_and_collect_acceptance.sh
  bash ./run_v2801_short_tests.sh
)
```

Erwartet: `INSTALL_ACCEPTANCE=PASS`, `SHORT_TESTS=PASS` und jeweils ein
`EVIDENCE_ZIP`. Der Installer führt die vollständige ausgewählte Offlineabnahme
in der vorhandenen Tool-Venv aus. Er installiert keine Vendor-/Frameworkpakete.
Profile, Registry, Kalibrierung, Ergebnisse und Modellcaches werden erhalten.
Bei Fehler bitte das erzeugte ZIP zurückgeben; das Update kann bereits erfolgt sein.

Die neuen CPU-Integrationstests führen ONNX Runtime mit kleinen synthetischen
Classification- und Detectionmodellen durch den echten Management-/Suitepfad aus.
Sie sind Softwaretests; keine NPU-/GPU-Freigabe und keine B500-Modellqualität.
Die Pflichtauswahl darf keine Skips/Xfails enthalten. Fehlende benötigte Pakete
blockieren die Abnahme und gelten nicht als PASS.

## Gestufte Zielprüfung nach Installation

G1: Installations- und Kurztest-ZIPs prüfen. Neue Hardwareausführung in der lokalen
Releaseabnahme ist `NOT_RUN`.

G2: Für je eine bestehende Classification- und Detectionquelle eine kleine neue
Testsuite mit dem aktuellen Generator und festen, vor Ausführung bestimmten
Diagnosebildern erzeugen. Über `generate_management_cpu_reference()` die normale
Referenz rechnen; keine erfundene physische Setup-ID, kein Cache-only-Modus.
Der neue Status muss Referenz und Sourcebindung benennen; der zentrale Consumer
muss sie lesen können. Historische Suites/Ergebnisse nicht überschreiben.

G3: Das effektive Profil der geplanten drei Setups verwenden. Der vorhandene
Preflight ist im neuen Release ebenfalls korrigiert und überträgt zuerst alle
Paketdateien, prüft anschließend gebundene Imports. Beispiel für die vorhandene
BiggerSet-Datei (nur verwenden, wenn der Dateiname dort so lautet):

```bash
(
  set -e
  cd -- "${ONNX_SPLITPOINT_TOOL_DIR:-$HOME/ONNX-Splitpoint-Tool}"
  IMPORT_OUT="$(mktemp -d "$HOME/Downloads/v2801_native_import_XXXXXXXX")"
  .venv/bin/python -I -B scripts/preflight_v27521_native_remotes.py \
    --profile profiles/BiggerSet.yaml --timeout 600 \
    --json-out "$IMPORT_OUT/native_import_preflight.json"
)
```

Der Preflight aktualisiert den eigenen Remote-Code auf den drei konfigurierten
Setups; er startet keine Modellkompilierung, Performance- oder Energiemessung.
Ein Fehler stoppt die Freigabe des betroffenen Pfads. Ein fehlendes eigenes Modul
wird nicht per pip nachinstalliert. Das Gate benötigt gültige Setup-/Runtimewerte.

G4: Danach begrenzter normaler Workflow mit zuvor festgelegten Cases, mindestens
Hailo-Native und DeepX-Full-Mean/Std sowie Classification und Detection. Normaler
Modus, Force AUS, `relaxed`, bestehende kompatible Artefakte verwenden. Nicht
`artifact_policy=cache_verify_only`: dieser reine Reusecanary schaltet bewusst
CPU-Referenzjobs ab. Compilerdispatch und HIT/MISS/Grund sichtbar bilanzieren;
unvermutete MISS vor teurem Folgeablauf untersuchen. Kein unveränderter BiggerSet-
Großlauf und kein neuer privater GPU-Kaltbuild als Ersatz für diese Prüfung.

G5: Frische gepaarte B500-Qualitätsauswertung, MobileNet zuerst. Technisches PASS
bedeutet, dass die Auswertung lief. Echte Quality-FAIL/INCONCLUSIVE bleiben sichtbar.
Anschließend bei Bedarf begrenzter Native-Energieübergang mit vorhandener Bindung.
Die spätere 5.000-Bilder-Finalkampagne erhält eine eigene Freigabe.

## Fortgeltende Grenzen

Management-CPU-Referenzen sind semantisch und bleiben außerhalb von
Latenz/FPS/Energie/Ranking/Pareto. Generic-Full-/Composed-Kandidatengates behalten
reale Identitätspflichten. Native TensorRT Quality-FIRST bleibt ein eigener Pfad.

Force bleibt produktiv gesperrt. Hailo B500/Opt1/Batch8, DeepX EMA/Opt0 und
`imagenet_mean_std` bleiben unverändert. Keine neue Hash-/Seal-/Cacheidentität.
`COMPILE_INFEASIBLE` bleibt negative Recipe-Evidenz; `TRANSIENT_INFRASTRUCTURE`
bleibt separat. Hailo10-/YOLO26-Decoderbefunde und andere Modellfehler werden durch
diesen Patch nicht als behoben ausgegeben.

Die bestehenden Energieverträge bleiben getrennt: 1 s × 3 ist ein kurzer Smoke,
30 s × 3 eine getrennte Zwischenprüfung, 60 s × 3 der festgelegte spätere Vertrag;
physischer Scope FS/command, keine generische Energiemessung aus CPU-Referenzjobs.
Das vorhandene Vorbereitungsprofil bleibt `first_selected_model_only`,
`quality_first_trt_binding_ready=False`. HAR-Stufen ohne Daten bleiben
`not_available`; automatische HAR-Emulationssammlung wird hier nicht hinzugefügt.

Im normalen Debugexport müssen die kleinen Status-/stdout-Diagnosen direkt unter
`quality_management/references/<model>/` enthalten sein. Referenzkörper, Workspaces,
Modelle, Roharrays, Venvs und weitere Archive bleiben ausgeschlossen. Fehlende oder
zu große Diagnosen werden im Inventar sichtbar, nicht als vollständig ausgegeben.

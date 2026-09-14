# Bau- und Testvertrag v2.79.34

Build `v2.79.34-hailo-compiler-context-force-off`, Ausgangsbasis v2.79.33
`v2.79.33-force-profile-scope-safety`.

Dieser im Source enthaltene Bericht beschreibt den prüfbaren Umfang. Die nach dem
Erzeugen dieses Sourcearchivs ausgeführten Tests mit exakten Zahlen, JUnit,
Source-SHA256 und Installerprotokollen stehen im zugehörigen Delivery-Bundle.
Eine Testanforderung im Implementationsplan ist kein bestandener Test.

## Änderungen

- Produktive Force-Neubauten abweisen; mitgelieferte Hailo-/Native-Profile auf AUS.
  Der alte Force-Canary-Starter wird vor Seiteneffekten beendet. Fehlende oder
  inkompatible Artefakte bleiben normal baubar.
- Familienbezogene Computeentscheidung und prozesslokaler Hailo-Compilerkontext;
  absolute Assembler-/libdevice-Auflösung, dynamisches GPUziel, private Komponentenansicht,
  konsistente letzte Kindumgebung vor SDK-Initialisierung.
- Begrenzte private Modellbuilddiagnose über den bestehenden Builder; Produktions-
  Cachepublikation getrennt, tatsächliche Phasen und Fehler erhalten.
- Hailo8-Paketplanung separat anhand aktueller Metadaten; keine impliziten Paketänderungen.
- Vorbereitungsprofilkopie mit effektiver Herkunft und Wiederverwendungspolitik.
- Standalone-Supervisor-Test in eigenem Prozess, damit er keinen gemeinsamen
  multiprocessing resource_tracker des pytest-Prozesses beendet.
- Aktuelle Installertexte und Paketmetadaten-Versionsermittlung ohne deprecated
  jsonschema.__version__.

## Erhaltene Verträge

V33 Boolean-/Speicherkonflikte und Generic-Full-Bindung; v32 Full-Fehlerpersistierung,
Primärexception und explizite Compilerpriorität; vorhandene Receipts und atomare
Publikation; Native/Quality-FIRST; historisches Replay und ungecacheter Terminalabschluss.
Keine neue Hashart, kein globaler CUDAwechsel, keine neue Compilerparallelität.

Normale Forcefreigabe ist bewusst enger als v33. Der einzige Neubau-Ausnahmefall
ist ein ausdrücklich angeforderter, privater Diagnoseauftrag ohne produktive
Publikation. Historische Run-Snapshots werden nicht umgeschrieben.

## Grenzen

Reale Hardwareausführung in der lokalen Softwareabnahme: **NOT_RUN**.
Aktuelle Smartmirror-Installation, normaler Hardware-Reuse, GPU-Modellbuild und
Finalqualität sind eigene Gates. Historische Hailo10-R2-Rechnung ist getrennte
Evidence; Hailo8-GPU, MobileNet-/YOLO26-Qualität und DeepX-B500 bleiben offen bis
zu tatsächlicher Zielausführung. Ein technischer Build darf trotz Quality-FAIL
erfolgreich sein; daraus folgt kein positiver wissenschaftlicher Claim.

`COMPILE_INFEASIBLE` ist eine exakte deterministische Buildaussage.
`TRANSIENT_INFRASTRUCTURE` ist ein erneut versuchbarer Infrastrukturfehler.
Fehler, not_run, not_available und unproven werden nicht als PASS zusammengefasst.

# Testanleitung ONNX-Splitpoint-Tool 2.79.21

Build-ID: `v2.79.21-cache-preflight-atomic-publication`

Diese Version ergänzt die Cache-Preflight-Barriere um die endgültige
Splitauswahl und führt HEF, Receipt und Cache-Meta bei der Veröffentlichung
zusammen. Ein vorhandenes HEF allein ist kein belegter Cache-HIT.

## Installation

Das vollständige Lieferpaket entpacken. Das Source-ZIP muss beim Installer
oder in `~/Downloads` liegen. Bei geschlossenem Tool ausführen:

```bash
bash install_v27921_and_collect_acceptance.sh
```

Der Installer aktualisiert die vorhandene Installation
`~/ONNX-Splitpoint-Tool`, erhält deren `.venv` sowie benutzereigene Profile
und prüft die Versionsidentität. Er startet keine Compiler oder Hardwareläufe.
Das Ergebnis steht in dem ausgegebenen `EVIDENCE_ZIP`.
Bei einem Installationsfehler bleibt dessen ursprüngliche Ursache im Log
sichtbar. Ist zusätzlich die Rücksetzung unvollständig, benennt die
Diagnose die erhaltenen Backups und betroffenen Zielpfade.

## Offline-Acceptance

```bash
cd ~/ONNX-Splitpoint-Tool
bash scripts/run_v27921_small_acceptance.sh
```

Erwartete Schlusszeilen:

```text
PASS v2.79.21 smoke
PASS v2.79.21 small acceptance (real cache preflight and hardware EvalRun: NOT_RUN)
```

Die Acceptance prüft die neue Release-Identität, die finale Cache-Matrix,
Cold-Build-Gründe, Auswahländerungen, atomare Cache-Publikation,
`legacy_unsealed`, deterministische Duplikatauswahl sowie die bisherigen
Hailo-, TensorRT- und DeepX-Reuse-Regressionsfälle. Anschließend folgen die
Prüfung des ausgelieferten Source-Manifests, Python-Kompilation und Shell-Syntax.
Alte versionsgebundene Release-Smokes bleiben historische Diagnosen; der
aktuelle allgemeine Acceptance-Aufruf führt zu 2.79.21.

## Erster kurzer Lauf auf Smartmirror2

Zuerst einen begrenzten Lauf mit den bekannten Modellen und Setups verwenden.
Bei normaler, compilerunabhängiger Auswahl muss die endgültige
Splitauswahl samt vollständiger Cache-Matrix vor dem ersten Backend-Compiler
vorliegen. Danach die Cache-Preflight-Ausgabe im Workflow-Log und die
erzeugten JSON-, CSV- und Markdown-Berichte prüfen. Für compilerabhängige
Gate-A-Auswahl gilt der zusätzliche Ablauf im folgenden Abschnitt.

Für jeden angeforderten Full- oder Split-Artefakt muss eine eigene Diagnose
vorliegen: Modell, Boundary beziehungsweise `full`, Backend, Rolle, Status
und Grund. Ein unerreichbarer Remote-Cache ist `UNKNOWN`; daraus darf weder
ein HIT noch ein bewiesener Cold Build werden.

Für das ursprüngliche Problem gilt: Wenn YOLO26m `b398` gewählt wird und
kein gültiger Hailo-10H-Part1-Cache existiert, muss dies vor Optimierung oder
Kompilierung als Cold Build sichtbar sein. Vorhandene Builds anderer
Boundaries ersetzen kein Artefakt für `b398`.

Die folgende Ausgabe zeigt das erwartete Informationsformat; die vorherigen
Boundaries werden aus dem tatsächlich vorhandenen Bestand ermittelt:

```text
selection_changed:
previously_built_boundaries=[b250, b350]
currently_selected=b398
expected_cold_builds=[hailo10h_part1]
```

Die Beispiel-Boundaries `b250` und `b350` sind keine Aussage über den
tatsächlichen Bestand des Geräts.

## Gate-A: Compilerprüfung während der Auswahl

Gate-A kann einen Compiler benötigen, um die Machbarkeit eines Kandidaten
zu beurteilen und dadurch die endgültige Auswahl festzulegen. Für jeden
solchen Kandidaten wird zunächst ein Cache-Preflight mit dem Scope
`selection_probe` in `selection_probe_cache_preflight.json` gespeichert
und sichtbar protokolliert. Er zeigt die für diese Auswahlprüfung
erwarteten Cold Builds, bevor der zugehörige
Compiler startet. Diese Ausgabe ist noch nicht die finale Splitauswahl.

Nach Abschluss der Auswahl folgt zusätzlich die vollständige finale
Cache-Matrix für die tatsächlich ausgewählten Boundaries. Erst danach
starten die noch benötigten regulären Backend-Builds. Im strikten
Warm-Cache-Modus blockieren ein unerwarteter MISS bei erwartetem warmem
Cache sowie UNKNOWN bereits die vorgeschaltete Auswahlprüfung, bevor
deren Compiler starten kann. Explizit als erwartet deklarierte Cold Builds
bleiben gemäß der bestehenden Policy zulässig. Im diagnostischen Modus
kann ein zuvor gemeldeter Cold Build ebenfalls ausgeführt werden.

Direkte Builds im Benchmark-Tab behalten den bisherigen Ablauf.

## Cache-Publikation und Legacy-Bestand

Nach einem erfolgreichen Hailo-Cold-Build müssen HEF, Receipt und Cache-Meta
gemeinsam als vollständige Generation vorhanden sein. Eine unterbrochene
Publikation darf keine halb geschriebene Generation als gültigen HIT
sichtbar machen; ein zuvor gültiger Bestand muss bei einem fehlgeschlagenen
Ersatz erhalten bleiben. Die Publikation setzt ein Dateisystem mit
Symlink-Unterstützung voraus; andernfalls bricht sie vor dem Ersetzen
bestehender Kompatibilitätsdateien ab.

Ein altes HEF ohne Receipt wird mit `legacy_unsealed` bezeichnet. Die Datei
bleibt als Legacy-Bestand erhalten; sie wird nicht allein wegen ihrer
Existenz zu einem validierten Cache-HIT. Ist dagegen ein gültiges Receipt
vorhanden und fehlt lediglich `cache_meta.json`, kann der historische
Bestand anhand des Receipts geprüft und in eine vollständige Generation
migriert werden. Bestehende gültige Cache-Schlüssel bleiben erhalten.
Ein ungültiger oder unvollständiger Artifact-Store-Kandidat darf einen
anderen passenden und vollständig validen Kandidaten nicht verdecken.

## Unveränderter zweiter Lauf

Den gleichen begrenzten Lauf mit unveränderter Splitauswahl wiederholen.
Für nun gültige Cache-Artefakte erwartet die Matrix HITs; die jeweiligen
Compiler dürfen dafür nicht erneut starten. TensorRT Full bleibt
modellgebunden und wird durch einen bloßen Boundary-Wechsel nicht zu einem
Split-Artefakt. Ein Cache-HIT ersetzt weiterhin keine Quality-Prüfung.

## Statusgrenzen

Ein Offline-PASS belegt die geprüften Kontrollflüsse einschließlich
simulierter Fehlerfälle. Es belegt keinen realen Hailo-, TensorRT- oder
DeepX-Build und keine Messung mit u.RECS.

```text
REAL_SEVEN_MODEL_CACHE_PREFLIGHT=NOT_RUN
REAL_HAILO_COLD_TO_WARM=NOT_RUN
REAL_TRT_REUSE=NOT_RUN
REAL_DEEPX_REUSE=NOT_RUN
REAL_MULTI_HOST_EVALRUN=NOT_RUN
```

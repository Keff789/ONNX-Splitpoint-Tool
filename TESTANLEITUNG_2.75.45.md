# Testanleitung 2.75.45

Release: `2.75.45`

Build/Workflow: `v2.75.45-gui-large-audit-trt-working-set-admission`

## Zweck

v2.75.45 stellt den vollständigen GUI-Startpfad für einen großen,
score-unabhängigen Ranking-Audit wieder her. Audit-Größen von 20 und 30
Kandidaten pro Modell bleiben wissenschaftlich gültig und werden weder
verkleinert noch auf den sichtbaren Wert **Cases/model** begrenzt. Vor einem
neuen Audit zeigt die GUI deshalb die tatsächliche Audit-Größe, die minimale
und maximale Ausführungsunion, die erwarteten Generic-Zeilen sowie Native- und
Native-Energy-Status an und verlangt eine explizite Bestätigung.

Der bisherige 20-GiB-Wert begrenzt ab diesem Release nur noch die
**aufbewahrten, nicht aktuellen** verwalteten TensorRT-Namespaces. Der aktuelle
Namespace und sein konservativ geplanter Zuwachs bilden das aktive Working Set.
Dieses Working Set darf größer als 20 GiB sein, sofern die unveränderte
physische Speicherplatzprüfung es zulässt. Der stabile aktuelle Namespace
bleibt erhalten und kann bei Resume oder Wiederverwendung erneut genutzt
werden.

Aktive, fremde/unowned, verlinkte oder anderweitig unsichere Namespaces werden
nicht gelöscht. Bereits vorhandene, eigene, mit gültigen Receipts attestierte
und inaktive alte Namespaces bleiben wie bisher LRU-fähig, wenn das getrennte
Retentionsbudget oder die Namespace-Anzahl tatsächlich überschritten wird.
Temporäre Run-Verzeichnisse folgen weiterhin ihrem vorhandenen Cleanup-Vertrag.

Zusätzlich macht der Profil-Editor das DeepX-Classification-Preprocessing
explizit. Neue Profile verwenden `imagenet_mean_std`. Bei alten Profilen ohne
dieses Feld wird aus Kompatibilitätsgründen `current_scale_only` materialisiert;
ein bereits expliziter Wert bleibt über Run-Modi hinweg erhalten.

## Lokale Release-Prüfung

Aus dem Tool-Ordner:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool
PY=/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python \
  bash scripts/run_v27545_small_acceptance.sh
```

Diese Prüfung startet keine Hardware, kein SSH, keine Inferenz, keinen
TensorRT-/DX-COM-Build und keine Energieerfassung. Sie prüft insbesondere:

- Audit-Größe 20 und 30 ohne Clamp;
- die Profilbedingung `minimum_valid_audit_candidates <= audit_size`;
- GUI-Abbruch ohne Results-Ordner und ohne registrierten Job;
- GUI-Bestätigung mit registriertem Audit-Job;
- ein aktives 30-GiB-TensorRT-Working-Set bei 20-GiB-Retentionsbudget;
- unveränderte reusable, aktive und fremde Cache-Namespaces;
- Wiederholung mit demselben aktuellen Namespace für Resume/Reuse;
- den unveränderten physischen Speicherplatz- und Pre-Mutation-Gate;
- explizites DeepX-Classification-Preprocessing und historische Releases.

## Unveränderter Final-Quality-Pfad

**Final Quality** bleibt der vorhandene **Standard**-Pfad mit 5.000
Classification- und 5.000 Detection-Validation-Items. v2.75.45 verändert
diesen wissenschaftlichen Umfang nicht und erzeugt dafür **keinen Pflicht-Canary**.
Die historischen Werkzeuge
`run_v27528_*_final_canary.sh` bleiben optionale Archivdiagnostik.

## Headless GUI-Vertragsprüfung

Der fokussierte Test kann separat ausgeführt werden:

```bash
cd /home/kmika/ONNX-Splitpoint-Tool
/home/kmika/ONNX-Splitpoint-Tool/.venv/bin/python -B -m pytest -q \
  -p no:cacheprovider --tb=short \
  tests/test_v27545_large_audit_working_set_admission.py
```

Er benötigt keine Hardware. Der Test hält einen fremden aktiven Namespace
absichtlich gesperrt und prüft zweimal, dass der große aktuelle Audit zugelassen
wird, ohne reusable, aktive oder unowned Daten zu entfernen.

## Manueller GUI-Akzeptanztest

1. Im Profil-Editor `score_independent_audit` aktivieren und als
   `audit_size` zuerst 20, danach optional 30 wählen. Der Mindestwert darf nicht
   größer als die Audit-Größe sein.
2. Das gewünschte DeepX-Classification-Preprocessing sichtbar auswählen. Für
   neue ImageNet-Classification-Profile ist `imagenet_mean_std` der Standard.
3. Native und Native Energy nach dem Messplan aktivieren. Beide Zustände müssen
   in der Profilzusammenfassung und im Startdialog sichtbar sein.
4. **Start** wählen. Der Bestätigungsdialog muss die Kandidaten pro Modell,
   Ausführungsunion und Generic-Zeilen nennen und darauf hinweisen, dass der
   Audit unabhängig von **Cases/model** läuft.
5. Einmal **Nein** wählen. Es darf kein neuer Results-Ordner und kein neuer Job
   entstehen; der Status lautet
   `Start abgebrochen: Ranking-Audit nicht bestätigt.`
6. Erneut **Start** und anschließend **Ja** wählen. Der Job muss angelegt werden;
   seine Startzeilen enthalten
   `Ranking-audit start confirmation: accepted`.

Resume, **Rerun generated** und das Finalisieren eines Partial-Runs verlangen
keine erneute Fresh-Start-Bestätigung. Sie verwenden weiterhin den eingefrorenen
Start-Snapshot des vorhandenen Runs.

## Hardwarelauf und Cache-Nachweis

Der echte Audit darf erst nach erfolgreichem lokalem Release-Test und mit
ausreichend freiem Speicher auf dem ausgewählten Remote-System gestartet
werden. Es ist weder nötig noch zulässig, alte Caches allein deshalb manuell zu
löschen, um das aktive Working Set künstlich unter 20 GiB zu drücken. Auch eine
Erhöhung von `ONNX_SPLITPOINT_REMOTE_TRT_CACHE_MAX_BYTES` ist für einen
20-/30-Kandidaten-Audit nicht erforderlich.

Im Workflow-Log erscheint die strukturierte Zeile
`[remote][cache][retention]`. Für einen großen aktuellen Audit müssen dort unter
anderem folgende Beziehungen gelten:

```text
selected_plan_reserve_applied = true
active_working_set_bytes = current_namespace_bytes + planned_current_growth_bytes
retained_noncurrent_bytes <= retained_cache_budget_bytes
projected_managed_bytes <= effective_admission_max_bytes
```

Eine Zulassung durch diesen Retentionsvertrag ersetzt nicht die physische
Speicherplatzprüfung. Reicht der tatsächlich freie Platz für Cold/Warm-Reserve,
Run-Artefakte und das aktive Working Set nicht aus, muss der Lauf weiterhin vor
Remote-Mutation blockieren.

Nach einer Unterbrechung wird derselbe Run über **Resume latest** fortgesetzt.
Der aktuelle stabile TensorRT-Namespace muss wiederverwendbar bleiben. Ein
manuelles Entfernen aktiver oder nicht eindeutig Tool-eigener Remote-Daten ist
kein Bestandteil dieser Anleitung.

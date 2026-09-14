# v2.79.33: Force bewusst starten

`hailo_build.force_build` und `deepx_build.force_build` bleiben reguläre
Buildoptionen. `true` bedeutet weiterhin: passende Cachetreffer absichtlich
übergehen. Die neue Startbestätigung schaltet Force weder ein noch aus.

## Artefaktvorbereitung mit Wiederverwendung

Nach dem Ende des laufenden Workflows alte Einstellungsdialoge schließen und
die GUI frisch öffnen. In `Tool Config → Run modes → final → Edit selected…`
beide Forcefelder ausschalten, speichern und neu laden. Keine Modusdefaults
zurücksetzen. Im tatsächlich verwendeten Evaluationsprofil zusätzlich
`deepx_build.classification_preprocessing: imagenet_mean_std` auswählen.

Die Startsummary muss die tatsächliche Quelle und folgende Werte zeigen:

- Hailo Force AUS und DeepX Force AUS;
- Buildmodus `reuse_and_build_missing`;
- Hailo-Integrität weiterhin `relaxed` für den vereinbarten Vorbereitungslauf;
- DeepX Classification `imagenet_mean_std` aus dem Evaluationsprofil.

Hailo B500/Opt1/Batch8 und DeepX B500/EMA/Opt0 bleiben bestehen. Einen später
explizit gewählten Strict-/Freeze-Vertrag behandelt der vorhandene Integritäts-
und Qualitätsvertrag weiterhin gesondert.

Bei `follow_tool_config=true` kommen die Forcewerte aus dem ausgewählten Modus
der zentralen Registry. Bei `false` gilt der gebundene Profilsnapshot. Das
Ändern der Registry ändert weder einen gestarteten Lauf noch dessen archivierte
Profile. Die Installation setzt vorhandene `true`-Werte nicht zurück.

## Absichtlicher Force-Start in der GUI

Vor dem Start nennt ein Bestätigungsdialog die betroffenen Backendfamilien
Hailo und/oder DeepX. Der Text erklärt, dass kompatible Cachetreffer bewusst
übergangen werden. Ablehnen startet keinen Compiler. Die Bestätigung gilt
ausschließlich für diesen Start; ein erneuter Start verlangt eine neue
Entscheidung.

## CLI und automatisierte Aufrufe

Ohne aktive Forcewerte bleiben bestehende Aufrufe unverändert. Ein geerbtes
`true` aus einer YAML oder zentralen Registry ist allein keine Startfreigabe.
Der Runner meldet dann `force_build_confirmation_required`, nennt die
betroffenen Backends und beendet sich vor dem Compilerdispatch.

Die neue Option ist wiederholbar und bestätigt ausschließlich die benannte
Backendfamilie für diesen Aufruf:

```bash
python scripts/run_evaluation_workflow.py \
  --profile-driven --profile /pfad/zum/profil.yaml \
  --out /pfad/zu/EvaluationRuns --execution-mode generate_and_run \
  --confirm-force-build hailo --confirm-force-build deepx
```

Nur die tatsächlich gewünschten Bestätigungen angeben. Ist beispielsweise
auch DeepX Force aktiv, genügt `--confirm-force-build hailo` nicht. Ist Force
ausgeschaltet, aktiviert diese Bestätigungsoption keinen Neubau.

Die vorhandene Option `--hailo-force-build` ist im klassischen CLI-Aufruf eine
ausdrückliche Anforderung und Bestätigung für Hailo zugleich. Sie gilt nicht
als Zustimmung für DeepX. `--profile-driven` erlaubt weiterhin keine solche
Buildüberschreibung; dort ausschließlich `--confirm-force-build hailo` nutzen,
wenn der unveränderte Profilvertrag Hailo Force vorsieht.

Automationen, die bisher still aus dem Profil geerbtes Force verwendet haben,
müssen die konkrete Bestätigung ausdrücklich in ihren Startbefehl aufnehmen.
Keine Bestätigungsfelder in eine Profil-YAML schreiben.

## Resume

Ein Resume verlangt ebenfalls eine neue Forceentscheidung. Vorhandene
Bestätigungen aus alten Execution-Sessions oder aus Profilfeldern werden
nicht wieder als Zustimmung verwendet. Bei einem explizit ausgewählten Run
zeigt die GUI die Forcewerte der gegen den Manifest-Hash geprüften
archivierten `profile.yaml` und deren Pfad. Abweichungen vom heutigen
aufgelösten Profil bleiben sichtbar.

Die Bestätigung hebt kein Resume-Gate auf. Die vorhandenen Prüfungen für
Quellprofil, Effektivprofil, Snapshot, Ausführungsplan, Optionen und
Artefaktidentität müssen weiterhin passen. Nach einer bewussten Änderung von
Force oder Mean/Std einen neuen Lauf mit neuem Snapshot starten. Historische
`profile.yaml`, `profile_source.yaml` und Snapshotdateien nicht bearbeiten.

Im klassischen Resume-Aufruf können die notwendigen
`--confirm-force-build hailo` beziehungsweise `--confirm-force-build deepx`
zusätzlich zu den bestehenden, passenden Resumeparametern angegeben werden.
Die Option `--profile-driven` erhält dadurch keine neue `--resume`-Semantik.

## API und Provenienz

Direkte Aufrufer von `EvaluationWorkflowRunner` müssen nach einer eigenen
bewussten Startentscheidung die aktuellen `WorkflowOptions` setzen:

```python
options.force_build_confirmed_backends = ("hailo",)  # ggf. zusätzlich "deepx"
options.force_build_confirmation_source = "my_explicit_start_confirmation"
runner = EvaluationWorkflowRunner(options)
runner.run()
```

Keine alten Manifestoptionen als neue Zustimmung importieren. Der Runner
prüft die effektiven Werte erneut nach der Resume-Snapshotprüfung und vor
der ersten Laufmutation beziehungsweise dem ersten Compilerdispatch. Bereits bei jedem Startversuch
verbraucht er die Zustimmung dieser Invocation, auch wenn ein früheres Gate
oder eine unvollständige Zustimmung den Start abweist. Er protokolliert
die erfolgreiche Zustimmung unter
`run_manifest.json → execution_sessions[] → force_build_start` mit Session-ID,
Zeitpunkt, betroffenen Backends und Herkunft. Die Zustimmung ist kein
Bestandteil der Profil-, Cache-, Stage- oder Resumeidentität.

## Typfehler

Reguläre Loader, GUI, CLI, Snapshotvalidierung und Buildgrenzen akzeptieren
für Force ausschließlich echte Booleans. Fehlende Forcefelder erhalten
`false`. Texte wie `"false"` oder `"true"`, Zahlen, Listen und `null` werden
mit `config_boolean_invalid:<konkretes Feld>` abgelehnt. Insbesondere wird
`"false"` nicht mehr durch Python-Wahrheitskonvertierung zu `true`.

Der gemeinsame Parser bietet einen ausdrücklich aufzurufenden
Legacy-Textmodus für Migrationstools. Die regulären Startpfade aktivieren
diesen Modus nicht und schreiben keine Registry nur für eine Summary um.

## Prüfgrenze

Die Regressionen verwenden synthetische Fehlwerte und Compilerzähler sowie
den realen lokalen Runner-, Lock-, CLI- und Resume-Startpfad. Sie prüfen
Ablehnung vor Dispatch, selektive einmalige Zustimmung, unveränderte
Archivbytes und bestehende Identitäten. Daraus folgt keine neue
Hardware-, Durchsatz- oder Qualitätsabnahme.


Der vorhandene ausdrücklich erzwungene Hailo-Parallelbuild-Canary übergibt
seine Hailo-Bestätigung weiterhin automatisiert als Teil genau dieses
Canary-Auftrags. Seine Parallelitäts- und Buildparameter ändern sich nicht.
`scripts/resume_missing_full_quality.py` übernimmt keine Zustimmung aus alten
Manifestoptionen; auch dieser Wrapper akzeptiert die wiederholbare
`--confirm-force-build`-Option für eine neue ausdrückliche Entscheidung.

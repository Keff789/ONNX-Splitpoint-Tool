# v2.79.33 – Konfiguration, Wiederverwendung und getrennte Hardwareabnahme

Build-/Workflow-ID: `v2.79.33-force-profile-scope-safety`.

Die Source baut auf dem vollständig abgenommenen v2.79.32 FIX2 auf. Die lokalen Ausführungszahlen und die Identität des endgültigen Sourcearchivs stehen im Prüfbericht des Delivery-Bundles. Zielhardware, GPU G0/G1 und neue Qualitätsmessungen haben hier den Status `NOT_RUN`.

## Installation und Softwaregate

GUI, offene Einstellungsdialoge sowie laufende oder pausierte Workflows regulär beenden. Im Delivery-Ordner `bash install_v27933_and_collect_acceptance.sh` als Besitzer der vorhandenen Installation ausführen. Der Installer verwendet deren Venv und den bestehenden Source-Updater. Er installiert keine Abhängigkeiten. Nach einem Fehler können die Sourcebytes bereits aktualisiert sein; `FINAL_STAGE`, installierte Identität und das vollständige Log prüfen. Ohne bestandenes Gate keinen anschließenden Hardwarelauf starten.

Das reguläre Gate kann separat ausgeführt werden:

```bash
bash scripts/run_v27933_small_acceptance.sh --report /tmp/v27933-acceptance.json
```

Die vorhandene Tool-Venv wird automatisch gewählt; `PY` erlaubt eine ausdrücklich gewählte bestehende Test-Venv. Bericht und JUnitdateien müssen außerhalb des Sourcebaums liegen. Pflichtskips, Xfails oder fehlende Abhängigkeiten sind kein PASS. Historische v32-F1/F2/F3- und FIX2-Verhaltensprüfungen bleiben Teil der Abnahme. Nur die fest auf v32 gebundene Releaseprüfung wird durch die v33-Prüfung ersetzt.

## Einmalig das tatsächlich verwendete Profil korrigieren

Nach dem Update die GUI frisch öffnen. Unter **Tool Config → Run modes → final → Edit selected… → Data & build** beide Force-Optionen bewusst ausschalten, speichern und neu laden. Kein Reset der Modusdefaults. Die Originalregistry enthielt echte `true`-Werte für beide Final-Backends; deren ursprünglicher Schreiber ist nicht belegt. Das Update erhält diese Nutzerwerte.

Separat im verwendeten Evaluationsprofil **DeepX Classification = imagenet_mean_std** auswählen. Bei `follow_tool_config=true` stammen die Force-Werte aus der zentralen Registry. Eine Änderung allein in `hailo_build.force_build` oder `deepx_build.force_build` des Profils reicht deshalb nicht. Die explizite Classification-Auswahl stammt dagegen aus dem Evaluationsprofil und übersteuert den zentralen Default.

Für den vereinbarten Standard+-Vorbereitungsablauf gelten weiterhin:

| Einstellung | Erwarteter effektiver Wert |
|---|---|
| Hailo / DeepX Force | AUS / AUS |
| Buildmodi | reuse_and_build_missing |
| DeepX Classification | imagenet_mean_std, ausdrücklich im Profil |
| Hailo-Rezept | B500, Opt1, Batch8 |
| DeepX-Rezept | B500, EMA, Opt0 |
| Hailo-Integrität | relaxed |

Die Startsummary zeigt die tatsächliche Quelle einschließlich Registrypfad und aufgelöstem Modus. Ein gebundener Snapshot mit `follow_tool_config=false` bleibt maßgeblich. Nicht allein auf die Combobox oder `default_mode` vertrauen. Ein ausdrücklich gewählter strenger Final-/Freeze-Vertrag wird nicht abgeschaltet, um einen Cachetreffer zu erhalten.

Force AN bedeutet: kompatible Cachetreffer werden bewusst übergangen. Die GUI verlangt dafür bei jedem neuen Start und Resume eine backendbezogene Bestätigung. Ein altes YAML-`true` oder eine alte Bestätigung ist keine neue Zustimmung. Die CLI-Optionen und die genaue Zustimmungssyntax stehen in `python scripts/run_evaluation_workflow.py --help`; die ergänzende Force-Anleitung dokumentiert Beispiele. Zustimmung ist Laufprovenienz und keine Cacheidentität.

Bei einer Konfliktmeldung des Konfigurationseditors neu laden und die gewünschte Änderung erneut vornehmen. Es erfolgt kein automatischer Merge. Alle Tool-Writer koordinieren sich über eine lokale Registry-Schreibsperre; externe, nicht kooperierende Texteditoren liegen außerhalb dieser Garantie. Ungültige Boolwerte wie `"false"`, Zahlen, Listen oder `null` werden mit konkretem Feldnamen abgelehnt. Sie werden nicht durch unverändertes Speichern zu `true`.

## Vor dem nächsten Nachtlauf

Den wirksamen neuen Profilsnapshot kontrollieren: Force AUS, Mean/Std ausdrücklich, korrekte YOLOv7-Modellquelle, vereinbarte Splitpunkte und unveränderter Buildvertrag. Historische Run-YAMLs und Resultate bleiben unverändert. Resume eines alten Force-Snapshots zeigt den alten Vertrag und benötigt eine neue bewusste Entscheidung.

Zuerst einen kleinen normalen Workflow mit vorhandenen passenden Artefakten ausführen. Die Prüfung muss an der tatsächlichen Build-/Runtimegrenze positive Receipt-/Cachevalidierung und **null Compilerdispatches** belegen. Eine reine Preflight-HIT-Zeile reicht nicht. Eine zweite identische Anfrage muss denselben Artefaktvertrag wiederverwenden. Neues Mean/Std bei bisherigem scale-only ist ein erwarteter Neubau und kein verlorener kompatibler Cache.

Vorher-/Nachherbestand aus vorhandenen Metadaten und Receipts vergleichen. Exakte `COMPILE_INFEASIBLE`-Evidenz verhindert denselben unnötigen Wiederholungsbuild; `TRANSIENT_INFRASTRUCTURE` bleibt wiederholbar. Unabhängige gültige Jobs bleiben ausführbar. Die bereitgestellten Offlineprüfungen dieses Kontrollflusses ersetzen den Nachweis an den echten vorhandenen HEFs/DXNNs nicht.

Im Vorlauf lagen sieben TensorRT-Namespaces bei Softlimit sechs vor. Vor dem Nachtlauf die vorhandene Limit-/Pinpolicy bewusst für den gewünschten Bestand konfigurieren; Details stehen in der ergänzenden Retentionsanleitung. Keine Caches löschen und keine Pruningpolicy still umstellen. Tatsächliche Löschungen separat ausweisen. Der Nachtlauf dient Artefaktvorbereitung und Integration.

## GPU G0/G1 und Qualitätsgrenzen

Der mitgelieferte G0-Rechensmoke nutzt die vorhandenen DFC-Venvs nacheinander unter der bestehenden Workflow-/Plattformsperre. Er installiert nichts und schaltet den produktiven Hailo-CPU-Default nicht um. Zuerst dessen Anleitung und `--help` verwenden. Ein Lockkonflikt startet kein Kind und erzeugt kein leeres Erfolgsarchiv. Simulierte TensorFlow-Tests sind keine echte GPU-Abnahme.

G1 erst nach erfolgreichem G0: genau ein vorhandener Modell-/Buildvertrag, bestehendes CPU-HEF behalten, GPU-HEF außerhalb des Produktionscaches erzeugen. Tatsächliche Geräte, TF/CUDA/cuDNN/XLA, Optimierungsphasen und Laufzeiten protokollieren; ausgewählte Eingaben, Ausgaben und Top-k gegen CPU und Float prüfen. Kein bloßer SDK-Import als Buildbeleg, keine geänderte Recipe als reiner Geschwindigkeitsvergleich, kein B500-/B5000-Claim aus Rechensmokes.

Der bereits erfolgreiche DeepX-Full-Diagnosesmoke und die FIX2-Matrixkorrektur bleiben gültige historische Software-/Ausführungsnachweise. MobileNet-Hailo-B5000-Verlust, Hailo10/YOLO26 AP6b sowie produktive DeepX-Mean/Std-Qualität bleiben eigene offene Abnahmen. Keine neue Decoder-/Quantisierungskorrektur aus Vermutungen und kein erneutes identisches R1/R2 ohne neue Fragestellung.

Die bisherigen Integrationsumfänge 1 s × 3 und 30 s × 3 sind kein finaler Energievertrag. Wissenschaftlicher Finalscope bleibt separat: Native Energy measure, FS/command, 60 s × 3, passende Gain-/Idlebelege und gültige Endpunktbindung. B500 ist Screening; 5.000 Bilder und 5.000 Bootstrapwiederholungen müssen separat vorab festgelegt und ausgeführt werden.

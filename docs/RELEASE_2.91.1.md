# Releasevorbereitung 2.91.1

## Nachbesserung vom 29.09.: Cache-Preflight und Ergänzungs-Reentry

Der erste produktive Ergänzungsversuch scheiterte vor dem Dispatcher an
`supplement_source_contract_changed:yolo26m:build_backend_artifacts`.
Der frische Preflight hatte bei YOLO26m und YOLO26s jeweils 16 abgeleitete
JSONs neu geschrieben. Alle ursprünglichen Bindungen waren vorhanden;
geändert waren ausschließlich Probezeitstempel und im DeepX-Status der
explizite Preflightmodus `cache_verify_only`.

Der enge Fix verwendet weiterhin den ursprünglichen Artefaktindex aus der
ersten Supplement-Sicherung. Nur die genau benannten Zeitstempelfelder und
dieser DeepX-Probeoverride werden gegen die erhaltenen Originalbelege geprüft.
Das gesamte übrige JSON muss die vorhandene ursprüngliche SHA256 und Größe
reproduzieren. Die vier vom alten Sicherungsmuster ausgelassenen Dateien pro
Modell lassen sich aus erhaltenen Writerzeitstempeln und dem archivierten
DeepX-Modus exakt gegen diese alten Prüfsummen belegen. Diese Rekonstruktion
bleibt im Speicher; Originaldateien und Receipts werden nicht zurückgeschrieben.
Neue Attempts sichern auch diese vier Metadaten. Abweichende Grenzen,
Precision, Artefaktbindungen oder fehlende Originalbelege bleiben Fehler.
Ein späterer Attempt ersetzt nicht die ursprüngliche Vertragsbasis.

Zusätzliche Erhaltungsprüfungen verhindern, dass fehlende oder veränderte
Originalergebnisse als neue Messaufgaben erscheinen. Die CLI zeigt vorhandene
Start-, Reuse-, Skip- und Fehlermeldungen sofort. Ursprüngliche Messzeiträume
bleiben erhalten; neue Prüfzeiten stehen getrennt in `reuse_check_*` und
`resume_attempt_*`.

Die langen Skipabstände entstanden durch wiederholte Pfadprüfung aller
257.313 Indexeinträge bei bloßer Neuregistrierung der Jobprojektionen.
Diese redundante Arbeit entfällt beim Supplement-Reuse; normale
Artefaktinhaltsprüfungen und die terminale Inventarisierung bleiben erhalten.
Gemessen: Registrierung vorher 125,461 s; normaler isolierter Reuse danach
0,691 s beziehungsweise 0,624 s. Eine anschließend geänderte Originalquelle
wird weiterhin abgelehnt. Kein neuer Prüfungs- oder Persistenzcache.

Die lokale Abnahme umfasst echte Preflightwriter, beide späteren Reusezweige,
den normalen Dispatcher mit kontrollierten äußeren Blättern, Wiedereintritt
und Negativkontrollen. Der geprüfte Originalplan bleibt bei 399 Generic,
414 Quality und 213 Native erhaltenen Ergebnissen sowie 37 negativen
Buildbefunden; offen sind 152 Generic-Kandidaten, sechs Quality-Companions
und 21 Native-Fälle. Der unten stehende Fortsetzungsbefehl gilt weiterhin für
denselben Run. Kein produktiver Ergänzungs-, SSH-, Compiler-, Collector- oder
Workloadstart gehört zu dieser Nachbesserung. Softwareabnahme und physische
Abnahme bleiben getrennt.

Dieser Patch macht den vorhandenen THESIS20-Run selektiv ergänzbar. Er startet
keine neue Kampagne. Messungen und deren ursprüngliche Codeprovenienz bleiben
erhalten; positive wie negative abgeschlossene Qualityresultate werden verwendet.
Die wissenschaftlichen Verträge, Geräteinstallation und Artefaktidentitäten
bleiben unverändert. Force bleibt aus.

Der normale Workflow ist über `onnx-splitpoint-complete-evaluation --run-dir RUN`
erreichbar. `--plan-only --output-dir OUT` prüft die archivierten Verträge und
schreibt ausschließlich einen lokalen Ergänzungsplan nach OUT. Dieser Einstieg
rekonstruiert die ursprünglichen Optionen aus dem Run; er erlaubt keine neue
Kohortenauswahl oder pauschales Ignorieren von Resume-Konflikten.

Die Quellenauflösung verwendet die bestehenden Splitmanifeste und lässt
Hailo-Hilfsdateien unangetastet. Native projiziert die kompatible Teilmenge anhand
sämtlicher tatsächlicher Part2-Eingänge. Der lokale Validator zeigt seinen
Fortschritt, vermeidet wiederholte Serialisierung identischer Qualityevidenz und
nutzt beim Ergänzen mindestens 3600 Sekunden. Fehlende aggregierte Validierung
leert keine bereits exakt gebundene Energieprovenienz mehr. Claimfreigabe,
Accuracyentscheidung und technische Messzulassung bleiben getrennt.

Neue Ergebnisse entstehen in Sitzung-/Attemptverzeichnissen. Vor der neuen
Projektion bleiben mutable Originalpläne und Berichte als gezielte Metadatenkopien
erhalten. Geräteartefakte, Vorhersagen und Messpayloads werden dafür nicht kopiert
oder überschrieben. Der reguläre Reporter bilanziert ursprüngliche und neue
Ergebnisse gemeinsam. Der vorhandene ZIP-Stream akzeptiert notwendige
Replay-Metadaten bis 512 MiB ohne Abschneiden oder vollständige Speicherkopie.

Die lokalen Prüfungen und verbleibenden Grenzen sind im
[Arbeitsstand](ARBEITSSTAND.md) dokumentiert. Keine Hardware-, Compiler-,
G1/G2- oder neue Qualitykampagne wurde zur Patchabnahme gestartet. Die bestehende
Development-/Screeningrolle wird nicht zu einer ungeöffneten Hold-out-Freigabe.

## Restarbeit im bestehenden THESIS20-Run

| Modell | Generic behalten | Negative Builds behalten | Generic-Kandidaten | Native behalten | Native nicht unterstützt | Native offen |
|---|---:|---:|---:|---:|---:|---:|
| MobileNet V3 Large | 84 | 0 | 0 | 57 | 9 | 0 |
| ResNet50 | 84 | 0 | 0 | 63 | 3 | 0 |
| YOLO11l | 63 | 21 | 0 | 12 | 54 | 0 |
| YOLO26m | 0 | 10 | 74 | 3 | 51 | 12 |
| YOLO26s | 0 | 6 | 78 | 3 | 54 | 9 |
| RegNet X 1.6GF | 84 | 0 | 0 | 66 | 0 | 0 |
| YOLOv7 | 84 | 0 | 0 | 9 | 57 | 0 |
| Gesamt | 399 | 37 | 152 | 213 | 228 | 21 |

Die 152 Generic-Kandidaten verteilen sich auf YOLO26m: H8 16, H10 16,
DeepX-Setup 42; YOLO26s: H8 18, H10 18, DeepX-Setup 42. Das sind Kandidaten,
keine garantierten Erfolge. Sechs zusätzliche TensorRT-Full-Quality-Companions
für m/s auf den drei Setups fehlen. Die 414 abgeschlossenen Qualityresultate
bleiben inklusive FAIL erhalten. Nur neue oder noch nicht abgeschlossene
Requests werden mit dem ursprünglichen N5000/B1000-Vertrag ausgewertet.

Native fehlen YOLO26m b003/b038/b043 und YOLO26s b003/b021 jeweils auf
H8/H10/DeepX sowie TensorRT Full m/s auf allen drei Setups. Tatsächliche
Multi-Input-Part2-Fälle bleiben dokumentiert und für Native nicht anwendbar.
Kein Nachrücken, keine Änderung der 140 archivierten Genericgrenzen.

Die lokale Nachvalidierung der 213 alten Nativeoutputs ist abgeschlossen:
alle technisch ausführbar, 207 mit exakt gebundener zentraler Qualityevidenz.
Sechs Vendor-Full-YOLO26-Ergebnisse benötigen noch die neue Generic-Qualitybindung;
zwei davon haben einen lokalen semantischen FAIL. Diese Befunde werden erhalten.
Der normale spätere Workflow validiert die zusammengeführte Ergebnismenge lokal.
Ein Quality-FAIL löst keine neue Performancemessung aus.

Energie wurde bislang nicht aufgenommen. Der alte fehlerhafte 213-Zeilen-Plan
bleibt historisch erhalten. Der neue Plan umfasst nach erfolgreicher Ergänzung
bis zu 234 technisch unterstützte Fälle, jeweils FS/command, 60 Sekunden und
drei gültige Replikate, unveränderte Retry-/Transportgrenzen 2/20. Zeilen- und
Teilreplikat-Reuse verwenden die bestehenden Journale und Kampagnenbudgets.
Eine abweichende alte Identität oder ein abweichender Messvertrag wird weiterhin
abgelehnt. Temporäre Ausführungseingaben werden über den normalen Restagepfad
aus lokalen Belegen wiederhergestellt; Geräte-Caches bleiben normal zu prüfen.

## Lokaler Einstieg und späterer Start

Die vorhandene Installation ist mit ihrem normalen Editable-Refresh auf
2.91.1 aktualisiert; alle Abhängigkeiten wurden erhalten (210 CLI-Einstiege).
Der CLI-Einstieg ruft den vorhandenen EvaluationWorkflowRunner auf.
Eine neue GUI-Abnahme gehört nicht zu diesem Auftrag.

Nur lokal planen, ohne Änderung des Originalruns:

```bash
$HOME/ONNX-Splitpoint-Tool/.venv/bin/onnx-splitpoint-complete-evaluation \
  --run-dir $HOME/Models/EvaluationRuns/thesis_20splits_n5000_b1000_20260925_20260925_103745 \
  --plan-only --output-dir /tmp/thesis20-completion-20260929/plan
```

Vollständiger Startbefehl für die spätere freigegebene Ergänzung:

```bash
$HOME/ONNX-Splitpoint-Tool/.venv/bin/onnx-splitpoint-complete-evaluation \
  --run-dir $HOME/Models/EvaluationRuns/thesis_20splits_n5000_b1000_20260925_20260925_103745
```

Dieser zweite Befehl wurde nicht ausgeführt. Er verwendet die archivierten
Profile, Setups und wissenschaftlichen Optionen, erhält den bisherigen
Metadatenstand unter `jobs/supplement_attempts/<session>/original` und schreibt
Ergänzungen in eigene Attempts. Er kann fehlende notwendige TensorRT-Engines
normal bauen, verwendet passende HEFs/DXNNs/Engines weiter und startet nur
ausstehende Messungen. Vor jeder Remoteaufnahme gelten die vorhandenen
Ownership-, Cache-, Eingangs- und Collectorprüfungen. Keine Force-Option.

## Lokale Belege und Grenzen

Baseline: sauberer Commit `182092216dfaf4f3ad36460a883098548c83bd8f`.
Die normale Quellenauflösung wurde an sämtlichen 280 Part1/Part2-Quellen
geprüft; b064/b136 scheitern mit der ursprünglichen Funktion und bestehen
mit Manifestauflösung. Die ursprünglichen 177 Energie-Provenienzfehler wurden
reproduziert. Der vollständige normale Offline-Energiecheck besteht für alle
213 Zeilen sowohl ohne aggregierten Validatorbericht als auch mit dem neuen
vollständigen Bericht und `--previous-results`; keine Messwrapper gestartet.
Der Validator endet nach 953,74 Sekunden mit rc=0. Der vollständige große
Diagnoseexport ist bytegetreu und ohne Änderung der Originalsummary geprüft.

Kontrollierte lokale Prozessblätter prüfen die normalen Stage-, Generic-,
Native-, Reporter- und Energieübergaben einschließlich erneutem Ergänzen,
abgeschlossenen FAILs, kompatiblen Artefakten und negativen Vertragskontrollen.
Der konsolidierte Integrationslauf endet mit 104 PASS, die Release-/Versions-
und Smokeprüfung mit 4 PASS. Der abschließende strikte Native-Energie-Reuse-
Test endet mit 3 PASS. Separat bestehen 187 Energie-/Validatorregressionen;
drei auf Baseline reproduzierte alte v2727-Testfehler sind ausgenommen.
Die Gruppen überlappen. Source- und Installed-Prüfung umfassen 2091 Quelldateien;
43 lokale Profile bleiben als Nutzerdaten außerhalb des Releases erhalten.
Das ist keine physische Geräteabnahme. Ein zukünftiger Hardwarefehler bleibt
von einem Qualitätsbefund und von gespeicherter Compilerinfeasibilität getrennt.
Globale historische Testsuiten, Hardwarekampagnen, Compileraktionen und G1/G2
wurden bewusst nicht ausgeführt. Frühere Release-Tags bleiben unverändert;
dieser Stand wird lokal vorbereitet, nicht veröffentlicht.

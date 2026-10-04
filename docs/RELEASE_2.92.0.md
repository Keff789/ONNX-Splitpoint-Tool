# Release 2.92.0 – selektiver Kampagnenabschluss und Completion

Paket/GUI `2.92.0`, Build/Workflow `v2.92.0`, Entwicklungslinie `v2.79`.
Diese Minorversion veröffentlicht den zusammengehörigen Softwarestand der
THESIS20-Ergänzung, Generic-Completion und separaten YOLO-Min3-Augmentierung.
Historische Messungen behalten ihre ursprünglichen Commands, Quellsnapshots,
Versionen und Attempts. Der Versionswechsel erzeugt keine neue Messfreigabe.

## Enthaltene Produktpfade

- Selektiver Ergänzungseintritt mit geprüften Originalartefakten, normalen
  Generator-/Cacheprojektionen, getrennten Attemptausgaben und unveränderten
  gültigen sowie negativen Ergebnissen.
- Generic-Completion aus gebundenen vorbereiteten Eingaben: drei unabhängige
  Wiederholungen, je 100 Warmups und 1000 fertige Tasks; normale Replikat-,
  Output- und Paarprüfer bleiben maßgeblich. Dataset-Qualityreferenzen werden
  beim Moduswechsel nicht als Completionreferenzen missverstanden.
- Graphbasierte YOLO-Min3-Auswahl, explizite Mixed-composed-Profile,
  N5000/B1000-Quellenübergabe, zentraler Qualityconsumer und begrenzte
  Startbarrieren. Fehlende Qualityquellen passieren beide normalen Selektoren;
  der Quellenmodus startet keine Performancekampagne.
- Streng gebundene Backend-/Precisionprojektion und vertragliche Raw-/NPY-
  Eingabeauflösung über den gemeinsamen bestehenden Arrayloader. Explizite
  Alias-, Artefakt-, Slot-, Format- und Endpunktkonflikte bleiben Fehler.
- H8-Dreistufen-Runtime mit tatsächlicher HEF-/Enginebindung und vorbereitetem
  Input außerhalb des Messfensters. Der historische Canary bleibt separat;
  Oracle- und Fast-Completionnachweise verwenden ihre bestehenden Prüfer.
- Normale Energieplanung, Quellenabschluss, Besitzer-/Quarantäneprüfung und
  begrenzte Fortsetzung mit erhaltenen Teilreplikaten. Eine ausdrücklich
  autorisierte H8-Limiterhöhung zählt frühere Versuche mit; verschachtelte
  Retries vervielfachen das Budget nicht. Ein neuer unbestätigter Quellenabschluss
  sperrt erneut. Adressen stammen aus den exakt gebundenen Originalressourcen,
  nicht aus privaten Quellcodekonstanten.

## Wissenschaftliche und technische Grenzen

Technisch abgeschlossene Messung, Qualityentscheidung, semantische Validierung
und wissenschaftliche Claimfreigabe sind getrennt. `accuracy_loss` und
`INCONCLUSIVE` werden nicht zu PASS umgedeutet. Development/screening,
postgeplante Zusatzkohorte und eingeschränkte Runtimevergleichbarkeit bleiben
ausgewiesen. Diese Veröffentlichung erfüllt oder ersetzt keinen Hold-out-Gate
für die weiterhin getrennte 3.0.0-Kampagne.

Die Quellenabnahme umfasst gezielte lokale Regressionen, normalen Paketbau,
Manifest-/Archivprüfung, Installation und CLI-/GUI-Versionsauflösung ohne Job.
Sie ist keine erneute Hardware-, Quality-, Compiler- oder Energieabnahme.
Die abgeschlossenen Messungen und ihre ursprünglichen Quellen werden separat
im Ergebnisarchiv berichtet; Softwaretestzahlen sind kein Hardwarebeleg.

## Lieferung und Installation

`scripts/build_source_manifest.py` und `scripts/build_source_release.py`
erzeugen das vollständige geprüfte Quellarchiv. Installation mit
`scripts/update_source_release.sh RELEASE.zip TOOL_DIR`; der vorhandene
stdlib-only Editable-Installer erhält Dependencies und lokale Profile.
Keine SDK-/Treiberaktualisierung, kein Force-Rebuild.

43 lokale Profile, Venvs, Modelle, Caches, Messrohdateien und private
Arbeitskopien gehören nicht zur öffentlichen Quellenlieferung. Neue
Dokumentationspfade verwenden `$HOME`; Testnetzadressen sind synthetisch.
Bereits veröffentlichte historische Dokumente und Hashfixtures bleiben
unverändert, einschließlich ihrer damaligen Beispielpfade. Die aktuelle
öffentliche Lieferung ist mit Git-Quellen und Manifest bytegleich; es gibt
keine separat umgeschriebene öffentliche Quellfassung.

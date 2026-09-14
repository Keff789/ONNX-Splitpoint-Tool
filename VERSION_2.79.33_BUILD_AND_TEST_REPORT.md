# v2.79.33 – Sourcevertrag und Abnahmegrenzen

Version `2.79.33`, Build `v2.79.33-force-profile-scope-safety`.

Ausgangsstand ist v2.79.32 FIX2, Source-SHA256 `aa901238a25889d742c24e4b77301bb94c600f13b0bccf67d961e8fe760c9043`. Dieser Stand wurde auf Smartmirror2 mit 1.988 Pflichtprüfungen ohne Fehler oder Skips abgenommen. Die sechs im Force-Audit gelieferten installierten Module stimmen bytegenau mit dieser vollständigen Basis überein. Die einzige bekannte Abweichung zum früheren Q3-Vorlauf unter den dort dokumentierten 171 kritischen Modulen ist das später in FIX2 reparierte Suite-Statusmatrix-Template.

## Implementierungsumfang

- AP0: kleine Originalfixtures, tatsächlicher historischer Resolverreplay und klare Trennung von historischen Befunden und synthetischen Gegenproben.
- AP1: gemeinsame typisierte Boolverarbeitung vor Widgets, Resolver, Profil- und CLI-Verwendung.
- AP2: kooperierende Registrywriter prüfen die geladene Revision erneut innerhalb einer lokalen Dateisperre und ersetzen atomar; Konflikte bleiben sichtbar.
- AP3: effektive Force-/Integritäts-/Classification-Werte und deren Quelle; neue Zustimmung für einen neuen Force-Start oder Resume, außerhalb der Cacheidentität.
- AP4a: getrennte sichtbare Grid-Zeilen für den DeepX-Legacyhinweis; explizite Profile bleiben erhalten.
- AP4b: Generic Full erhält die exakt belegte Setupzuordnung; widersprüchliche oder mehrdeutige Identitäten bleiben ungültig. Keine Änderung gemessener Endpunkte oder Vendor-FPS zu Pipeline-FPS.
- AP5: Runtime-/Buildgrenzen für Wiederverwendung und exakte negative Evidenz prüfen; bestehende Retention erläutern; G0-Diagnose sicher integrieren; Sourcearchive und Upgrade vollständig prüfen.

Die bestehenden Native-Full-Fehlerpfade, Primärexceptions, Compilerprioritäten und FIX2-Statusmatrixregressionen bleiben Pflichtprüfungen. `COMPILE_INFEASIBLE` und `TRANSIENT_INFRASTRUCTURE` bleiben getrennte Ergebnisse. Es entstehen keine neuen Hash-/Seal-/Attestorsysteme, Cacheidentitäten oder Messparallelitäten.

## Ausführungsnachweise

Der zu dieser Source gehörende Delivery-Prüfbericht und die originalen JUnit-/Installerdateien enthalten die tatsächlich ausgeführten Prüfungen, deren Zahl, den Source-SHA256 und die Erhaltungskontrollen. Dieses Dokument nimmt keine erfolgreiche Ausführung vorweg. Der Installer darf nur PASS melden, wenn alle erforderlichen Prüfungen bestanden sind; fehlende Abhängigkeiten, Pflichtskips und Xfails blockieren.

Die 32 Gruppen des Nutzerplans werden im Delivery-Abgleich einzeln ihren Nachweisen zugeordnet. Offline simulierte Compiler-/TensorFlow-Grenzen sind ausdrücklich gekennzeichnet. Echter Wiederverwendungsnachweis auf dem Zielbestand, G0/G1 auf GPU, neue MobileNet-/YOLO26-Qualität, B500/B5000 und finale Energie haben den Status `NOT_RUN`, solange entsprechende neue Hardwareevidence fehlt.

Anleitung: `TESTANLEITUNG_2.79.33.md`.

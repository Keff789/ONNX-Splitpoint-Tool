# Build- und Testbericht 2.79.25

Build-ID: `v2.79.25-deepx-gpu-quality-followup`

Basis ist die ausgelieferte 2.79.24. Anlass sind die erneuten Abnahmen
MobileNet b027 und YOLO11l b003 vom 06.09.2026 sowie die separat nachgewiesene
GPU-Rechenfähigkeit der GTX 1080 Ti mit dem vorhandenen cu126-Overlay.
Original-Debug-Packs und bisherige Ergebnisse werden nicht verändert.

## Begrenzter Umfang

- Die DeepX-Compilerumgebung kann das bereits vorhandene cu126-Overlay
  ausschließlich für Compiler-Prüfung und DX-COM-Unterprozesse verwenden.
  Die GUI und andere Backends übernehmen dessen Python-Pakete nicht.
  Es werden keine Pakete installiert und keine Treiber geändert.
- Beim Hailo-Full-Transfer werden die im bestehenden Receipt referenzierten
  Pfade der ausgewählten Generation als interne Tar-Hardlinks auf die
  öffentlichen regulären HEF-/Receipt-/Meta-Dateien erhalten. Das HEF wird
  nur einmal übertragen; alte Generationen werden nicht mitgenommen. Der
  bestehende Publisher-Validator und alle Inhaltsbindungen bleiben bestehen.
- H8-Detection übernimmt die bestehende Fast-Oracle-Evidenz durch die
  Hostvalidierung. Der Vertrag, sein Siegel und die referenzierten Ergebnisse
  werden inhaltlich geprüft. Fehlende oder veränderte Evidenz wird nicht
  akzeptiert. Qualitätsgrenzen und Native-Hotloop bleiben bestehen.
- Der Energieimport übernimmt den tatsächlich geprüften und angewandten
  FS-Skalierungsbeleg. Die zwölf vorhandenen A/B-Energiezeilen wurden zuvor
  wegen eines fehlenden optionalen älteren Methodenmanifests fälschlich als
  unkalibriert ausgeschlossen. Anwendbarkeit, verifizierte Evidenz-SHAs,
  positiver endlicher Faktor, tatsächliche Anwendung und Wiederholungszahlen
  bleiben verbindlich. Der optionale externe Methodennachweis bleibt separat.
  Gemessene Werte, Quality-Entscheidungen, Claim-Flags und Collector sind
  unverändert.
- Das vorhandene Top-Level-Opt-in
  `energy.include_raw_parquet_in_debug_pack` für reguläre Native-Energie-
  Roh-Parquets wird vom Debug-Pack berücksichtigt. Nur kanonische Collector-Parquets werden in
  sortierter Reihenfolge aufgenommen, begrenzt auf 64 MiB je Datei und
  512 MiB gesamt. Angeforderte, vorhandene, archivierte und ausgelassene
  Dateien erscheinen im Packmanifest; Auslassungen bei gesetztem Opt-in
  ergeben keinen vollständigen Pack. Standardmäßig bleibt die Aufnahme aus;
  der separate Methodenprobe-Umfang und dessen Budget bleiben unabhängig.
- Releaseidentität, generische Einstiegspunkte, Updater und Softwaregate
  werden gemeinsam auf 2.79.25 aktualisiert. Die verhaltensbezogenen
  Regressionstests aus 2.79.24 und früher bleiben enthalten.

Es entsteht keine neue Datenbank, Cachehierarchie oder Pipelineebene.
`COMPILE_INFEASIBLE` und `PARSER_UNSUPPORTED` werden weiterhin ausschließlich
bei exakt passender Identität wiederverwendet; `TRANSIENT_INFRASTRUCTURE`
bleibt wiederholbar. Artefaktverlust wird nicht durch Abschwächen einer
Identitätsprüfung kaschiert. Tatsächlich fehlende Artefakte dürfen in den
korrigierten Abnahmeprofilen gezielt neu gebaut werden.

## Softwareabnahme

Die gelieferte Source-Prüfsumme, das vollständige Softwaregate und die
isolierte Installation 2.79.24 → 2.79.25 werden nach dem abschließenden
Quellpaketbau in `VERIFICATION_V27925.json` sowie `verification/` dokumentiert.
Der Installationstest verwendet eine eigene temporäre Toolinstallation.
Fünf Testbestände für HEF, Receipt, DXNN, Negativ-Evidenz und ein eigenes
Profil müssen inhaltlich erhalten bleiben; die `.venv` darf nicht ersetzt
werden. Eine echte Geräteabnahme ist daraus nicht ableitbar.

## Grenzen des Nachweises

Der vom Nutzer dokumentierte cu126-GPU-Matmul und DX-COM-Import bestätigen
die getesteten Operationen. Ein echter vollständiger DX-COM-GPU-Compile in
der lokalen Vendorumgebung bleibt noch erforderlich. Eine CPU-/GPU-
Zeitrelation wurde nicht gemessen und wird nicht erfunden.

Die Standard-TRT-Splitreferenz erzeugte in den neuen Abnahmen außerdem
eine Part1-Engine (A etwa 70 Sekunden, B etwa 38 Sekunden). Die bisherige
Preflight-Matrix erfasst bei TRT Full und Part2, aber diesen Standard-Part1-
Build noch nicht vollständig. Dieser Anzeigeumfang wird in 2.79.25 nicht
behoben; eine vollständige Preflight-Abdeckung aller TRT-Stufen wird nicht
behauptet.

Erfolgreiche technische Bindung erklärt vorhandene Accuracyverluste nicht
für behoben. Die Full-/Split-Vergleiche müssen diese zunächst lokalisieren.
Die vorhandenen Performance-/Energy-Befunde ersetzen keine vollständige
Kampagne und keine externe absolute Leistungsreferenz.

```text
REAL_DXCOM_GPU_BUILD=NOT_RUN
REAL_HARDWARE_ACCEPTANCE=NOT_RUN
REAL_ACCURACY_RECOVERY=NOT_RUN
REAL_ENERGY_REFERENCE_VALIDATION=NOT_RUN
```

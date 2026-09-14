# Originalbelege für AP0 (v2.79.33)

`force_origin_audit_20260909T095730Z_ni3utrwy.zip` ist die unveränderte Originaldatei
aus Q1. Sie enthält Rohregistry, ausgewähltes Quell-/Effektivprofil, sechs
installierte Quellmodule und die ausgelesenen Felder von 16 historischen Profilen.
Die vollständigen 16 historischen Profildateien liegen damit nicht vor. Es sind
keine Modelle, Cacheartefakte oder Datenbilder enthalten.

`run_modes_v30_original.py` und `REPLAY_FORCE_ORIGIN.json` wurden bytegleich aus
`Force_Ursprung_v27932_Nachweise_2026-09-09.zip` übernommen. Der historische
Resolver und der originale v32-Resolver im Auditarchiv werden ausschließlich in
diagnostischen Tests im Speicher geladen. Produktive Aufrufer nutzen diese
Dateien nicht. Ein grüner Test, der den alten Stringfehler reproduziert, ist
kein Nachweis dieses Fehlers als historische Ursache und keine v33-Freigabe.

`baseline_source_manifest.json` ist das originale Manifest des vollständig
geprüften v32-FIX2-Source-ZIPs. `baseline_audit.json` dokumentiert die Prüfung
aller 1.532 Payloads und den Bytevergleich aller sechs Q1-Module.
`Q3_critical_module_sha256.json` enthält die vorhandenen Modulidentitäten aus
`Q3/run/run_manifest.json`: 170 von 171 stimmen mit FIX2 überein. Der einzige
Unterschied ist die abgenommene FIX2-Korrektur des Statusmatrix-Templates.
Diese vorhandenen Release-/Runbindungen sind keine neue produktive
Cacheidentität oder zusätzliche Hashinfrastruktur.

Die Hashgegenprobe gilt ausdrücklich für v30 → den tatsächlich auditierten
v32-Resolver: 16/16 historische Bindungen passen; der Delta besteht ausschließlich
aus den drei Mean/Std-Defaultfeldern. Eine spätere bewusste Beschreibungskorrektur
in v33 ändert diesen historischen Befund nicht. Der allererste Schreiber der
Force-Werte bleibt unbekannt; weder ein manueller Klick noch ein alter
Stringfehler oder ein konkurrierendes Panel sind dadurch belegt.

Die Tests vergleichen die Originalbytes vor und nach dem Replay und prüfen die
bereits im Audit aufgezeichneten Dateiidentitäten. Sie führen keine Hardware-,
Compiler-, SSH- oder produktiven Konfigurationsschreibaktionen aus.

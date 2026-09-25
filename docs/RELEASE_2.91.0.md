# Release 2.91.0

Version `2.91.0`, Build-ID und annotierter Tag `v2.91.0` veröffentlichen den
kumulierten AP00–AP08-Stand einschließlich der gezielten Quality-/Transfer-
und Wiederanlaufkorrekturen. Smartmirror2 bleibt der x86-Controller. Dieser
Software-Release verändert keine gespeicherten Ergebnisse und erteilt keine
neue Hardware- oder wissenschaftliche Gesamtfreigabe.

Die optimierte COCO-Statistik und die CPU-gerechte Transferzulassung bleiben
erhalten: ein ausführbarer kleiner Transfer wartet nicht allein wegen eines
älteren, aktuell nicht ausführbaren großen CPU-Auftrags. Ressourcen-, Capture-,
DUT-, Quellen- und Cancelgrenzen gelten weiter. Der bestehende Single-Tensor-
Schalter steuert die Generic-Auswahl; Native verwendet deren kompatible
Teilmenge ohne eigene Quote oder Nachrücken. Angeforderte Full-Baselines bleiben
auch bei leerer Native-Split-Teilmenge erhalten. Vorhandene Native-Full-, Resume-
und explizite Recoverykorrekturen sind Teil dieses Standes.

Die Energie-Kampagnenkonfiguration prüft `max_transport_failures` durchgängig
als positive Ganzzahl. Booleans, Null, nichtpositive und nichtganzzahlige Werte
werden abgewiesen. Explizite Kampagnen-Retries erreichen auch den vorhandenen
äußeren Invalid-Repeat-Pfad im direkten Mess-CLI. Es gibt keine zusätzliche
Retry-Schleife. Standardwerte anderer Profile und der globalen Energieconfig
bleiben unverändert.

Die neue private 20er-Profilkopie setzt `max_retries=2` und
`max_transport_failures=20`: drei logische Replikate, je höchstens drei Versuche,
insgesamt höchstens neun reservierte Preflight-/Aufnahmeketten pro Zeile. Der
erste vollständig gültige Versuch zählt. Gültige Replikate werden nicht erneut
aufgenommen; ungültige Versuche bleiben als Belege erhalten.
Direkter CLI-Wiedereintritt kann auch mit einem frischen Ausgabepfad kein
bereits gültiges Kampagnenreplikat neu reservieren; die vorhandene Budget-
Admission sperrt mit `energy_repeat_already_valid`. Der normale Resume überspringt
vollständig verifizierte Zeilen. Dies ergänzt keine Teilresume-Rekonstruktion.
Der kumulative
Quellenzähler bleibt über erfolgreiche Retries, Modelle und Resume bestehen;
der 20. Transportfehler stoppt die Quelle. 20 ist ein begrenztes Betriebsbudget,
kein nachgewiesenes optimales Limit. Unbestätigtes Quellenende, Cancel und echte
konkurrierende Arbeit bleiben unabhängig vom freien Budget blockierend.
60 s, FS/command, Kalibrierung, Rate, Work Units, Backoff und technische
Aufnahmeprüfungen werden nicht gelockert.

Die Nativevalidierung bindet zentrale Qualityevidenz vor der abschließenden
Claimprüfung. Zuvor konnte die vorgezogene Prüfung einen semantisch gültigen
ResNet-Claim wegen der zu diesem Zeitpunkt noch fehlenden Bindung unwiderruflich
auf false setzen. Ein unveränderter gespeicherter ResNet-Fall erreicht jetzt
mit exakter Bindung `quality_claim_result_verified=true` den Energieplan und
Reporter. Die vorhandene Screeningrolle hält den wissenschaftlichen
Energieclaim weiterhin gesperrt (`screening_energy_policy_nonclaimable`).
Der konkrete negative H10-Fall und eine wirklich fehlende zentrale Bindung
bleiben gesperrt. Technische Aufnahmegültigkeit, Modellergebnis, Paarung,
Matrixvollständigkeit und wissenschaftliche Claim-Eignung bleiben getrennt.
Die lokale Wiedergabe schreibt ausschließlich separate Testausgaben; es gibt
keine Neuberechnung oder Korrektur der alten Ergebnisdateien.

Die explizite Ressourcen-Recovery unterscheidet eine tatsächlich ausgeführte
Betreiberhandlung von einer ausdrücklich autorisierten Neuversuchsfreigabe.
Beide verwenden dieselben vorhandenen Ownership-/Lease-/Cleanupprüfungen.
Es gibt keine automatische Quellenfreigabe, keinen erfundenen Reset oder
Idle-ACK. Die alten STOP-, Mess-, Budget- und Quarantänebelege bleiben erhalten.
R1-Hardcrash-Recovery bleibt partiell; der Release verspricht keine allgemeine
Wiederherstellung nach beliebigem Controllerabbruch. Die bekannte fehlende
AP06-Brokeraktivierung bei Global-/Modellbarrieren wird nicht als behoben erklärt.

Der abgeschlossene 7×1-Integrationslauf vom 24.09.2026 ist technisch weitgehend
ausgeführt: 61 erfolgreiche Nativezeilen, zwei bekannte Buildausschlüsse,
75 zentrale Qualitätsauswertungen (62 referenznah, 13 Genauigkeitsverluste),
48/61 vollständig verifizierte Energiezeilen. Zwei physische Versuche waren
ungültig; nach dem zweiten Transportfehler blockierte das damalige H10-
Quellenbudget. Die beiden Sampleverlustketten meldeten bestätigten
Quellenabschluss; sie sind kein Fall des historischen fehlenden G2-Protocol-END.
Der alte Run bleibt unverändert, einschließlich der 13 fehlenden Energiezeilen.

Bekannte Grenzen gelten weiterhin für die konkreten Artefaktbindungen:
H10 YOLO26m/b398 und YOLO26s/b364 waren technisch ausführbar, lieferten aber
ungeeignete Detectionausgaben. Reparierte Diagnosebindung bedeutet keine
reparierte HEF-Numerik. Der [vorhandene H10-Befund](https://github.com/Keff789/onnx-splitpoint-results/blob/main/diagnostics/v2.83_h10_output/H10_OUTPUT_BEFUND.md)
bleibt maßgeblich; keine neue Numerikdiagnose, keine Threshold-/NMS-/Precision-
oder Kalibrierungsänderung. H8-Negativbelege für diese Grenzen bleiben
Buildausschlüsse und werden nicht durch erneute Kaltbuilds übergangen.
Keine globale Modell-/Backendsperre und kein Ersatz eines vorab ausgewählten
negativen Falls durch einen besseren Split. Ein technischer Energieabschluss
begründet keinen gleichwertigen Completed-Detection-Effizienzclaim aus Nullausgaben.

Die neue Profilkopie `Thesis_20Splits_N5000_B1000_20260925.yaml` ist für eine
einzige stratifizierte Kohorte vorgesehen: sieben Modelle, 20 Generic-Grenzen
je Modell, Single-Tensor-Schalter aus, kein zusätzlicher Auditsatz und kein
Backend-Backfill. Native nutzt nur kompatible Grenzen derselben Auswahl.
N5000/B1000, Seed20260710, optimized_coco_v1 mit vier Workern/einem Request,
Block256 und Checkpoints; Native1000/100/3, Native-Energie60s×3 mit Full-Baselines.
Global-/Modellbarrieren und ein Uploadslot bleiben erhalten. Hailo balanced/
Opt1/B500/Batch8 und DeepX EMA/Opt0/B500 bleiben unverändert; Force aus,
`keep_artifacts=true`, gültige Artefakte wiederverwenden, fehlende regulär bauen.
Das gebundene normale Profilsnapshot verhindert einen stillen Rückfall auf
B5000 oder Legacy. Die Rollen bleiben development, Claimscope evaluated_matrix.
Die neue Kampagne wurde nicht gestartet; lokale Projektionen sind keine
Laufbarkeits-, Builddauer- oder Energiezeilengarantie.

Gezielte Software-, reale GUI-Profil- und Installedprüfungen sowie der begrenzte
Quality→Native→Energie→Reporter-Befund werden im [Arbeitsstand](ARBEITSSTAND.md)
getrennt von historischer Hardwareevidenz dokumentiert.

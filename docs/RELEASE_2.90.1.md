# Release 2.90.1

Version `2.90.1`, Build-ID und annotierter Releasetag `v2.90.1` sichern den
geprüften kumulierten R9L-Produktstand einschließlich Energie176-Korrekturen.
Der Releaseabschluss ändert Metadaten und aktuelle Releaseassertions; er startet
keine Geräte, Builds oder Messungen. Compiler-/Cacheverträge und wissenschaftliche
Akzeptanzkriterien bleiben unverändert. Smartmirror2 ist der x86-Controller.

Der R9L-Standardlauf enthält 63 technisch vollständige Native- und 77 zentrale
Qualitätszeilen sowie 61 vollständige Energiezeilen. Die separate
Energie176-Nachabnahme ergänzt zwei TRT-Full-Energiezeilen mit 6/6 gültigen
logischen Wiederholungen, sechs physischen Versuchen und keinem Retry. Das ist
zusammengesetzte Evidenz, kein rückwirkend ununterbrochener 63/63-Energielauf.
Originalergebnisse und historische Ergebnisheader bleiben unverändert.

Die historische Transportursache bleibt offen; ein PCAP fehlt. Technisch gültige
Qualitätsverluste sind von Laufzeitfehlern und der statistischen 95%-Unsicherheit
zu unterscheiden. Kurze Screeningmessungen belegen keine Langzeitstationarität.
Weder ein einzelner Split noch die neue Releasebezeichnung begründen allgemeine
Ranking-, Thesis- oder neue Hardwareabnahmeaussagen.

Die private Finalprofilkopie wird aus dem normalen Benutzerprofil und den
bestehenden Produktresolvern aufgelöst, ohne Laufanlage. Final Quality (Standard+)
verwendet 5000/5000 Validierungsbilder, 5000 Bootstrap-Wiederholungen und
B500-Kalibrierung; Nativeframes/Warmup/Wiederholungen sind 1000/100/3. Sieben
Modelle, drei Setups und ein akzeptierter Split pro Modell bleiben erhalten
(`stratified_windows`, Shortlist 2, Suchpool `auto`). Native Full/Split und
Native-Energie sind an, Generic-Energie und Force aus. Vorhandene gültige
Artefakte werden wiederverwendet; fehlende dürfen erst beim späteren Lauf normal
gebaut werden.

Das Benutzerprofil enthält ausdrücklich 1 s Native-Energie-Solldauer. Dieser
Wert wird unverändert mit seiner Herkunft festgehalten, nicht aus dem
Energie176-CLI-Rezept übernommen. Der Hardwarestandard von 60 s ist deshalb
nicht wirksam. Final Quality macht diese Energieeinstellung nicht zur langen
Finalmessung. Bestehende Aufnahmeableitung und Fehlerschutzregeln bleiben
unverändert; es gibt keine neue Dauergrenze oder automatische Verkürzung.

Regulär gelten drei logische Energiereplikate, höchstens ein Retry je Replikat,
5 s Backoff und Quellenfehlergrenze 2; ungeklärter Quellenabschluss sperrt sofort.
Der erste gültige Versuch zählt. Die selektive Energie176-CLI-Freigabe von zwei
Retries und Quellenfehlergrenze 3 wird nicht zur GUI-/Finalprofilpolicy.

Lokale Release-, Installed-, Mirror- und Konfigurationsprüfungen werden getrennt
von der bestehenden realen Abnahme im [Arbeitsstand](ARBEITSSTAND.md) erfasst.
Private Profile, Liveadressen und Laufbelege gehören nicht zum Git-Release.

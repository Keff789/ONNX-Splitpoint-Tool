# v2.79.29 – DeepX-Full ohne unnötige OpenCV-Abhängigkeit

Build: `v2.79.29-deepx-prepared-full-without-opencv`.

## Änderung
Nur `_run_deepx_prepared_feed_benchmark` im Benchmark-Template verwendet
Pillow statt cv2 zum Lesen der Originalbildabmessungen. Das ist dieselbe
Bildbibliothek wie im unveränderten `native_full_input`-Vorbereitungspfad.
Die gespeicherte Bildorientierung wird benutzt; kein EXIF-Transpose.
Defekte Bilder bleiben `image_read_failed`, inklusive konkreter Fehlermeldung.
Das Decoder-Modul, Float32-Toleranz 2^-23, NMS, Qualitätsmargen und numerische
Eingabevorbereitung sind unverändert. Keine neuen Pakete oder Cacheänderungen.

## Abnahme
Die beiden bisherigen v28-Prozesstests bleiben im Gate. Zusätzlich wird der
v29-Worker mit blockiertem OpenCV-Import und echtem aufgezeichnetem Tensor
getestet, inklusive Fehlerfall -0.01. Nur die DXRT-Engine ist simuliert.
Die vollständige bisherige Verhaltenstestliste bleibt erhalten; nur die
versionsgebundene Release-Identitätsprüfung wird auf v29 aktualisiert.
Siehe aktuellen Prüfbericht und Testprotokolle im Lieferbundle für Ergebnisse.

## Hardware
Echter DX-M1, B500, FPS und Energiemessung: NOT_RUN in der Buildumgebung.
Nach `INSTALL_ACCEPTANCE=PASS` die begrenzte Full-Kurzprobe starten.
Ein Einzelbild-PASS ist keine Modellqualitätsabnahme.
`COMPILE_INFEASIBLE` und `TRANSIENT_INFRASTRUCTURE` bleiben unverändert getrennt.

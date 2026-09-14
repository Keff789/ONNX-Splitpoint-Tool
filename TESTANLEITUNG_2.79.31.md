# Testanleitung 2.79.31

Build-ID: `v2.79.31-complete-set-quality-integration`

Releasekandidat mit offenen Hardwaregates (`hardware_pending`, neue Geräteausführung `NOT_RUN`). Der endgültige Prüfbericht und die JUnit-Dateien liegen im Lieferbundle. Keine Abnahme aus simulierten Deviceaufrufen ableiten.

1. GUI schließen, `bash install_v27931_and_collect_acceptance.sh` aus dem Bundle ausführen. Der Installer erhält Venv, eigene Profile, Modelle und historische Evidence. Er startet keine Hardwaremessung und installiert keine Testabhängigkeiten nach. Eine fehlende Pflichtabhängigkeit blockiert sichtbar.
2. H1b: `bash run_terminal_closure_smoke_v27931.sh`; fast/strict sowie erwartete späte Manipulation werden mit dem installierten Produktionsabschluss geprüft.
3. H2: `bash run_deepx_full_workflow_smoke_v27931.sh`; vorhandenes YOLO11l Full, ein begrenztes Replikat über den normalen Native-Runner. Das ist kein B500-/Energie-Pass. Der bisherige D-Run bleibt reine Quelle.
4. `bash run_complete_set_replay_v27931.sh`: unveränderter historischer Umfang; 63 vorhanden, 55 erfolgreich, acht erfolglos, null fehlend. Energie: 55 Zeilen/165 Replikate, keine neue Kampagnenfreigabe.
5. Weitere enge Gates stehen im Bundle: H3 Hailo8 YOLO11l b062 und YOLOv7 b044; H4 DeepX b062/b398/b364; H5 Hailo10-Rohprobe; H6 produktives Mean/Std-Routing und anschließende frische zentrale Qualität. Kein neuer Complete-Set-Langlauf vor Klärung dieser Gates.

Bestehende Grenzen: `COMPILE_INFEASIBLE` bleibt exakte negative Evidence; `TRANSIENT_INFRASTRUCTURE` ist kein dauerhafter Modell-Ausschluss. Quality-Margen, Seeds, Geometrie und Energie-Messmethoden bleiben unverändert. R1/R2 sind historische Gerätebelege, keine v31-Hardwareabnahme.

Offline: `ORT_DISABLE_TELEMETRY=1 PY=/pfad/test-venv/bin/python bash scripts/run_v27931_small_acceptance.sh --report /ausserhalb/source/acceptance.json`. Das Gate verlangt reale ONNX/ORT/COCO/GUI-Importabhängigkeiten und akzeptiert keine Skips/Xfails.

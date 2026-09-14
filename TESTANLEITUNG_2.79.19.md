# Testanleitung ONNX-Splitpoint-Tool 2.79.19

Build-ID: `v2.79.19-calibration-warning-evalrun-closure`

Diese Version ist eine kleine Korrektur der Full-System-Kalibrierung und der
aus dem fehlgeschlagenen Complete-Set-Lauf abgeleiteten EvaluationRun-Probleme.
Sie ändert weder die Rankingmethode noch die Pipeline-FPS-Definition und fügt
keine neue Hash-, Manifest-, Signatur- oder Versiegelungsebene hinzu.

## 1. Offline-Acceptance

Im installierten Tool-Verzeichnis:

```bash
cd ~/ONNX-Splitpoint-Tool
bash scripts/run_v27919_small_acceptance.sh
```

Erwartung:

```text
PASS v2.79.19 smoke
PASS v2.79.19 small acceptance (hardware calibration and real EvaluationRun: NOT_RUN)
```

Der Lauf prüft ohne Hardware insbesondere:

- Paket-, Release-, Workflow- und Build-Identität 2.79.19;
- Kalibrierungswarnungen gegenüber weiterhin harten technischen Blockern;
- Remote-Staging von `hailo_attempt_receipts` und `hailo_timeout_policy`;
- die drei duplizierten Quality-Vertragsfelder in Request und Candidate;
- Required Scope aus den final akzeptierten Cases nach Reject/Backfill;
- eng begrenzte Bereinigung absichtlich ersetzter Report-Indexeinträge;
- Classification-Split-Policy und Orchestrierung für MobileNet und RegNet;
- Hailo-8-ResNet mit quantisiertem HEF-Speicher und FLOAT32-VStream-Vertrag;
- DeepX-Fehlerstatus nach erfolgtem Dispatch sowie konkrete Bindungsgründe für
  ein fehlendes oder widersprüchliches `input_contract.model_id`.

## 2. Identität prüfen

```bash
cd ~/ONNX-Splitpoint-Tool
PYTHONPATH= PYTHONHOME= .venv/bin/python -I -B - "$PWD" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

print("VERSION=" + package.__version__)
print("RELEASE=" + package.__release__)
print("BUILD_ID=" + package.__build_id__)
print("WORKFLOW_VERSION=" + WORKFLOW_VERSION)
PY
```

Erwartung:

```text
VERSION=2.79.19
RELEASE=2.79.19
BUILD_ID=v2.79.19-calibration-warning-evalrun-closure
WORKFLOW_VERSION=v2.79.19-calibration-warning-evalrun-closure
```

## 3. Full-System-Kalibrierung prüfen

Der sichere Hardwareablauf aus 2.79.18 bleibt unverändert:

1. Elektronische Last an `9V_20V_IN` hinter R16 gegen GND anschließen und auf
   **Output OFF / 0 A** stellen.
2. Einen bereits absichtlich ausgeschalteten Jetson ausdrücklich bestätigen
   oder einen SSH-bereiten Jetson kontrolliert durch das Tool ausschalten
   lassen. M.2 darf in beiden Fällen nicht geschaltet werden.
3. Die Phasen `idle_before`, `load_0.5A`, `idle_between`, `load_1A` und
   `idle_after` vollständig aufnehmen. An beiden Lastpunkten Ist-Strom und
   Ist-Spannung eingeben.
4. Ergebnisdialog prüfen und bewusst speichern oder verwerfen.

Die lineare Korrektur bleibt durch den Ursprung gebunden:

```text
factor = sum(dP_i * P_ref_i) / sum(dP_i²)
```

Die folgenden Abweichungen erscheinen als **Plausibility warnings (saving
remains allowed)** und dürfen ein technisch gültiges Ergebnis nicht mehr
blockieren:

- Unterschied der beiden Punktfaktoren über dem konfigurierten Richtwert;
- Faktor außerhalb des engeren erwarteten Konfigurationsbereichs, aber
  innerhalb der breiten harten Grenze;
- Ist-Strom weiter vom nominellen Sollpunkt entfernt als die
  Plausibilitätstoleranz.

Weiterhin hart blockiert werden unter anderem:

- nicht endliche oder nicht positive Mess-/Referenzwerte;
- zu kleiner gemessener Leistungssprung;
- zu hohe Idle-Drift;
- singulärer Fit oder Faktor außerhalb der breiten harten Grenze;
- nicht bestandene Acquisition-Integrity- oder Energy-Gates.

Eine Warnung ändert keinen Messwert und keinen Faktor. Sie dokumentiert nur,
dass die beiden realen Kalibrierpunkte nicht ideal übereinstimmen.

Kurzprüfung der gespeicherten Evidenz:

```bash
jq '{
  scale_factor,
  warnings,
  quality_gate: {
    pass: .quality_gate.pass,
    reasons: .quality_gate.reasons,
    warning_reasons: .quality_gate.warning_reasons
  }
}' /ABSOLUTER/PFAD/full_system_input_scale_calibration.json
```

Bei einer nur plausibilitätsbezogenen Abweichung müssen `pass: true`,
`reasons: []` und mindestens ein `warning_reasons`-Eintrag erscheinen.

## 4. Begrenzter EvaluationRun vor dem finalen Lauf

Vor dem mehrstündigen Complete-Set-Lauf einen kurzen Run mit mindestens zwei
gültigen Modellen und optional einer bewusst ungültigen Modellzeile ausführen.
Energy nur dann aktivieren, wenn die reale u.RECS-/Idle-Kalibrierung aktuell
und grün ist. Prüfen:

- ein modellbezogener Preflight-Fehler stoppt nicht die gültige Modellzeile;
- der Remote-Prozess importiert das Hailo-Backend ohne
  `ModuleNotFoundError` für `hailo_attempt_receipts` oder
  `hailo_timeout_policy`;
- übertragene/gestartete Zähler der gültigen Zeile sind größer als null;
- `required_run_scope.json` enthält nur die final akzeptierten Cases aus dem
  erzeugten BenchmarkSet, keine verworfenen Kandidaten;
- das Terminal erzeugt `artifact_index.json`, ohne
  `artifact_index_registered_file_missing:reports/prediction_vs_benchmark.csv`;
- ein anderes wirklich fehlendes, nicht durch den Reporter verwaltetes
  Artefakt würde weiterhin hart fehlschlagen.

Für Generic TensorRT Quality müssen Request und Candidate jeweils diese bereits
vorhandenen Werte an der obersten Ebene führen:

```text
decoder_contract_sha256
nms_contract_sha256
quality_record_endpoint_contract_sha256
```

Sie müssen mit `producer_identity` übereinstimmen. Bei Classification dürfen
Decoder/NMS vertragsgemäß leere Strings sein; das Feld selbst darf nicht
fehlen.

## 5. Zusätzliche Modellpfade

- **MobileNet/RegNet Classification:** Ein kurzer Native-Split-Preflight muss
  die Classification-Split-Policy auswählen und darf das Modell nicht allein
  aufgrund seiner Familie als Detection behandeln oder verwerfen.
- **ResNet auf Hailo-8:** Ein quantisiert gespeicherter HEF-Tensor ist zulässig,
  wenn der Runtime-/VStream-Vertrag weiterhin FLOAT32 verlangt und erfüllt.
  Speicherformat und Runtime-Format sind getrennte Vertragsfelder. Fordert der
  Runtimepfad rohe Integerdaten, muss die native HEF-Streambreite dagegen exakt
  passen; ein `UINT16`-Stream darf nicht als `UINT8` etikettiert werden.
- **DeepX:** Ein Fehler bei der lokalen Verarbeitung eines bereits dispatchten
  Remote-Ergebnisses muss als Post-Dispatch-Verarbeitungsfehler erhalten
  bleiben und darf nicht als „nicht dispatcht“ umetikettiert werden. Neu
  erzeugte Einzelverträge müssen `model_id` direkt enthalten. Sowohl ein
  fehlendes als auch ein widersprüchliches `input_contract.model_id` bleibt
  strikt ungültig; die Evidenz muss den konkreten Reason-Code ausweisen.

Alte oder resumierte DeepX-Einzelverträge ohne eigenes `model_id` werden nicht
automatisch umgedeutet. Für die v2.79.19-Prüfung ist ein frischer Run zu
verwenden.

Der reale DeepX-YOLO11-End-to-End-Hardware-Replay ist im Build-Umfeld
`NOT_RUN`. Bis zu diesem Replay ist die DeepX-Korrektur eine softwaregeprüfte
Fehlerstatus-/Diagnosekorrektur, keine Aussage über erfolgreichen
Hardwaredurchsatz oder vollständige E2E-Qualität.

## 6. Statusgrenzen

Die Offline-Acceptance führt keine u.RECS-, Jetson-, Hailo- oder DeepX-
Hardwareaktion aus. Deshalb bleiben bis zur Ausführung auf Smartmirror2:

```text
REAL_FULL_SYSTEM_CALIBRATION=NOT_RUN
REAL_MULTI_HOST_EVALRUN=NOT_RUN
DEEPX_YOLO11_E2E_HARDWARE_REPLAY=NOT_RUN
```

Der zuvor volle Datenträger wurde laut Betreiber bereits bereinigt. Das ist
eine notwendige operative Voraussetzung, aber kein Bestandteil oder
Erfolgsnachweis dieses Source-Releases.

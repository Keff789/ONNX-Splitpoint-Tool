# Hailo GPU-Rechensmoke R1

Stand: 9. September 2026. G0-Diagnosehelfer in v2.79.33, aus dem separat geprüften Rechensmoke R1 übernommen. Die Modellbuild-Policy des Tools bleibt unverändert.

## Zweck und Grenzen

Erster, kleiner Kompatibilitätstest vor einem Hailo-GPU-Modellbuild. In den vorhandenen DFC-Venvs werden nacheinander der SDK-Import sowie vier tatsächliche TensorFlow-GPU-Berechnungen ausgeführt: Matrixmultiplikation, Conv2D, XLA-Matrixrechnung mit Sinus/Exponentialfunktion und XLA-Conv2D. Jeder Output wird synchronisiert und gegen eine unabhängige NumPy-Referenz geprüft. Die Toleranz für diesen synthetischen Float32-Test ist `rtol=atol=3e-4`; sie ist keine neue Modellqualitätsmarge.

Soft-Device-Placement wird abgeschaltet. CPU-Outputs gelten nicht als GPU-PASS. `jit_compile=True` fordert für die XLA-Tests die tatsächliche Kompilierung dieser **synthetischen Kernels**. Keine ONNX-/HEF-/DXNN-Datei wird geöffnet oder erzeugt, kein Modell geparst/optimiert/kompiliert und kein `ClientRunner` instanziiert.

Ein `compute_pass` bedeutet nur, dass diese Operationen in der konkret protokollierten Umgebung funktionieren. Es beweist weder einen erfolgreichen DFC-Modellbuild noch dessen Beschleunigung oder identische Quantisierungsqualität. Danach folgt ein separat freizugebender isolierter Build mit derselben B500/Opt1/Batch8-Recipe und getrenntem Ausgabeordner.

## Start auf Smartmirror2

Aktive Tool-Messungen und Compilerjobs erst beenden; GUI schließen. Der gleiche Workflow-/Plattformlock wird exklusiv für die gesamte Probe gehalten. Ist er belegt, wird **vor Start und vor Anlegen eines Ergebnisordners** `SMOKE_STARTED=NO` ausgegeben. Der Helfer löscht oder leert die Lockdatei niemals. Fremde, nicht am Toollock beteiligte GPU-Jobs müssen ebenfalls beendet sein.

```bash
(
  set -e
  cd "$HOME/ONNX-Splitpoint-Tool"
  bash scripts/run_hailo_gpu_compute_v27933.sh
)
```

Voreinstellungen, aus der vorhandenen Tool-Konfiguration abgeleitete Standardpfade:

```text
~/.onnx_splitpoint_tool/hailo/venv_hailo8
~/.onnx_splitpoint_tool/hailo/venv_hailo10
```

Ein fehlender Pfad wird ausdrücklich gemeldet; keine automatische Installation und kein Ausweichen auf ein anderes Python. Abweichende Pfade können mit `--venv-hailo8 /pfad` beziehungsweise `--venv-hailo10 /pfad` angegeben werden. `--families hailo10h` testet nur die zweite Umgebung.

## Isolation

GPU 0 wird **nur im gestarteten Kindprozess** ausgewählt (`--gpu` erlaubt eine andere einzelne Nummer oder GPU-UUID). Auch ein geerbtes `CUDA_VISIBLE_DEVICES=-1` wird damit ausschließlich im Diagnosekind auf `0` gesetzt und in `environment_overrides.json` sichtbar dokumentiert. `ONNX_SPLITPOINT_HAILO_ALLOW_GPU=1` gilt nur dort. Die reguläre Hailo-CPU-Policy des Tools bleibt unverändert.

PYTHONPATH/PYTHONHOME werden im Kind entfernt, damit insbesondere kein DeepX-PyTorch-Overlay hineingelangt. Venv-PATH und temporäre CUDA-/XDG-Caches werden nur für das Kind gesetzt. **Kein CUDA-Toolkitwechsel**: CUDA_HOME, CUDA_PATH, LD_LIBRARY_PATH und XLA_FLAGS bleiben für die Diagnose unverändert und werden protokolliert. Das kann einen bestehenden Toolkit-/XLA-Konflikt bewusst reproduzieren. Kein sudo, pip, Netzwerkdownload, SSH, Treiberwechsel oder globales Environment-Update.

Der Controller schreibt nur in den eigenen Diagnoseordner; zusätzlich wird die existierende Tool-Lockdatei geöffnet, falls nötig erstmals angelegt. Python-Bytecode und Core-Dumps werden nicht geschrieben. Framework-/Treiberinitialisierung ist echte lokale Rechenaktivität; unabhängige Messungen dürfen deshalb nicht parallel laufen. CUDA-Caches und Temporärdateien liegen im Diagnoseordner und werden nicht ins Evidence-ZIP übernommen.

## Laufzeit und Ergebnis

Pro DFC-Umgebung maximal **180 Sekunden**, danach gegebenenfalls fünf Sekunden TERM-Frist und fünf Sekunden Nachlaufkontrolle. Ausgabe etwa alle 15 Sekunden. Beide Umgebungen werden streng nacheinander geprüft; ein normaler fachlicher Fehler der ersten verhindert die zweite nicht. Bei unvollständiger Prozessbereinigung wird keine weitere gestartet.

```text
GPU_SMOKE_STATUS=compute_pass | partial | failed | interrupted
MODEL_BUILD=NOT_RUN
MODEL_ACCEPTANCE=NOT_EVALUATED_BY_DIAGNOSTIC
DIAGNOSTIC_ZIP=.../hailo_gpu_compute_r1_....zip
```

Auch ein Absturz oder fehlender SDK-/TensorFlow-Import erhält ein Diagnose-ZIP mit Prozessstatus und Konsolenprotokoll. Nur wenn bereits der gemeinsame Lock belegt ist, entsteht absichtlich noch kein Ergebnis-ZIP.

Das ZIP enthält ausschließlich die eigenen Ergebnis-JSONs, Logs und die verwendeten Skripte; keine Modelle, Tensorpayloads, GPU-Binaries oder ganzen Verzeichnisbäume. Logs können lokale Pfade und installierte Bibliotheksinformationen enthalten.

## Referenzen für die verwendeten APIs

- TensorFlow, GPU-Konfiguration und Memory Growth: https://www.tensorflow.org/guide/gpu
- TensorFlow, `set_soft_device_placement`: https://www.tensorflow.org/api_docs/python/tf/config/set_soft_device_placement
- TensorFlow, `tf.function` / `jit_compile`: https://www.tensorflow.org/api_docs/python/tf/function
- TensorFlow, Laufzeit-Gerätedetails: https://www.tensorflow.org/api_docs/python/tf/config/experimental/get_device_details

API-Unterlagen am 09.09.2026 geprüft. Maßgeblich für die Kompatibilität sind die tatsächlich protokollierten Versionen auf Smartmirror2, nicht eine aus der Dokumentation unterstellte installierte Version.

## Integrationsänderungen gegenüber R1

Der produktive CLI-Einstieg verwendet ausschließlich den bestehenden Workflow-/Plattformlock; eine umleitbare `--lock`-Option wird nicht angeboten. Der einstellbare Timeout ist auf 10 bis 180 Sekunden pro Umgebung begrenzt. Die Prozesskontrolle berücksichtigt auch Container, in denen `/proc` IDs eines übergeordneten PID-Namespace zeigt; eine erfolgreiche Bereinigung erfordert außerdem das tatsächlich beobachtete Ende des Workerprozesses. Die 19 ursprünglichen Offline-Prüfungen wurden in die Release-Tests aufgenommen und um Integrationsprüfungen ergänzt. Ihr simuliertes TensorFlow ist ausdrücklich keine GPU-Hardwareevidence. G1 und die Nachtvorbereitung stehen in `docs/V27933_NACHTLAUF_UND_GPU.md`.

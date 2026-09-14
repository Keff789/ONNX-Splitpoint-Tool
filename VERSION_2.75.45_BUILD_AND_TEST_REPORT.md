# Build- und Testbericht 2.75.45

Release: `2.75.45`

Build/Workflow: `v2.75.45-gui-large-audit-trt-working-set-admission`

## Änderung

Ein 20-/30-Kandidaten-Ranking-Audit konnte in v2.75.44 zwar als gültiges
Evaluation Profile beschrieben und über die GUI angestoßen werden, scheiterte
aber bei der Remote-TensorRT-Cache-Zulassung deterministisch, sobald der
konservativ geplante Zuwachs zusammen mit vorhandenen Caches den nominellen
20-GiB-Wert überschritt. Der Wert behandelte bisher Retention und das aktive
Working Set wie dasselbe Budget. Dadurch war ein wissenschaftlich gültiger
großer Audit praktisch nicht aus der GUI ausführbar, obwohl die separate
physische Speicherplatzprüfung genügend freien Speicher hätte bestätigen
können.

v2.75.45 trennt beide Verträge:

- `retained_cache_budget_bytes` begrenzt aufbewahrte, nicht aktuelle
  Namespaces;
- `current_namespace_bytes + planned_current_growth_bytes` bildet das aktive
  Working Set des ausgewählten Runs;
- `effective_admission_max_bytes` enthält dieses Working Set zusätzlich zum
  Retentionsbudget;
- die vorhandene Cold-/Warm-Speicherplatzprüfung bleibt der maßgebliche
  physische Gate;
- der aktuelle Namespace bleibt stabil und damit für Resume/Reuse verfügbar.

Ein geplanter 30-GiB-Zuwachs darf deshalb bei 20 GiB Retentionsbudget
zugelassen werden, ohne in-budget alte Caches allein für diesen aktuellen Audit
zu löschen. Aktive, unowned, verlinkte, receiptlose oder sonst unsichere
Namespaces bleiben fail-closed geschützt. Alte, eigene, gültig receipted und
inaktive Namespaces bleiben nach dem bisherigen LRU-Vertrag löschbar, wenn das
separate Retentionsbudget oder die maximale Namespace-Anzahl überschritten
wird. Ephemeral Run-Cleanup ist unverändert.

Die GUI macht die tatsächliche Größe eines score-unabhängigen Audits vor einem
Fresh Start sichtbar und verlangt eine explizite Bestätigung. Ablehnen erfolgt
vor Results-Ordner- und Job-Erstellung. Audit-Größen 20 und 30 werden nicht
geclampt; ungültig ist nur ein
`minimum_valid_audit_candidates > audit_size`. Resume und Rerun verwenden den
vorhandenen eingefrorenen Startvertrag und werden nicht als neuer Audit
bestätigt.

Der Profil-Editor besitzt außerdem ein explizites Feld für
`deepx_build.classification_preprocessing`. Neue Profile starten mit
`imagenet_mean_std`; ein altes Profil ohne Feld wird kompatibel als
`current_scale_only` geladen. Explizite Werte überleben Roundtrip und
Run-Mode-Materialisierung.

Der bestehende **Final Quality**-Modus bleibt der **Standard**-Ausführungspfad
mit jeweils 5.000 Classification- und Detection-Validation-Items. Diese
Releaseänderung erweitert weder dessen wissenschaftlichen Umfang noch seine
Quality-Schwellen.

## Release-Vertrag

- Version/Release: `2.75.45`
- Lineage: `v2.75.45`
- Build/Workflow:
  `v2.75.45-gui-large-audit-trt-working-set-admission`
- neue Features:
  - `gui_large_audit_start_confirmation`
  - `audit_minimum_valid_bound`
  - `gui_explicit_deepx_classification_preprocessing`
  - `large_audit_active_trt_working_set_admission`
  - `retained_trt_cache_budget_separation`
  - `large_audit_resume_cache_reuse`
- neue Smokes: `onnx-splitpoint-smoke-v27545` und
  `onnx-splitpoint-smoke-v2-75-45`
- v2.75.44-Smokes, Harnesses, Entry-Points und Dokumente bleiben erhalten.

## Strukturierter Cache-Nachweis

Der Remote-Retentionsnachweis enthält zusätzlich zu den historischen Feldern:

- `retained_cache_budget_bytes`
- `retained_noncurrent_bytes`
- `current_namespace_bytes`
- `active_working_set_bytes`
- `effective_admission_max_bytes`
- `selected_plan_reserve_applied`

Die Zulassung hält folgende Invarianten ein:

```text
retained_noncurrent_bytes
  = eligible_remaining_bytes + protected_bytes

active_working_set_bytes
  = current_namespace_bytes + planned_current_growth_bytes

projected_managed_bytes
  = retained_noncurrent_bytes + active_working_set_bytes

effective_admission_max_bytes
  = retained_cache_budget_bytes + active_working_set_bytes
```

Die maximale Namespace-Anzahl zählt den aktuellen Namespace weiterhin mit; es
gibt keinen zusätzlichen `+1`-Count-Sonderfall.

## Prüfstatus vor dem finalen Manifest-Freeze

Hardwarefrei bestätigt:

- integrierter v2.75.45-Vertrag für 20/30, GUI Accept/Decline und 30-GiB-
  Working-Set mit zweimaligem Reuse: **6/6 bestanden**;
- kombinierter aktueller v2.75.45-Provenance-/Integrationsblock:
  **13/13 bestanden**;
- aktueller v2.75.45-Smoke: **13/13 Prüfungen bestanden**;
- breiter historischer und aktueller Version-/Provenance-Block:
  **190/190 bestanden**;
- vollständiger fokussierter Stage-3-Block des v2.75.45-Small-Acceptance-
  Harnesses: **421/421 bestanden**;
- vollständige Remote-TRT-Retentionsdatei: **30/30 bestanden**;
- kombinierter Retentions-/Pre-Mutation-Block: **40/40 bestanden**;
- fokussierter Audit-/DeepX-Profilblock: **17/17 bestanden**;
- vollständige Profil-Editor-/Run-Mode-/Execution-Plan-Suite:
  **48/48 bestanden**;
- breite DeepX-Preprocessing-/B1000-Regressionssuite: **118/118 bestanden**;
- Python-Compile/Syntax der geänderten Produktionsmodule und Tests:
  **bestanden**.

Die vollständige Small Acceptance, die strikte Source-Manifest-Verifikation
und die reproduzierbare Archivprüfung werden nach dem finalen Manifest-Freeze
auf dem gefrorenen Baum erneut ausgeführt. Bis dahin sind die oben genannten
Zahlen ausdrücklich Komponentenprüfungen, kein fertiger
Release-Freeze-Nachweis.

Alle genannten Läufe waren hardwarefrei. Die Small Acceptance meldet explizit
`SKIP hardware, SSH, DX-COM, TensorRT build, Energy and inference`. Ein echter
Remote-Audit bleibt ein gesonderter Hardware-Akzeptanzlauf und wird nicht durch
den Release-Smoke simuliert.

Archivgröße und Archiv-SHA-256 stehen ausschließlich im externen
Builder-/Handoff-Nachweis beziehungsweise im `.sha256`-Sidecar. Ein Archiv kann
seinen eigenen endgültigen Hash nicht inhaltlich attestieren.

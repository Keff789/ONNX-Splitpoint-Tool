#!/usr/bin/env python3
"""Read-only replay of v2.82 status and exact Full quality evidence transport.

Writes derived results outside the original EvaluationRun. No compiler/runtime,
quality evaluation or energy measurement is started. Original hashes and values
are retained; new request bindings are derived from their exact central result.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts import native_full_baseline_eval_runner as full
from scripts import native_producer_validate_visualize as validator
from onnx_splitpoint_tool.native_job_identity import native_identity_key, known_build_exclusion
from onnx_splitpoint_tool.workflow.evidence_status import project_historical_workflow_status, workflow_completion_projection
from onnx_splitpoint_tool.workflow.runner import _vendor_full_completed_quality_fields_v282
from onnx_splitpoint_tool.workflow.scientific_reporting import project_central_quality_status
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def replay(run_root: Path, out: Path) -> dict:
    run_root=run_root.resolve(strict=True);out=out.resolve()
    if out==run_root or run_root in out.parents:
        raise ValueError('Derived output must be outside the original run')
    out.mkdir(parents=True,exist_ok=True)
    status=load(run_root/'reports/run_status_summary.json')
    evidence=load(run_root/'reports/native_evidence_status.json')
    central=load(run_root/'quality_management/central_quality_summary.json')
    validation=load(run_root/'reports/native_validation/native_producer_validation_summary.json')
    summary=load(run_root/'reports/native_producer_summary.json')
    runtime_by_key={native_identity_key(r):r for r in summary['rows']}
    central_by_sha={validator._canonical_json_sha256(r):r for r in central['results']}
    derived_validation=copy.deepcopy(validation['rows']);repaired=[];rejected=[]
    policy=AccuracyGatePolicy.from_mapping(validation['accuracy_gate_policy'])
    for path in sorted((run_root/'native_producers').glob('*/analysis_tables/native_full_baseline_eval.json')):
        backend_root=path.parent.parent
        original_binding_path=backend_root/'quality_first/vendor_full_quality_request_binding_set.json'
        if not original_binding_path.is_file():continue
        binding_set=load(original_binding_path);original_set_sha=binding_set.get('binding_set_sha256')
        unsigned={k:v for k,v in binding_set.items() if k!='binding_set_sha256'}
        if validator._canonical_json_sha256(unsigned)!=original_set_sha:
            raise ValueError('Original binding set hash mismatch: '+str(original_binding_path))
        for source_row in load(path)['rows']:
            if source_row.get('backend') not in {'native_full_hailo8','native_full_hailo10h','native_full_deepx'}:continue
            if source_row.get('quality_request_binding_status')=='verified_exact':continue
            key=source_row['backend']+'|'+source_row['model']
            original_binding=binding_set['bindings_by_backend_model'].get(key)
            if not original_binding:continue
            unsigned={k:v for k,v in original_binding.items() if k!='binding_sha256'}
            if validator._canonical_json_sha256(unsigned)!=original_binding.get('binding_sha256'):
                raise ValueError('Original binding hash mismatch: '+key)
            result=central_by_sha.get(original_binding['central_quality_result_sha256'])
            if result is None:continue
            derived_set=copy.deepcopy(binding_set);binding=derived_set['bindings_by_backend_model'][key]
            binding.update(_vendor_full_completed_quality_fields_v282(result,result.get('request_identity') or {}))
            binding.pop('binding_sha256');binding['binding_sha256']=validator._canonical_json_sha256(binding)
            derived_set.pop('binding_set_sha256');derived_set['binding_set_sha256']=validator._canonical_json_sha256(derived_set)
            row=copy.deepcopy(source_row)
            ns=argparse.Namespace(quality_request_binding_set_data=derived_set,setup_id=row['setup_id'],comparison_backend=row['comparison_backend'])
            row,attach_status=full._attach_full_quality_request_binding(row,row['full_command_contract'],model=row['model'],ns=ns)
            if attach_status!='quality_request_binding_verified_exact':
                rejected.append({'key':key,'status':attach_status,'errors':row.get('quality_request_binding_errors')});continue
            contract=row['full_command_contract'];contract.update({k:row[k] for k in ('quality_request_binding','quality_request_binding_sha256','quality_request_binding_set_sha256')})
            contract.pop('contract_sha256');contract['contract_sha256']=validator._canonical_json_sha256(contract)
            row['full_command_contract_sha256']=contract['contract_sha256']
            rec=next((r for r in derived_validation if native_identity_key(r)==native_identity_key(row)),None)
            if rec is None:raise ValueError('Original validation identity missing: '+key)
            for field in list(rec):
                if field.startswith(('quality_','central_','precision_quality','vendor_full_')):rec.pop(field,None)
            rec['full_command_contract']=contract;rec['full_command_contract_sha256']=contract['contract_sha256']
            validator._merge_vendor_full_quality_evidence(rec,row)
            validator._bind_central_quality_evidence(rec,[result],policy)
            repaired.append({'key':key,'status':rec.get('central_quality_binding_status'),
                             'original_binding_sha256':original_binding['binding_sha256'],
                             'original_binding_set_sha256':original_set_sha,
                             'central_quality_result_sha256':original_binding['central_quality_result_sha256'],
                             'original_quality_decision':result.get('decision'),
                             'row':row,'validation_row':rec})
    def measured_errors(rows):
        return [r for r in rows if r.get('status')!='native_row_not_ok'
                and not known_build_exclusion(runtime_by_key.get(native_identity_key(r),{}))
                and validator._technical_quality_error(r)]
    original_errors=measured_errors(validation['rows']);remaining_errors=measured_errors(derived_validation)
    evidence=copy.deepcopy(evidence)
    evidence['validation_measured_technical_error_count']=len(remaining_errors)
    projection=project_historical_workflow_status(status,evidence)
    completion=workflow_completion_projection(projection['projected_technical_status'],native_evidence=evidence,central_quality=project_central_quality_status(central))
    result={'schema':'onnx-splitpoint/v282-status-quality-replay','source_run':str(run_root),
            'source_modified':False,'hardware_started':False,'quality_recomputed':False,
            'historical_status_projection':projection,'completion':completion,
            'original_measured_validation_error_count':len(original_errors),
            'remaining_measured_validation_error_count':len(remaining_errors),
            'remaining_validation_errors':[{'identity':list(native_identity_key(r)), 'reason':r.get('self_reference_reason') or r.get('central_quality_binding_status')} for r in remaining_errors],
            'full_bindings_repaired':repaired,'full_bindings_rejected':rejected}
    (out/'replay_status_quality.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (out/'replay_status_quality.md').write_text('# v2.82: abgeleitete Status- und Full-Qualitätsprüfung\n\n'+completion['message']+'\n\n'+f'Full-Bindungen: {len(repaired)} abgeleitet, {len(rejected)} abgelehnt. Technische Validierungskonflikte auf gemessenen Zeilen: vorher {len(original_errors)}, nach Bindungsreparatur {len(remaining_errors)}.\n\nOriginale bleiben unverändert. Keine Hardware-, Energie- oder Qualitätsneuberechnung.\n',encoding='utf-8')
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=replay(args.run_root,args.out)
    print(result['completion']['message'])
    print(f"Full bindings repaired: {len(result['full_bindings_repaired'])}; rejected: {len(result['full_bindings_rejected'])}")
    return 1 if result['full_bindings_rejected'] else 0

if __name__=='__main__':raise SystemExit(main())

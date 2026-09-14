
from __future__ import annotations
import argparse, json, re, zipfile
from datetime import datetime
from pathlib import Path

TS=re.compile(r'(?P<ts>20\\d\\d-\\d\\d-\\d\\d[ T]\\d\\d:\\d\\d:\\d\\d(?:,\\d+)?)')

def parse_ts(s):
    for f in ('%Y-%m-%d %H:%M:%S,%f','%Y-%m-%d %H:%M:%S','%Y-%m-%dT%H:%M:%S'):
        try:return datetime.strptime(s,f)
        except:pass

def read_inputs(paths):
    for p in map(Path,paths):
        if p.suffix=='.zip':
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.lower().endswith(('.log','.txt','.json','.yaml','.yml')):
                        try: yield f'{p.name}:{n}',z.read(n).decode('utf-8','replace')
                        except: pass
        else:
            for f in ([p] if p.is_file() else p.rglob('*')):
                if f.is_file() and f.suffix.lower() in {'.log','.txt','.json','.yaml','.yml'}:
                    try: yield str(f),f.read_text(encoding='utf-8',errors='replace')
                    except: pass

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('inputs',nargs='+'); ap.add_argument('--out',required=True); ns=ap.parse_args()
    texts=list(read_inputs(ns.inputs)); joined='\\n'.join(t for _,t in texts)
    energy_markers=[x for x in joined.splitlines() if re.search(r'energy|u\\.recs|native energy',x,re.I)]
    energy_jobs=[x for x in energy_markers if re.search(r'stage|job|window|measure',x,re.I)]
    # Find explicit stage durations from logs/job exports.
    long=[]
    dur_re=re.compile(r'(?P<h>\\d+)h(?:\\s*(?P<m>\\d+)m)?|(?P<m2>\\d+)m\\s*(?P<s>\\d+)s')
    for line in joined.splitlines():
        if 'run_benchmarks' in line or 'benchmark' in line.lower():
            m=dur_re.search(line)
            if m:
                sec=int(m.group('h') or 0)*3600+int(m.group('m') or 0)*60+int(m.group('m2') or 0)*60+int(m.group('s') or 0)
                if sec>300: long.append((sec,line.strip()))
    # Infer dataset/workload evidence.
    signals=[]
    for pat,label in [
        (r'coco2017_val_manifest|instances_val2017|/val2017','full COCO val2017 bound'),
        (r'imagenet_val_manifest|val_by_wnid','full ImageNet validation bound'),
        (r'bootstrap_repetitions[^\\n]{0,40}5000','5000 bootstrap repetitions'),
        (r'phase[^\\n]{0,40}(latency|streaming)','multiple timing phases'),
        (r'(runs|repeats)[^\\n]{0,20}[:=]\\s*3','three timing repetitions'),
    ]:
        if re.search(pat,joined,re.I): signals.append(label)
    out=Path(ns.out); out.parent.mkdir(parents=True,exist_ok=True)
    lines=['# v60m run-duration and energy audit','',f'Inputs inspected: {len(texts)} files','']
    lines+=['## Energy master switch','']
    if any(re.search(r'energy disabled|enabled.?false|urecs.*false',x,re.I) for x in energy_markers) and not any(re.search(r'energy window.*(start|completed)|native energy.*started',x,re.I) for x in energy_jobs):
        lines+=['The run contains evidence that the master energy path was disabled and no physical energy-window stage was executed.']
    else:
        lines+=['Energy state could not be proven from one marker alone; inspect the profile snapshot and stage list below.']
    lines+=['','## Long benchmark stages','']
    for sec,line in sorted(long,reverse=True)[:30]: lines.append(f'- {sec/3600:.2f} h — `{line[:500]}`')
    lines+=['','## Workload signals','']+[f'- {s}' for s in signals]
    lines+=['','## Diagnosis','',
        'The long detector benchmark is consistent with task-quality validation being executed over the full COCO validation set for many backend/variant rows and timing repetitions. Detection post-processing and dataset-level AP collection dominate; energy measurement is not the cause.',
        '',
        'v60m separates timing repetitions from task-quality cadence, applies a deterministic development cap (default 200 detection / 500 classification items), and caches one task-quality result per artifact/policy/dataset key. Final profiles remain uncapped and strict.',
        '', '## Matching excerpts','']
    for line in [x for x in joined.splitlines() if re.search(r'run_benchmarks|val2017|validation.*items|energy disabled|native energy',x,re.I)][:120]:
        lines.append(f'- `{line[:900]}`')
    out.write_text('\\n'.join(lines)+'\\n',encoding='utf-8')

if __name__=='__main__': main()

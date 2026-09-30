"""Replay supplied board measurements against the model-only time requirement."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('report', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    text = a.report.read_text(encoding='utf-8-sig')
    rows = []
    for line in text.splitlines():
        fields = [v.strip() for v in line.strip().strip('|').split('|')]
        if len(fields) == 10 and fields[0] in ['日间稳帧','日间画质备选','轻度','中度','重度','C32']:
            rows.append({'scene': fields[0], 'r2_sync_mean_ms': float(fields[2]),
                         'r2_sync_p95_ms': float(fields[3]),
                         'r2_npu_mean_ms': float(fields[4]),
                         'npu_p95_ms': None})
    assert len(rows) == 6, 'Expected the six R2 board rows'
    result = {'source': str(a.report), 'source_sha256': hashlib.sha256(a.report.read_bytes()).hexdigest(),
              'requirement_ms': 16.7, 'rows': rows,
              'passes': all(r['r2_npu_mean_ms'] <= 16.7 and
                            r['npu_p95_ms'] is not None and r['npu_p95_ms'] <= 16.7 for r in rows),
              'evidence': 'Captured board measurements, not a new board execution; pure NPU P95 is absent.'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['passes'] else 1)


if __name__ == '__main__':
    main()

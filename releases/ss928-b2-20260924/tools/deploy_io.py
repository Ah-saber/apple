"""部署输入、参考推理与逐值误差核验。所有二进制均为小端连续数组。"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read_float(path, count):
    x=np.fromfile(path,dtype='<f4')
    if x.size!=count or not np.isfinite(x).all():raise ValueError('文件尺寸或数值无效；先去除板端 stride/padding：'+str(path))
    return x
p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
n=sub.add_parser('normalize');n.add_argument('--raw',required=True);n.add_argument('--metadata',required=True);n.add_argument('--output',required=True);n.add_argument('--frame',type=int,default=0)
i=sub.add_parser('infer');i.add_argument('--model',required=True);i.add_argument('--input',required=True);i.add_argument('--output',required=True)
c=sub.add_parser('compare');c.add_argument('--actual',required=True);c.add_argument('--reference',required=True);c.add_argument('--output',required=True)
v=sub.add_parser('verify')
b=sub.add_parser('calibrate-sequence');b.add_argument('--sequence',required=True);b.add_argument('--output',required=True)
a=p.parse_args()
if a.cmd=='verify':
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for row in manifest['files']:
        f=ROOT/row['path']; h=hashlib.sha256(f.read_bytes()).hexdigest()
        if h!=row['sha256'] or f.stat().st_size!=row['bytes']:raise ValueError('校验失败：'+str(f))
    print('PASS',len(manifest['files']),'files')
elif a.cmd=='normalize':
    raw=np.fromfile(a.raw,dtype='<u2')
    if raw.size!=1024*1280:raise ValueError('RAW 应为无文件头的 1024x1280 uint16')
    metadata=json.loads(Path(a.metadata).read_text())
    if 'raw_sha256' in metadata and hashlib.sha256(Path(a.raw).read_bytes()).hexdigest()!=metadata['raw_sha256']:raise ValueError('RAW与样例身份不一致')
    if 'normalization' in metadata:norm=metadata['normalization']
    else:
        if not 0<=a.frame<len(metadata['parameters']):raise ValueError('帧号超出参数范围')
        norm=metadata['parameters'][a.frame]
    if not np.isfinite([norm['offset'],norm['scale']]).all() or norm['scale']<=0:raise ValueError('无效归一化参数')
    ((raw.astype(np.float32)-norm['offset'])/norm['scale']).astype('<f4').tofile(a.output)
elif a.cmd=='infer':
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(a.model,options,providers=['CPUExecutionProvider'])
    x=read_float(a.input,1024*1280).reshape(1,1,1024,1280)
    y=session.run(['display'],{'raw':x})[0]
    if y.shape!=(1,1,3072,3840) or not np.isfinite(y).all():raise ValueError('无效输出')
    y.astype('<f4').tofile(a.output)
elif a.cmd=='compare':
    x=read_float(a.actual,3072*3840).astype(np.float64);y=read_float(a.reference,3072*3840).astype(np.float64)
    d=x-y;mse=float(np.mean(d*d));r={'max_abs':float(abs(d).max()),'mae':float(abs(d).mean()),'rmse':float(np.sqrt(mse)),'mean_bias':float(d.mean()),'psnr_range1_unclipped':float(-10*np.log10(mse)) if mse else 'inf','meaning':'相对同权重 CPU ONNX 的部署误差，未裁剪；无 GT 质量含义。阈值需另行约定。'}
    Path(a.output).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False))
else:
    sys.path.insert(0,str(ROOT/'source/src'))
    from ir_sr.sequence_normalization import calibrate_sequence
    raw=np.load(a.sequence,mmap_mode='r',allow_pickle=False)
    if raw.ndim!=3 or raw.shape[1:]!=(1024,1280) or raw.dtype!=np.uint16:raise ValueError('需要完整 T,1024,1280 uint16 序列')
    rows,segments=calibrate_sequence(raw)
    Path(a.output).write_text(json.dumps({'parameters':rows,'segments':segments,'mode':'offline complete sequence; future RAW required'},indent=2))

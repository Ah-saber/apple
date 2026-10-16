"""Inventory the retained controller's neural modules and source dependencies."""
import json
from pathlib import Path
import sys
import torch
from quality_shape330 import QualityShape330
from quality_control43 import QualityControl43


def roots(obj):
    visited = set()
    result = []
    def visit(value, path):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.nn.Module):
            result.append((path, value))
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path+'.'+str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, path+'.'+str(index))
        elif hasattr(value, '__dict__') and value.__class__.__module__ not in ['builtins', 'numpy', 'torch']:
            for key, child in vars(value).items():
                if not key.startswith('__'):
                    visit(child, path+'.'+key)
    visit(obj, 'controller')
    return result


def main():
    report = []
    for scene in ['weather_heavy_c32', 'weather_heavy', 'night_special']:
        controller = QualityShape330(332) if scene == 'weather_heavy_c32' else QualityControl43(scene)
        items = [dict(path=path, type=module.__class__.__module__+'.'+module.__class__.__name__,
            parameters=sum(p.numel() for p in module.parameters())) for path, module in roots(controller)]
        report.append(dict(scene=scene, neural_roots=items))
        print(json.dumps(report[-1]), flush=True)
        del controller
        torch.cuda.empty_cache()
    files = sorted({str(Path(module.__file__).resolve()) for module in list(sys.modules.values()) if getattr(module, '__file__', None) and '/SS928-ROUND5-20261007/runtime/' in str(module.__file__)})
    output = Path('/tmp/rawir-current-deploy-20261016')
    output.mkdir(exist_ok=True)
    (output/'inventory.json').write_text(json.dumps(dict(scenes=report, runtime_files=files), indent=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(3.8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    main()

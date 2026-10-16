"""Nonlinear phase detail with optional causal sensor inputs and a frozen reference."""
import copy
import json
import time
import torch
from torch import nn
from torch.nn import functional as F
from target_residual import ROOT, PhaseResidual, SpatialResidual, ShrinkResidual
from direct_motion import DirectBoard
from run_verified import half_board
from streaming_ring_data import GuidanceView as RingView

class GuidanceView(RingView):
    sensor = False
    def inputs(self, pipeline):
        key = pipeline + ':sensor3'
        if not self.sensor:
            return super().inputs(pipeline)
        if key not in self.cache:
            start = time.perf_counter()
            corrected = super().inputs(pipeline)
            # This causal ring has no background correction. It keeps the same
            # segment reset and normalization as the frozen current pipeline.
            plain = self._nine_without_field()[-3:]
            self.cache[key] = torch.cat([corrected, plain])
            self.preprocess_log.append(dict(pipeline=key, frame=int(self.current_record['frame_id']), elapsed_ms=(time.perf_counter()-start)*1000, scope='complete corrected input plus sensor ring and concatenation'))
        return self.cache[key]

class CausalDetail(nn.Module):
    sensor = False
    def __init__(self, count):
        super().__init__()
        self.core_count = count
        self.conv = nn.Conv2d(count + (3 if self.sensor else 0), 12, 4, stride=2, padding=1, bias=True)
        self.mix = nn.Conv2d(12, 4, 1, bias=False)
        nn.init.normal_(self.conv.weight, std=.06)
        nn.init.constant_(self.conv.bias, -1/255)
        nn.init.zeros_(self.mix.weight)
        self.project()
    def project(self):
        with torch.no_grad():
            for a, b in [(0, self.core_count)] + ([(self.core_count, self.core_count+3)] if self.sensor else []):
                w = self.conv.weight[:,a:b]
                w.sub_(w.mean(1, keepdim=True))
                w.clamp_(-1, 1)
                w.sub_(w.mean(1, keepdim=True))
            self.conv.bias.clamp_(-8/255, 0)
            self.mix.weight.clamp_(-2, 2)
    def forward(self, raw):
        return self.mix(F.relu(self.conv(raw.to(self.conv.weight.dtype))))

class SensorDetail(CausalDetail):
    sensor = True

class ReferencePhases(DirectBoard):
    def __init__(self, board, count, kind, accepted):
        super().__init__(board, PhaseResidual(count).cuda(), count, kind)
        self.accepted = copy.deepcopy(accepted).half().eval() if accepted is not None else None
    def phases(self, raw, context):
        raw = raw[:,:self.source_count]
        value = self.night_phases(raw, context) if self.kind else self.board.stages(raw, context)['preclip_phases']
        return value if self.accepted is None else value + self.accepted(raw)
    def forward(self, raw, context):
        value = self.phases(raw, context)
        output = self.board.output if self.kind else self.board.rows
        return output(value.clamp(0,1)*255)

def reference_board(base, count, kind, box, scene):
    decisions = json.loads((ROOT/'TARGET-RESIDUAL-DECISIONS.json').read_text())
    family = decisions[scene]['review_candidate']
    accepted = None
    if family:
        ck = torch.load(ROOT/(family+scene)/'best.pt', map_location='cpu', weights_only=False)
        if ck.get('branch_class') == 'ShrinkResidual':
            linear = torch.load(ROOT/(ck.get('source_family','target_residual_')+scene)/'last.pt', map_location='cpu', weights_only=False)['branch']
            accepted = ShrinkResidual(count, linear, ck['threshold_gray']).cuda()
        else:
            accepted = (SpatialResidual if ck.get('branch_class') == 'SpatialResidual' else PhaseResidual)(count).cuda()
        accepted.load_state_dict(ck['branch'])
    board = base if kind else half_board(base, box.cuda())
    return ReferencePhases(board, count, kind, accepted).cuda().eval().requires_grad_(False)

class DetailBoard(nn.Module):
    def __init__(self, reference, branch):
        super().__init__()
        self.reference = copy.deepcopy(reference)
        self.branch = copy.deepcopy(branch).half().eval()
    def phases(self, raw, context):
        return self.reference.phases(raw, context)
    def forward(self, raw, context):
        value = self.phases(raw, context) + self.branch(raw)
        output = self.reference.board.output if self.reference.kind else self.reference.board.rows
        return output(value.clamp(0,1)*255)

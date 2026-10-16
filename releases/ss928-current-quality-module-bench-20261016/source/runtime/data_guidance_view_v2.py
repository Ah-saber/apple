"""Additional causal reference choices after real short-input static guard failures."""
import numpy as np
import torch
from data_guidance_view import GuidanceView as OriginalView, InputModel
from protected_data import correct_selected, correct_frame


class GuidanceView(OriginalView):
    def inputs(self, pipeline):
        if pipeline in self.cache:
            return self.cache[pipeline]
        if pipeline in ['control9', 'legacyK']:
            return super().inputs(pipeline)
        if pipeline == 'protected9K':
            value = super().inputs('protected9')[self.indices].contiguous()
        else:
            ids = self.source.frame_ids(self.current_record)
            selected = ids if pipeline == 'lastprotectedK' else [ids[i] for i in self.indices]
            top, left, h, w = self.current_crop
            values = []
            for f in selected:
                rr = dict(self.current_record, frame_id=f)
                norm = self.source.normalization_for(rr)
                raw = self.source.base._raw(rr)[top:top+h, left:left+w].astype(np.float32)
                values.append((raw-norm['offset'])/norm['scale'])
            stack = np.stack(values)
            if pipeline == 'lastprotectedK':
                value = super().inputs('legacyK').clone()
                if self.source.dynamic_correction:
                    reference = np.median(stack[:6], axis=0)
                    value[-1] = torch.from_numpy(correct_frame(stack[-1], reference, True))
            else:
                if self.source.dynamic_correction:
                    stack = correct_selected(stack, protect=pipeline == 'protectedReferenceK', reference='all')
                value = torch.from_numpy(np.ascontiguousarray(stack))
        self.cache[pipeline] = value
        return value


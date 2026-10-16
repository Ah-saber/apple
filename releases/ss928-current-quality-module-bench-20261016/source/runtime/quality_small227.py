"""Current compact-target correction above the retained complete causal output."""
import json
import cv2,numpy as np,torch
import quality_c32206 as frozen206
from target_residual import ROOT
from compact_match93 import TOP,BOTTOM
from quality_surround216 import QualitySurround216
from small_context227 import SmallContext227,features227

class QualitySmall227(QualitySurround216):
    def __init__(self,number=227):
        super().__init__(222)
        fit=json.loads((ROOT/('MOTION%d-20261015'%number)/'weather_heavy_c32'/'fit.json').read_text())
        if fit['status']!='passed' or fit['current_complete_parent']!=222:
            raise ValueError('Selection relative to retained output required')
        self.foreground_alpha=fit['best']['alpha']
        if number==244:
            from small_wide244 import SmallWide244
            self.foreground_model=SmallWide244().cuda().eval().requires_grad_(False)
        else:self.foreground_model=SmallContext227().cuda().eval().requires_grad_(False)
        self.foreground_halo=64 if number==244 else 32
        checkpoint_number=233 if number==236 else number
        self.foreground_model.load_state_dict(torch.load(ROOT/('MOTION%d-20261015'%checkpoint_number)/'weather_heavy_c32'/'last_step20000.pt',map_location='cpu',weights_only=False)['model'])

    def reset(self):
        super().reset()
        self.previous_small_base=None

    @torch.inference_mode()
    def native(self,fields,row,frame,initial=False):
        original=frozen206.extra170
        count=0;reference=None
        def capture(*args,**kwargs):
            nonlocal count,reference
            result=original(*args,**kwargs);count+=1
            if count==2:
                weight=args[4]
                reference=np.clip(args[3]+np.divide(result[0],weight,out=np.zeros_like(weight),where=weight>0),0,255)
            return result
        frozen206.extra170=capture
        try:before=super().native(fields,row,frame,initial=initial)
        finally:frozen206.extra170=original
        self.last_before_native=before.clone()
        self.last_small_features=None
        self.last_physical_reference195=reference
        whole=before[0,0].float().cpu().numpy();body=whole[TOP:BOTTOM]
        if self.previous_small_base is None:self.previous_small_base=body.copy()
        proposed=before.clone()
        if self.last_nn_context is not None:
            if reference is None:raise RuntimeError('Physical reference missing')
            physical=self.parent.last_physical_fore95.astype(np.float32)
            hp=whole-cv2.blur(whole,(9,9),borderType=cv2.BORDER_REFLECT)
            independent=self.parent.last_body_image107[TOP:BOTTOM].astype(np.float32)
            extra=np.stack([body/255,hp[TOP:BOTTOM]/16,(body-self.previous_small_base)/16,
                            (body-independent)/16,(reference-body)/16]).clip(-4,4).astype(np.float16)
            feature=features227(self.last_nn_context,extra,self.last_release)
            self.last_small_features=feature.copy()
            core=self.road>.9;core[:,:64]=False
            values=physical[core]
            sigma=max(.2,float(1.4826*np.median(np.abs(values-np.median(values)))))
            signal=np.clip((np.abs(physical)-sigma)/(2*sigma),0,1)
            mask=np.maximum(self.last_compact_mask,self.last_release)*self.last_guard*signal
            mask[:,:64]=0
            self.last_small_mask=mask.copy()
            head=np.zeros_like(body)
            for y in range(0,BOTTOM-TOP,128):
                end=min(y+128,BOTTOM-TOP);top=max(0,y-self.foreground_halo);bottom=min(BOTTOM-TOP,end+self.foreground_halo)
                values=torch.from_numpy(feature[:,top:bottom].astype(np.float32))[None].to(before.device)
                masks=torch.from_numpy(mask[top:bottom])[None,None].to(before.device)
                head[y:end]=self.foreground_model(values,masks)[0,0,y-top:end-top].cpu().numpy()
            proposed[...,TOP:BOTTOM,:]+=torch.from_numpy(head)[None,None].to(before.device)
            proposed=proposed.clamp(0,255)
        self.previous_small_base=body.copy()
        self.last_proposed_native=proposed
        self.last_context_head=(proposed-self.last_base206_native)[0,0,TOP:BOTTOM].float().cpu().numpy()
        return (before+self.foreground_alpha*(proposed-before)).clamp(0,255)

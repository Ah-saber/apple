"""Causal, current-anchored RAW body denoising with dense alignment."""
import cv2,numpy as np

def sigma194(physical,road):
    core=road>.9;core[:,:64]=False;values=physical[core]
    return max(.2,float(1.4826*np.median(np.abs(values-np.median(values)))))

def observations194(physical,road):
    sigma=sigma194(physical,road)
    value=cv2.bilateralFilter(physical.astype(np.float32),5,2*sigma,.8,borderType=cv2.BORDER_REFLECT)
    guide=cv2.GaussianBlur(physical,(0,0),.8,borderType=cv2.BORDER_REFLECT)
    return value,guide,sigma

def warp194(current_guide,past_guide,past_value):
    args=(.5,3,15,5,5,1.1,0)
    forward=cv2.calcOpticalFlowFarneback(current_guide,past_guide,None,*args)
    backward=cv2.calcOpticalFlowFarneback(past_guide,current_guide,None,*args)
    yy,xx=np.mgrid[:current_guide.shape[0],:current_guide.shape[1]].astype(np.float32)
    mx=xx+forward[...,0];my=yy+forward[...,1]
    value=cv2.remap(past_value,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    inverse=cv2.remap(backward,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    error=np.linalg.norm(forward+inverse,axis=2)
    confidence=np.clip(1-error/1.5,0,1)
    return value,confidence

def filter194(physical,history,active,road):
    current,guide,sigma=observations194(physical,road)
    past=[observations194(a,road) for a in history]
    result=current.copy();records=[]
    count,labels,boxes,_=cv2.connectedComponentsWithStats((active>0).astype(np.uint8),connectivity=8)
    height,width=current.shape
    for label in range(1,count):
        x,y,w,h,_=boxes[label];top=max(0,y-32);left=max(0,x-32);bottom=min(height,y+h+32);right=min(width,x+w+32)
        crop=np.s_[top:bottom,left:right];mask=labels[crop]==label
        value=current[crop];total=value.copy();denominator=np.ones_like(value);adjusted_errors=[];warped_errors=[];raw_errors=[];matched_counts=[]
        strong=mask&(np.abs(value)>sigma)
        for old,old_guide,_ in past:
            warped,confidence=warp194(guide[crop],old_guide[crop],old[crop])
            compatibility=np.clip(1-np.abs(warped-value)/(2*sigma),0,1)
            weight=compatibility*confidence
            matching=(warped*value>0)&(np.abs(warped)>.5*sigma)
            diagnostic=strong&(confidence>.5)&matching
            weight[strong&~matching]=0
            adjusted=warped.copy()
            # Remove current object brightness change before averaging its noise.
            # Compatibility is computed first, so a blank past cannot create a body.
            for sign in [1,-1]:
                group=mask&(sign*value>sigma)
                valid=group&(weight>0)&(confidence>.5)&matching
                if int(valid.sum())>=12:
                    offset=float(np.median((warped-value)[valid]));adjusted[group]-=offset
                else:weight[group]=0
            total+=weight*adjusted;denominator+=weight
            matched_counts.append(int(diagnostic.sum()))
            if diagnostic.any():
                raw_errors.append(float(np.median(np.abs(old[crop]-value)[diagnostic])))
                warped_errors.append(float(np.median(np.abs(warped-value)[diagnostic])))
                adjusted_errors.append(float(np.median(np.abs(adjusted-value)[diagnostic])))
            else:raw_errors.append(None);warped_errors.append(None);adjusted_errors.append(None)
        filtered=value+np.clip(total/denominator-value,-sigma/2,sigma/2)
        local=result[crop];local[mask]=filtered[mask]
        records.append(dict(box=[int(y),int(x),int(h),int(w)],pixels=int(mask.sum()),sigma=sigma,
                            unaligned_error=raw_errors,warped_before_offset_error=warped_errors,
                            adjusted_error=adjusted_errors,matched_pixels=matched_counts,
                            median_change=float(np.median((filtered-value)[strong])) if strong.any() else 0.))
    return result,records

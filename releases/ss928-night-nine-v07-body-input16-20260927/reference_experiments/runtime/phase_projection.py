"""Nine phases with linear constraints preserving the native 2x2 pixel means.
Equality applies before clipping, rounding, and finite-precision computation.
"""
import numpy as np

def preserve_native_means(weight,bias):
    weight=np.asarray(weight);bias=np.asarray(bias)
    if weight.shape!=(36,16,3,3) or bias.shape!=(36,):raise ValueError('Require original 36-phase projection')
    grid=weight.astype(np.float64).reshape(6,6,16,3,3);bg=bias.astype(np.float64).reshape(6,6)
    initial=np.stack([grid[y*2:y*2+2,x*2:x*2+2].mean((0,1)) for y in range(3) for x in range(3)])
    initial_bias=np.array([bg[y*2:y*2+2,x*2:x*2+2].mean() for y in range(3) for x in range(3)])
    target=np.stack([grid[y*3:y*3+3,x*3:x*3+3].mean((0,1)) for y in range(2) for x in range(2)])
    target_bias=np.array([bg[y*3:y*3+3,x*3:x*3+3].mean() for y in range(2) for x in range(2)])
    axis=np.array([[2/3,1/3,0],[0,1/3,2/3]],np.float64)
    mapping=np.kron(axis,axis);correction=mapping.T@np.linalg.inv(mapping@mapping.T)
    flat=initial.reshape(9,-1);flat=flat+correction@(target.reshape(4,-1)-mapping@flat)
    new_bias=initial_bias+correction@(target_bias-mapping@initial_bias)
    proof={'pre_round_weight_constraint_max':float(np.abs(mapping@flat-target.reshape(4,-1)).max()),'pre_round_bias_constraint_max':float(np.abs(mapping@new_bias-target_bias).max())}
    cast=flat.reshape(9,16,3,3).astype(weight.dtype);cast_bias=new_bias.astype(bias.dtype)
    proof['stored_weight_constraint_max']=float(np.abs(mapping@cast.astype(np.float64).reshape(9,-1)-target.reshape(4,-1)).max())
    proof['stored_bias_constraint_max']=float(np.abs(mapping@cast_bias.astype(np.float64)-target_bias).max())
    return cast,cast_bias,proof

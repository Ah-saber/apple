"""Stabilize only background repair, using independent compatible observations."""
from collections import deque
import cv2
import numpy as np

class SurroundStable211:
    def __init__(self):
        self.bank = deque(maxlen=4)

    def step(self, image, independent, physical, confirmed, weight, road):
        core = road > .9
        core[:, :64] = False
        values = physical[core]
        sigma = max(.2, float(1.4826*np.median(np.abs(values-np.median(values)))))
        guide = cv2.GaussianBlur(physical, (0, 0), .8, borderType=cv2.BORDER_REFLECT)
        correction = cv2.GaussianBlur(image-independent, (0, 0), 2., borderType=cv2.BORDER_REFLECT)
        total = correction.copy()
        count = np.ones_like(correction)
        for old_correction, old_guide, old_confirmed in self.bank:
            compatible = np.clip(1-np.abs(guide-old_guide)/(2*sigma), 0, 1)
            compatible *= old_confirmed <= 0
            total += compatible*old_correction
            count += compatible
        distance = cv2.distanceTransform((confirmed <= 0).astype(np.uint8), cv2.DIST_C, 3)
        outside = np.clip(distance/3, 0, 1)*np.clip(weight, 0, 1)*road
        outside[:, :64] = 0
        delta = outside*np.clip(total/count-correction, -3, 3)
        # Store the independent input observation, never the stabilized output.
        self.bank.append((correction.copy(), guide.copy(), confirmed.copy()))
        return delta.astype(np.float32), dict(sigma=sigma, history=len(self.bank), changed_pixels=int((np.abs(delta)>1e-5).sum()), maximum=float(np.abs(delta).max()))

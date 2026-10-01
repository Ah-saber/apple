"""Convert low-grid phases to bytes before value-preserving integer layout operations."""
import copy

import torch
from torch import nn


class ByteReorder(nn.Module):
    def __init__(self,factor,height=1024,width=1280):
        super().__init__();self.factor=factor
        self.register_buffer('rows',torch.arange(height*3)//3)
        self.register_buffer('columns',torch.arange(width*3)//3)

    def forward(self,phases):
        phases=phases.to(torch.uint8)
        n,c,h,w=phases.shape;s=self.factor
        image=phases.reshape(n,1,s,s,h,w).permute(0,1,4,2,5,3).reshape(n,1,h*s,w*s)
        return image.index_select(2,self.rows).index_select(3,self.columns)


class IntegerOutput(nn.Module):
    def __init__(self,board,factor):
        super().__init__();self.board=copy.deepcopy(board);self.board.layout='phases'
        self.output=ByteReorder(factor)

    def forward(self,x,c):return self.output(self.board(x,c))

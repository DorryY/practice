# -*- coding: utf-8 -*-
import numpy as np

class Sigmoid:
    
    def __init__(self):
        self.params=[]
        
    def forward(self,x):
        return 1/(1+np.exp(-x))
    
class Affine:
    
    def __init__(self,W,b):
        self.params=[W,b]
        
    def forward(self,x):
        W,b = self.params
        out = np.matmul(x, W) + b
        return out
    
class Softmax:
    def __init__(self):
        self.params=[]
    
    def forward(self,x,t):
        return sum(np.log(np.exp(x)/sum(np.exp(x)))*t)
    
class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        I,H,O = input_size,hidden_size,output_size
        
        W1 = np.random.rand(I,H)
        b1 = np.random.rand(H)
        W2 = np.random.rand(H,O)
        b2 = np.random.rand(O)
        
        self.layers=[
                Affine(W1,b1),
                Sigmoid(),
                Affine(W2,b2)
                ]
                
        self.params = []
        for layer in self.layers:
            self.params += layer.params
    
    def predict(self,x):
        for layer in self.layers:
            x= layer.forward(x)
        return x
    

x = np.random.rand(10,2) #2개짜리 input을 가진 데이터 10개
model = TwoLayerNet(2,4,3) # input 2개, 1차 layer 4개, 최종 아웃풋 3개
S = model.predict(x)  # 결과 돌리기


    
        
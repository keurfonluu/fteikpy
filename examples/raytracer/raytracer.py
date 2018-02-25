# -*- coding: utf-8 -*-

"""
Ray Tracer computes the first arrival traveltimes in a stratified medium. This
code is not optimized and is only provided as a comparison for Eikonal solvers.

Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from ._fminbnd import fminbnd

__all__ = [ "Ray3D" ]


class Ray3D:
    
    def __init__(self, arrival = 0, positions = None, traveltime = None):
        self.arrival = arrival
        self.positions = positions
        self.traveltime = traveltime
        
    def trace(self, src, rcv, vel, zint):
        # Check src and rcv
        assert src[2] < zint[-1]
        assert rcv[2] < zint[-1]
        
        self.source = src.copy()
        self.receiver = rcv.copy()

        # Determine source and receiver layers
        ns = 0
        nr = 0
        for i in range(len(zint)-1):
            if src[2] >= zint[i]:
                ns += 1
            if rcv[2] >= zint[i]:
                nr += 1
                
        # Number of layers between source and receiver
        nlayer = np.abs(ns - nr) + 1
        self.positions = np.zeros((nlayer + 1, 3))
        
        if ns == nr:
            self.positions[0,:] = src.copy()
            self.positions[-1,:] = rcv.copy()
            self.traveltime = np.linalg.norm(src - rcv) / vel[ns]
            return
        else:
            self.dhorz = np.linalg.norm(src[:2] - rcv[:2])            
            
            # Initialize positions
            self.positions[0,2] = src[2]
            if nlayer > 1:
                if ns > nr:
                    if nr == 0:
                        self.positions[1:nlayer,2] = zint[ns-1::-1]
                    else:
                        self.positions[1:nlayer,2] = zint[ns-1:nr-1:-1]
                else:
                    self.positions[1:nlayer,2] = zint[ns:nr]
            self.positions[-1,2] = rcv[2]
            
            # Layers of interest
            V = np.zeros(nlayer)
            V[0], V[-1] = vel[ns], vel[nr]
            H = np.zeros(nlayer)
            if ns < nr:
                H[0] = np.abs(self.source[2] - zint[ns])
                for i in range(1, nlayer-1):
                    V[i] = vel[ns+i]
                    H[i] = zint[ns+i] - zint[ns+i-1]
                H[-1] = np.abs(self.receiver[2] - zint[nr-1])
            elif ns > nr:
                H[0] = np.abs(self.source[2] - zint[ns-1])
                for i in range(1, nlayer-1):
                    V[i] = vel[ns-i]
                    H[i] = zint[ns-i] - zint[ns-i-1]
                H[-1] = np.abs(self.receiver[2] - zint[nr])
            
        # Shift so that xsrc, ysrc = 0, 0
        self._shift()
        
        # Rotate to remove y axis
        self._rotate()
            
        # Invert for the take-off angle
        iopt, gfit = fminbnd(self._costfunc, 0., 180., eps = 1e-16, args = (V, H))
        
        # Shoot with optimal take-off angle...
        self.positions[:,0] = self._shoot(iopt, V, H)
        
        # ...and compute traveltime
        self.traveltime = 0.
        for i in range(nlayer):
            self.traveltime = self.traveltime + np.linalg.norm(self.positions[i,:] - self.positions[i+1,:]) / V[i]
        
        # Unrotate
        self._unrotate()
        
        # Unshift
        self._unshift()
    
    def lay2tt(self, src, rcv, vel, zint):
        # Parameters
        nsrc = src.shape[0]
        nrcv = rcv.shape[0]
        
        # Compute traveltimes using a ray tracer
        tcalc = np.zeros((nrcv, nsrc))        
        for j in range(nsrc):
            for k in range(nrcv):
                self.trace(src[j,:], rcv[k,:], vel, zint)
                tcalc[k,j] = self.traveltime
        return tcalc
        
    def _shoot(self, i, V, H):
        p = np.sin(np.deg2rad(i)) / V[0]
        nlayer = len(V)
        X = np.zeros(nlayer + 1)   
        for i in range(1, nlayer + 1):
            X[i] = X[i-1] + H[i-1] * np.tan(np.arcsin(V[i-1]*p))
        return X
        
    def _costfunc(self, i, *args):
        V, H = args
        X = self._shoot(i, V, H)
        return np.abs(self.dhorz - X[-1])
        
    def _shift(self, pos = False):
        self.receiver[:2] -= self.source[:2]
        if pos is True:
            for i in range(self.positions.shape[0]):
                self.positions[i,:2] -= self.source[:2]
        
    def _unshift(self):
        self.receiver[:2] += self.source[:2]
        for i in range(self.positions.shape[0]):
            self.positions[i,:2] += self.source[:2]        
        
    def _rotate(self, pos = False):
        self.theta = -np.arctan2(self.receiver[1], self.receiver[0])
        x = self.receiver[0] * np.cos(self.theta) - self.receiver[1] * np.sin(self.theta)
        y = self.receiver[0] * np.sin(self.theta) + self.receiver[1] * np.cos(self.theta)
        self.receiver[0] = x
        self.receiver[1] = y
        if pos is True:
            x = self.positions[:,0] * np.cos(self.theta) - self.positions[:,1] * np.sin(self.theta)
            y = self.positions[:,0] * np.sin(self.theta) + self.positions[:,1] * np.cos(self.theta)
            self.positions[:,0] = x
            self.positions[:,1] = y
        
    def _unrotate(self):
        x = self.receiver[0] * np.cos(-self.theta) - self.receiver[1] * np.sin(-self.theta)
        y = self.receiver[0] * np.sin(-self.theta) + self.receiver[1] * np.cos(-self.theta)
        self.receiver[0] = x
        self.receiver[1] = y
        x = self.positions[:,0] * np.cos(-self.theta) - self.positions[:,1] * np.sin(-self.theta)
        y = self.positions[:,0] * np.sin(-self.theta) + self.positions[:,1] * np.cos(-self.theta)
        self.positions[:,0] = x
        self.positions[:,1] = y
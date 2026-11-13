"""Original ridge solver; the 16-head cap describes one matched fit.

This module is not invoked by the released frozen-policy verifier.
"""
import numpy as np
from scipy.linalg import solve
from .frozen_math import design, predict
FITS=0
def fit(X,y,odd):
 global FITS
 FITS+=1;assert FITS<=16 and 1359+FITS<=1500
 X=np.asarray(X,float);y=np.asarray(y,float);assert X.ndim==2 and y.shape==(len(X),) and len(X)
 mean=np.zeros(X.shape[1]) if odd else X.mean(0)
 scale=np.maximum(np.sqrt(np.mean(X*X,axis=0)) if odd else X.std(0),.0001 if odd else .05)
 m=dict(mean=mean.tolist(),scale=scale.tolist(),odd=odd);Z=design(m,X);lam=.01;n,d=Z.shape
 theta=solve(Z.T@Z/n+lam*np.eye(d),Z.T@y/n,assume_a='pos');grad=Z.T@(Z@theta-y)/n+lam*theta
 gap=float(grad@grad/(2*lam));assert gap<=1e-12 and np.isfinite(theta).all()
 m.update(theta=theta.tolist(),proof=dict(strong_convexity=lam,gradient_norm=float(np.linalg.norm(grad)),objective_gap_upper=gap,closed_form_updates=1,fit_rows=n))
 return m

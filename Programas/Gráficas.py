# -*- coding: utf-8 -*-
"""libraries"""
import glob
import os
import pandas as pd
from tkinter import filedialog
from tkinter import Tk
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

"""Calling for paths"""
root = Tk()
root.filename = filedialog.askdirectory()
root.withdraw()
direc_pwm=os.path.join(root.filename,'*.csv'); path_pwm=[]
direc_optical=os.path.join(root.filename,'*.lvm'); path_optical=[]

for fpwm in glob.glob(direc_pwm):
    path_pwm.append(fpwm)
for foptical in glob.glob(direc_optical):
    path_optical.append(foptical)
    
"""Calling files_pwm"""    
dfs={}
for serie in path_pwm:
    serie_pd = pd.read_csv(serie, sep = ',',encoding='latin-1')
    dfs[serie] = pd.DataFrame(serie_pd)

"""FUNCTIONS:"""
"""raw data Dataframes"""
def rawDatos(fuerza):
    data=dfs
    directs=path_pwm
    df2={}
    rows=[]
    for path in directs:
        data_act=data[path]
        RPM=data_act['Motor Electrical Speed (RPM)']
        Fuerza=data_act[fuerza]
        time=data_act['Time (s)']
        df={'RPM': RPM,fuerza: Fuerza,'time': time}
        df3=pd.DataFrame(df)
        df2[path]=df3.interpolate()#fillna(method='ffill')
        df2[path]=df2[path].fillna(0)
        zero_ind=np.array(df2[path][df2[path]['RPM']<500].index)
        df2[path]=df2[path].drop(zero_ind)
        row=np.shape(df2[path]['RPM'])
        rows.append(row)
    r_min=np.min(rows)
    for path in directs:
        indices=np.array(df2[path]['RPM'][r_min:].index)
        df2[path]=df2[path].drop(indices)
    return df2

"""Important Data"""
def datos(fuerza):
    data=rawDatos(fuerza)
    Ps=path_pwm
    RPMm=0
    l=len(data[Ps[0]]['RPM'])
    fuerzad=np.zeros((l,len(Ps)))
    for p in Ps:
        i=Ps.index(p)
        RPMm+=data[p]['RPM']
        fuerzad[:,i]=data[p][fuerza]
    RPMm=RPMm/len(Ps)
    RPMs=RPMm; RPMs.iloc[0]=500
    RPMs=RPMm.interpolate()
    return RPMs,fuerzad

"""Chauvenet"""
def chauvenet (x):
    xm=np.mean(x);xd=np.std(x)
    dev=[abs(i-xm) for i in x]; ts=dev/xd
    probs=2*scipy.stats.norm.cdf(xm-ts*xd,loc=xm, scale=xd)
    for i in probs:
        ind=list(probs).index(i)
        if i<=0.5:
            x[ind]=np.nan
    return x

"""Data Prom"""
def dataProm(fuerza):
    RPM=datos(fuerza)[0]
    fuerzas=datos(fuerza)[1]
    mx=np.amax(RPM);#mxi=RPM.index(mx)
    #RPMsout=RPM[mxi:]; indices=RPM.index(RPMsout)
    sh=RPM.shape
    fProm=[]
    for i in range (0,sh[0]):
        fi=chauvenet(fuerzas[i,:])
        f=fi[~np.isnan(fi)].mean()
        fProm.append(f)
    #fProm=np.delete(fProm,indices)
    return RPM,fProm,mx

"""Filtrados"""
def filtrados(fuerza,bandwidth):
    data=pd.DataFrame({'RPM':dataProm(fuerza)[0],
                    fuerza:dataProm(fuerza)[1]})
    #bandwidth=100
    steps=np.linspace(500,np.amax(data['RPM']),13)
    indices=[(np.abs(data['RPM']-a).argmin()) for a in steps]
    meanR=[];stdR=[];meanf=[];stdf=[]
    for i in indices:
        j=int(i-bandwidth/2);k=int(i+bandwidth/2)
        mR=np.mean(data['RPM'][j:k])
        dR=np.std(data['RPM'][j:k])
        mf=np.mean(data[fuerza][j:k])
        df=np.std(data[fuerza][j:k])
        meanR=np.append(meanR,mR);stdR=np.append(stdR,dR)
        meanf=np.append(meanf,mf);stdf=np.append(stdf,df)
    return meanR,stdR,meanf,stdf

"""Curva Potencia"""
def curvaPotencia(bdw):
    w=filtrados('Torque (NÂ·m)',bdw)[0]
    T=filtrados('Torque (NÂ·m)',bdw)[2]
    P=T*w
    x=np.log(w);y=np.log(P)
    s=np.polyfit(x,y,1)
    C=np.exp(s[-1]);k=s[-2]
    return w,C*(w**k),C,k

"""Plot Raw"""
th='Thrust (N)'
to='Torque (NÂ·m)'
dataTh=datos(th)
dataTo=datos(to)
Ps=path_pwm
l=len(Ps)
for i in np.arange(l):
    plt.figure(1)
    plt.plot(dataTh[0],dataTh[1][:,i],'^',markersize=1,label=Ps[i])
    plt.legend(loc='lower right', bbox_to_anchor=(1, -0.5))
    plt.xlabel('Velocidad [RPM]');plt.ylabel('Thrust [N]'); plt.title(th)
for i in np.arange(l):
    plt.figure(2)
    plt.plot(dataTo[0],dataTo[1][:,i],'^',markersize=2,label=Ps[i])
    plt.legend(loc='lower right', bbox_to_anchor=(1, -0.5))
    plt.xlabel('Velocidad [RPM]');plt.ylabel(to);plt.title(to)


"""PLot Filtrado"""

bdw=200
meanR_th=filtrados(th,bdw)[0];stdR_th=filtrados(th,bdw)[1]
meanth=filtrados(th,bdw)[2];stdth=filtrados(th,bdw)[3]

meanR_to=filtrados(to,bdw)[0];stdR_to=filtrados(to,bdw)[1]
meanto=filtrados(to,bdw)[2];stdto=filtrados(to,bdw)[3]

fig, axes= plt.subplots(ncols=2,sharex=True,figsize=(15,5))
axes[0].errorbar(meanR_th, meanth, yerr=stdth,xerr=stdR_th,fmt='o',markersize=3, ecolor='g', capsize=5,capthick=2)
axes[0].set_xlabel('RPM');axes[0].set_ylabel(th);axes[0].grid()

axes[1].errorbar(meanR_to, meanto, yerr=stdto,xerr=stdR_to,fmt='o',markersize=3, ecolor='g', capsize=5,capthick=2)
axes[1].set_xlabel('RPM');axes[1].set_ylabel(to);axes[1].grid()

"""Plot Potencia"""
C=curvaPotencia(bdw)[2];k=curvaPotencia(bdw)[3]
plt.figure(4)
plt.plot(curvaPotencia(bdw)[0],curvaPotencia(bdw)[1])
plt.plot(meanR_to,meanto*meanR_to,'ro')
plt.xlabel('RPM');plt.ylabel('Power (W)');plt.title('$P=\omega ^%.3f'%(k));plt.grid()
D="{:.3e}".format(C)
print('P=',D,'(w^%.3f)'%(k))

plt.show(block=True)
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
import datetime
now=datetime.datetime.now()

"""Calling for paths"""
root = Tk()
root.filename = filedialog.askdirectory(title = "Select folder with data")
root.saveData=filedialog.askdirectory(title = "Select folder where you want to store results")
root.withdraw()

bdw=int(input("Por favor ingrese el ancho de banda para el filtrado de datos: "))
#T1s,T2s,T3s=input("Ingrese los datos medidos de temperatura (3): ").split(','); T1=float(T1s);T2=float(T2s);T3=float(T3s)
T1=17.6; T2=17.7; T3=18.1
#R1s,R2s,R3s=input("Ingrese los datos medidos de humedad relativa (3): ").split(','); R1=float(R1s);R2=float(R2s);R3=float(R3s)
R1=68.8;R2=68.7; R3=68.8
#P1s,P2s,P3s=input("Ingrese los datos medidos de presión atmosférica (3): ").split(','); P1=float(P1s);P2=float(P2s);P3=float(P3s)
P1=746.61; P2=746.62; P3=746.62

direc_pwm=os.path.join(root.filename,'*.csv'); path_pwm=[]
direc_optical=os.path.join(root.filename,'*.lvm'); path_optical=[]
direc_induced=os.path.join(root.filename,'*.XLS'); path_induced=[]

for fpwm in glob.glob(direc_pwm):
    path_pwm.append(fpwm)
for foptical in glob.glob(direc_optical):
    path_optical.append(foptical)
for finduced in glob.glob(direc_induced):
    path_induced.append(finduced)

"""Calling files_pwm"""
dfs_pwm={}
for seriePwm in path_pwm:
    seriePwm_pd = pd.read_csv(seriePwm, sep = ',',encoding='latin-1')
    dfs_pwm[seriePwm] = pd.DataFrame(seriePwm_pd)

"""Calling files_optical"""
dfs_optical={}
for serieOp in path_optical:
    serieOp_pd = pd.read_csv(serieOp, sep = ',',encoding='latin-1',names=['Time','RPM'])
    dfs_optical[serieOp] = pd.DataFrame(serieOp_pd)
"""Calling V_induced files"""
dfs_induced={}
for serieInduced in path_induced:
    serieInduced_pd = pd.read_csv(serieInduced,sep='\t',decimal=',',encoding='latin-1')
    dfs_induced[serieInduced] = pd.DataFrame(serieInduced_pd)

"""FUNCTIONS:"""
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx
"""raw data Dataframes"""
def rawDatos(fuerza):
    data=dfs_pwm
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


"""Aligning optical velocities"""
for i in range(1,len(path_optical)):
    optical1=dfs_optical[path_optical[0]]
    optical_df=optical1
    optical_i=dfs_optical[path_optical[i]]
    optical_i=optical1.align(optical_i['RPM'],join='inner',axis=0)[1]
    optical_df['optical'+str(i)]=optical_i
ind_low=find_closest(np.asarray(optical_df['Time']),9.)
ind_up=find_closest(np.asarray(optical_df['Time']),np.asarray(optical_df['Time'])[-1]-10)
optical_df=optical_df.drop(index=range(ind_up,len(optical_df['Time'])))
optical_dfRPM=optical_df.drop(index=range(ind_low),columns='Time')

"""Promediating Optical RPM"""
mOptical=optical_dfRPM.mean(axis=1, skipna=True)
RPMoptical_m=[];RPMoptical_std=[]
steps=np.linspace(500,np.amax(mOptical),13)
indices=[(np.abs(mOptical-a).argmin()) for a in steps]
for i in indices:
    j=int(i-bdw/2);k=int(i+bdw/2)
    RPMoptical_m=np.append(RPMoptical_m,np.mean(mOptical[j:k]))
    RPMoptical_std=np.append(RPMoptical_std,np.std(mOptical[j:k]))

"""Aligning induced velocities"""
induced1=dfs_induced[path_induced[0]]
induced_df=induced1
induced_df=induced_df.drop(columns=['Place','Date','Unit','Value.1','Unit.1','Value'])
for i in path_induced:
    induced_df['Vel induced'+str(path_induced.index(i))]=dfs_induced[i]['Value']
induced_dfRPM=induced_df.drop(columns='Time')

"""Promediating induced velocities"""
mInduced=induced_df.mean(axis=1, skipna=True)
RPMInduced_m=[];RPMInduced_std=[]
steps=np.linspace(0,np.amax(mInduced),13)
indices=find_closest(np.asarray(mInduced),steps)
for i in indices:
    j=int(i-5/2);k=int(i+5/2)
    if i==0:
        meanI=mInduced[i]
        stdI=0.0
    else:
        meanI=np.mean(mInduced[j:k])
        stdI=np.std(mInduced[j:k])
    RPMInduced_m=np.append(RPMInduced_m,meanI)
    RPMInduced_std=np.append(RPMInduced_std,stdI)

"""Curva Potencia"""
def curvaPotencia(w,bdw):
    T=filtrados('Torque (NÂ·m)',bdw)[2]
    P=T*w
    x=np.log(w/(2*np.pi/60));y=np.log(P)
    s=np.polyfit(x,y,1)
    C=np.exp(s[-1]);k=s[-2]
    return w,C*((w/(2*np.pi/60))**k),C,k


"""Density"""
def densidad (R,T,P):
    Rm=np.mean(R)/100; Tm=np.mean(T)+273.15; Pm=np.mean(P)*100
    Rd=np.std(R)/100; Td=np.std(T)+273.15; Pd=np.std(P)*100
    esm=(1.7526*10**-11)*np.exp(5315.56/Tm)
    esd=(1.7526*10**-11)*np.exp(5315.56/Td)
    rhom=(0.0034847/Tm)*(Pm-0.003796*Rm*esm)
    rhod=(0.0034847/Tm)*(Pd-0.003796*Rd*esd)
    rhosim=0.877
    err=(rhom-rhosim)*100/rhosim
    return rhom,rhod

"""PLotting"""

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
plt.savefig(os.path.join(root.saveData,'Raw Plotting Thrust vs PWM RPM ('+ now.strftime("%Y-%m-%d")+').jpg'))
for i in np.arange(l):
    plt.figure(2)
    plt.plot(dataTo[0],dataTo[1][:,i],'^',markersize=2,label=Ps[i])
    plt.legend(loc='lower right', bbox_to_anchor=(1, -0.5))
    plt.xlabel('Velocidad [RPM]');plt.ylabel(to);plt.title(to)
plt.savefig(os.path.join(root.saveData,'Raw Plotting Torque vs PWM RPM ('+ now.strftime("%Y-%m-%d")+').jpg'))


"""PLot Filtrado"""
meanR_th=filtrados(th,bdw)[0];stdR_th=filtrados(th,bdw)[1]
meanth=filtrados(th,bdw)[2];stdth=filtrados(th,bdw)[3]

meanR_to=filtrados(to,bdw)[0];stdR_to=filtrados(to,bdw)[1]
meanto=filtrados(to,bdw)[2];stdto=filtrados(to,bdw)[3]

fig, axes= plt.subplots(ncols=2,sharex=True,figsize=(15,5))
axes[0].errorbar(meanR_th, meanth, yerr=stdth,xerr=stdR_th,fmt='o',markersize=3, ecolor='g', capsize=5,capthick=2,label='PWM')
axes[0].errorbar(RPMoptical_m, meanth, yerr=stdth,xerr=RPMoptical_std,fmt='^',markersize=3, ecolor='r', capsize=5,capthick=2,label='optical')
axes[0].set_xlabel('RPM');axes[0].set_ylabel(th);axes[0].grid(); axes[0].legend()

axes[1].errorbar(meanR_to, meanto, yerr=stdto,xerr=stdR_to,fmt='o',markersize=3, ecolor='g', capsize=5,capthick=2)
axes[1].errorbar(RPMoptical_m, meanto, yerr=stdto,xerr=RPMoptical_std,fmt='^',markersize=3, ecolor='r', capsize=5,capthick=2,label='optical')
axes[1].set_xlabel('RPM');axes[1].set_ylabel(to);axes[1].grid()
plt.savefig(os.path.join(root.saveData,'Filtered Torque and Thrust vs RPM ('+ now.strftime("%Y-%m-%d")+').jpg'))

"""Plot Potencia"""
C_pwm=curvaPotencia(meanR_to*(2*np.pi/60),bdw)[2];k_pwm=curvaPotencia(meanR_to*(2*np.pi/60),bdw)[3]
C_optical=curvaPotencia(RPMoptical_m*(2*np.pi/60),bdw)[2];k_pwm=curvaPotencia(RPMoptical_m*(2*np.pi/60),bdw)[3]
omegath_pwm=meanR_th*(2*np.pi/60); omega_optical=RPMoptical_m*(2*np.pi/60)
omegato_pwm=meanR_to*(2*np.pi/60)
P_real_pwm=meanto*omegato_pwm
P_real_optical=meanto*omega_optical

plt.figure(4)
plt.plot(curvaPotencia(omegato_pwm,bdw)[0]*60/(2*np.pi),
        curvaPotencia(omegato_pwm,bdw)[1],
        label='$P_{pwm}=$'+'{:.3e}'.format(C_pwm)+'$\\times \omega^{%.3f}$'%(k_pwm))
plt.plot(meanR_to,P_real_pwm,'ro', label='pwm')
plt.plot(RPMoptical_m,P_real_optical,'go',label='optical')
plt.xlabel('RPM');plt.ylabel('Power (W)');plt.grid(); plt.legend()
#plt.title('$P=$'+'{:.3e}'.format(C_pwm)+'$\\times \omega^{%.3f}$'%(k_pwm)); #plt.savefig('Curva de Potencia PWM.jpg')
#"""print('P=',D,'(w^%.3f)'%(k))"""
plt.savefig(os.path.join(root.saveData,'Power curve ('+ now.strftime("%Y-%m-%d")+').jpg'))

"""Plot induced"""
plt.figure(5)
plt.errorbar(RPMoptical_m, RPMInduced_m, yerr=RPMInduced_std,xerr=RPMoptical_std,fmt='^',markersize=3, ecolor='b', capsize=5,capthick=2)
plt.xlabel('RPM_Optical');plt.ylabel('Induced Velocity [m/s]');plt.grid()
plt.savefig(os.path.join(root.saveData,'Induced Velocity ('+ now.strftime("%Y-%m-%d")+').jpg'))

"""Plot M"""
diameter=0.36
Area=np.pi*diameter**2/4
rhom=densidad([R1,R2,R3],[T1,T2,T3],[P1,P2,P3])[0]
P_induced=meanth*RPMInduced_m
P_induced_ideal=(meanth**1.5)/np.sqrt(2*rhom*Area)
Mm_pwm=(P_induced_ideal/P_real_pwm)
Mm_optical=(P_induced_ideal/P_real_optical)
#Mm=((meanth**1.5)/np.sqrt(2*rhom*(np.pi*(0.3**2)/4)))/(meanth*RPMInduced_m)
plt.figure(6)
plt.plot(meanR_th,Mm_pwm,'ob', label='PWM')
plt.plot(RPMoptical_m,Mm_optical,'og', label='Optical')
plt.xlabel('RPM');plt.ylabel('M');plt.legend();plt.grid()
plt.gca().set_ylim([0.6,1])
plt.savefig(os.path.join(root.saveData,'Figure of Merit ('+ now.strftime("%Y-%m-%d")+').jpg'))
print(rhom)
print(RPMoptical_m)

Ct_pwm=meanth/(rhom*Area*(omegath_pwm*(diameter/2))**2)
Ct_optical=meanth/(rhom*Area*(omega_optical*(diameter/2))**2)
Cp_pwm=P_real_pwm/(rhom*Area*(omegath_pwm*(diameter/2))**3)
Cp_optical=P_real_optical/(rhom*Area*(omega_optical*(diameter/2))**3)
#Cpd_pwm=((1/Mm)-1)*Ct_pwm**1.5/np.sqrt(2);Cpd_optical=((1/Mm)-1)*Ct_optical**1.5/np.sqrt(2)
#Cpd_ideal=Ct_pwm**1.5/np.sqrt(2)
plt.figure(7)
plt.plot(meanR_th,Ct_pwm,'or', label='PWM')
plt.plot(RPMoptical_m,Ct_optical,'og',label='Optical')
plt.xlabel('RPM');plt.ylabel('Thrust coeficient');plt.legend();plt.grid()
plt.savefig(os.path.join(root.saveData,'Thrust coeficient ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.figure(8)
plt.plot(meanR_th,Cp_pwm,'or', label='PWM')
plt.plot(RPMoptical_m,Cp_optical,'og',label='Optical')
#plt.plot(RPMoptical_m,Cpd_ideal,'-b',label='model')
plt.xlabel('RPM');plt.ylabel('Power coeficient');plt.legend();plt.grid()
plt.savefig(os.path.join(root.saveData,'Power Coeficient ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.figure(9)
plt.plot(meanR_th,meanth*RPMInduced_m,'or', label='PWM')
plt.plot(RPMoptical_m,meanth*RPMInduced_m,'og',label='Optical')
plt.xlabel('RPM');plt.ylabel('Induced Power');plt.legend();plt.grid()
plt.savefig(os.path.join(root.saveData,'Induced_power ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.figure(10)
plt.plot(meanR_th,Ct_pwm,'or', label='PWM')
plt.plot(RPMoptical_m,Ct_optical,'og',label='Optical')
plt.xlabel('RPM');plt.ylabel('Thrust coeficient');plt.legend();plt.grid()
plt.gca().set_xlim([4500,6500]); plt.gca().set_ylim([0.01,0.015])
plt.savefig(os.path.join(root.saveData,'Zoomed Thrust coeficient ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.figure(11)
plt.plot(meanR_th,Cp_pwm,'or', label='PWM')
plt.plot(RPMoptical_m,Cp_optical,'og',label='Optical')
#plt.plot(RPMoptical_m,Cpd_ideal,'-b',label='model')
plt.xlabel('RPM');plt.ylabel('Power coeficient');plt.legend();plt.grid()
plt.gca().set_xlim([4500,6500])
plt.savefig(os.path.join(root.saveData,'Zoomed Power Coeficient ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.show(block=True)

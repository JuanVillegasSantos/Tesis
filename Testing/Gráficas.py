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
import warnings
warnings.simplefilter('ignore', np.RankWarning)

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
offth=1.2;offto=0.035

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
    T=filtrados('Torque (NÂ·m)',bdw)[2]+offto
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

"""Data for plotting"""

"""definition"""
th='Thrust (N)'
to='Torque (NÂ·m)'
dataTh=datos(th)
dataTo=datos(to)
Ps=path_pwm
l=len(Ps)

"""Filtered Data"""


meanR_th=filtrados(th,bdw)[0];stdR_th=filtrados(th,bdw)[1]
meanth=filtrados(th,bdw)[2]+offth;stdth=filtrados(th,bdw)[3]
meanR_to=filtrados(to,bdw)[0];stdR_to=filtrados(to,bdw)[1]
meanto=filtrados(to,bdw)[2]+offto;stdto=filtrados(to,bdw)[3]

"""Power Data"""
C_pwm=curvaPotencia(meanR_to*(2*np.pi/60),bdw)[2];k_pwm=curvaPotencia(meanR_to*(2*np.pi/60),bdw)[3]
C_optical=curvaPotencia(RPMoptical_m*(2*np.pi/60),bdw)[2]
#;k_optical=curvaPotencia(RPMoptical_m*(2*np.pi/60),bdw)[3]
#k_pwm=2.0875

omegath_pwm=meanR_th*(2*np.pi/60); err_omegath_pwm=stdR_th*(2*np.pi/60)
omegato_pwm=meanR_to*(2*np.pi/60); err_omegato_pwm=stdR_to*(2*np.pi/60)
omega_optical=RPMoptical_m*(2*np.pi/60); err_omega_optical=RPMoptical_std*(2*np.pi/60)

P_real_pwm=meanto*omegato_pwm
err_P_real_pwm=((meanto*err_omegato_pwm)**2+(stdto*omegato_pwm)**2)**0.5

P_real_optical=meanto*omega_optical
err_P_real_optical=((meanto*err_omega_optical)**2+(stdto*omega_optical)**2)**0.5

"""Figure of Merit Data"""
diameter=0.36
Area=np.pi*diameter**2/4
rhom=densidad([R1,R2,R3],[T1,T2,T3],[P1,P2,P3])[0]
P_induced=meanth*RPMInduced_m

P_induced_ideal=(meanth**1.5)/np.sqrt(2*rhom*Area)
err_P_induced_ideal=np.sqrt(meanth/(2*rhom*Area))*stdth

Mm_pwm=(P_induced_ideal/P_real_pwm)
err_M_pwm=((err_P_induced_ideal/P_real_pwm)**2+(P_induced_ideal*err_P_real_pwm/P_real_pwm)**2)**0.5
#err_M_pwm=err_P_induced_ideal/P_real_pwm

Mm_optical=(P_induced_ideal/P_real_optical)
err_M_optical=((err_P_induced_ideal/P_real_optical)**2+(P_induced_ideal*err_P_real_optical/P_real_optical)**2)**0.5
#err_M_optical=err_P_induced_ideal/P_real_optical

"""Thrust Coeficient Data"""
Ct_pwm=meanth/(rhom*Area*(omegath_pwm*(diameter/2))**2)
Ct_optical=meanth/(rhom*Area*(omega_optical*(diameter/2))**2)
"""Power Coeficient Data"""
Cp_pwm=P_real_pwm/(rhom*Area*(omegath_pwm*(diameter/2))**3)
Cp_optical=P_real_optical/(rhom*Area*(omega_optical*(diameter/2))**3)

"""Efficiency data"""
Volt=15
current=[0.03,1.19,1.57,1.87,2.35,3.27,4.15,5.68,7.5,8.38,9.49,12.67,14.97]

P_elec=[Volt*i for i in current]
eff_pwm=P_real_pwm/P_elec
eff_optical=P_real_optical/P_elec

"""PLotting"""

"""Plot Raw"""
for i in np.arange(l):
    plt.figure(1)
    plt.plot(dataTh[0],dataTh[1][:,i]+offth,'^',markersize=1,label=Ps[i])
    plt.legend(loc='lower right', bbox_to_anchor=(1, -0.5))
    plt.xlabel('Velocidad [RPM]');plt.ylabel('Thrust [N]'); plt.title(th)
plt.savefig(os.path.join(root.saveData,'Raw Plotting Thrust vs PWM RPM ('+ now.strftime("%Y-%m-%d")+').jpg'))
for i in np.arange(l):
    plt.figure(2)
    plt.plot(dataTo[0],dataTo[1][:,i]+offto,'^',markersize=2,label=Ps[i])
    plt.legend(loc='lower right', bbox_to_anchor=(1, -0.5))
    plt.xlabel('Velocidad [RPM]');plt.ylabel(to);plt.title(to)
plt.savefig(os.path.join(root.saveData,'Raw Plotting Torque vs PWM RPM ('+ now.strftime("%Y-%m-%d")+').jpg'))


"""PLot Filtrado"""
fig, axes= plt.subplots(ncols=2,sharex=True,figsize=(15,5))
axes[0].errorbar(meanR_th, meanth, yerr=stdth,xerr=stdR_th,fmt='o',markersize=3, ecolor='g', capsize=5,capthick=2,label='PWM')
axes[0].errorbar(RPMoptical_m, meanth, yerr=stdth,xerr=RPMoptical_std,fmt='o',markersize=3, ecolor='r', capsize=5,capthick=2,label='optical')
axes[0].set_xlabel('RPM');axes[0].set_ylabel(th);axes[0].grid(); axes[0].legend()

axes[1].errorbar(meanR_to, meanto, yerr=stdto,xerr=stdR_to,fmt='o',markersize=3, ecolor='g', capsize=5,capthick=2)
axes[1].errorbar(RPMoptical_m, meanto, yerr=stdto,xerr=RPMoptical_std,fmt='^',markersize=3, ecolor='r', capsize=5,capthick=2,label='optical')
axes[1].set_xlabel('RPM');axes[1].set_ylabel(to);axes[1].grid()
plt.savefig(os.path.join(root.saveData,'Filtered Torque and Thrust vs RPM ('+ now.strftime("%Y-%m-%d")+').jpg'))

"""Plot Potencia"""
plt.figure(4)
plt.plot(curvaPotencia(omegato_pwm,bdw)[0]*60/(2*np.pi),
        curvaPotencia(omegato_pwm,bdw)[1],
        label='$P_{pwm}=$'+'{:.3e}'.format(C_pwm)+'$\\times \omega^{%.3f}$'%k_pwm)
#PWM
plt.errorbar(meanR_to,P_real_pwm,yerr=err_P_real_pwm,xerr=stdR_to,fmt='og', label='pwm')
#Optical
plt.errorbar(RPMoptical_m,P_real_optical,yerr=err_P_real_optical,xerr=RPMoptical_std,fmt='or',label='optical')

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
#Mm=((meanth**1.5)/np.sqrt(2*rhom*(np.pi*(0.3**2)/4)))/(meanth*RPMInduced_m)
plt.figure(6)
#PWM
plt.plot(meanR_th,Mm_pwm,'og', label='PWM')
#Optical
plt.plot(RPMoptical_m,Mm_optical,'or', label='Optical')
plt.xlabel('RPM');plt.ylabel('M');plt.legend();plt.grid()
plt.gca().set_xlim([1500,6500]);plt.gca().set_ylim([0.48,0.7])
plt.savefig(os.path.join(root.saveData,'Figure of Merit ('+ now.strftime("%Y-%m-%d")+').jpg'))

"""Plot Thrust Coeficient"""
#Cpd_pwm=((1/Mm)-1)*Ct_pwm**1.5/np.sqrt(2);Cpd_optical=((1/Mm)-1)*Ct_optical**1.5/np.sqrt(2)
#Cpd_ideal=Ct_pwm**1.5/np.sqrt(2)
plt.figure(7)
plt.plot(meanR_th,Ct_pwm,'og', label='PWM')
plt.plot(RPMoptical_m,Ct_optical,'or',label='Optical')
plt.xlabel('RPM');plt.ylabel('Thrust coeficient');plt.legend();plt.grid()
plt.savefig(os.path.join(root.saveData,'Thrust coeficient ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.figure(8)
plt.plot(meanR_th,Cp_pwm,'og', label='PWM')
plt.plot(RPMoptical_m,Cp_optical,'or',label='Optical')
#plt.plot(RPMoptical_m,Cpd_ideal,'-b',label='model')
plt.xlabel('RPM');plt.ylabel('Power coeficient');plt.legend();plt.grid()
plt.savefig(os.path.join(root.saveData,'Power Coeficient ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.figure(9)
plt.plot(meanR_th,meanth*RPMInduced_m,'og', label='PWM')
plt.plot(RPMoptical_m,meanth*RPMInduced_m,'or',label='Optical')
plt.xlabel('RPM');plt.ylabel('Induced Power');plt.legend();plt.grid()
plt.savefig(os.path.join(root.saveData,'Induced_power ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.figure(10)
plt.plot(meanR_th,Ct_pwm,'og', label='PWM')
plt.plot(RPMoptical_m,Ct_optical,'or',label='Optical')
plt.xlabel('RPM');plt.ylabel('Thrust coeficient');plt.legend();plt.grid()
plt.gca().set_xlim([1600,6500]); plt.gca().set_ylim([0.008,0.02])
plt.savefig(os.path.join(root.saveData,'Zoomed Thrust coeficient ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.figure(11)
plt.plot(meanR_th,Cp_pwm,'og', label='PWM')
plt.plot(RPMoptical_m,Cp_optical,'or',label='Optical')
#plt.plot(RPMoptical_m,Cpd_ideal,'-b',label='model')
plt.xlabel('RPM');plt.ylabel('Power coeficient');plt.legend();plt.grid()
plt.gca().set_xlim([1500,6500]); plt.gca().set_ylim([0.001,0.003])
plt.savefig(os.path.join(root.saveData,'Zoomed Power Coeficient ('+ now.strftime("%Y-%m-%d")+').jpg'))

vi_ideal=np.sqrt(meanth/(2*rhom*Area))
Cp_pwm_mod_ideal=P_real_pwm/(rhom*Area*(vi_ideal)**3)
Cp_optical_mod_ideal=P_real_optical/(rhom*Area*(vi_ideal)**3)

Cp_pwm_mod_medido=P_real_pwm/(rhom*Area*(RPMInduced_m)**3)
Cp_optical_mod_medido=P_real_optical/(rhom*Area*(RPMInduced_m)**3)

plt.figure(12)
plt.plot(meanR_th,Cp_pwm_mod_ideal,'og', label='PWM')
plt.plot(RPMoptical_m,Cp_optical_mod_ideal,'or',label='Optical')
plt.title('Cp modificado')
#plt.plot(RPMoptical_m,Cpd_ideal,'-b',label='model')
plt.xlabel('RPM');plt.ylabel('Power coeficient modified');plt.legend();plt.grid()
plt.savefig(os.path.join(root.saveData,'Power Coeficient modified ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.figure(13)
plt.plot(meanR_th,eff_pwm,'og', label='PWM')
plt.plot(RPMoptical_m,eff_optical,'or',label='Optical')
plt.title('eficiencia global')
plt.gca().set_xlim([1000,6500]); plt.gca().set_ylim([0.2,1])
#plt.plot(RPMoptical_m,Cpd_ideal,'-b',label='model')
plt.xlabel('RPM');plt.ylabel('eficiencia');plt.legend();plt.grid()
plt.savefig(os.path.join(root.saveData,'Eficiencia ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.figure(14)
plt.plot(meanR_th,current,'og', label='PWM')
plt.plot(RPMoptical_m,current,'or',label='Optical')
plt.title('Corriente')
#plt.plot(RPMoptical_m,Cpd_ideal,'-b',label='model')
plt.xlabel('RPM');plt.ylabel('Current (A)');plt.legend();plt.grid()
plt.savefig(os.path.join(root.saveData,'Current ('+ now.strftime("%Y-%m-%d")+').jpg'))

plt.figure(15)
plt.plot(meanR_th,P_elec,'og', label='PWM')
plt.plot(RPMoptical_m,P_elec,'or',label='Optical')
plt.title('Potencia electrica')
#plt.plot(RPMoptical_m,Cpd_ideal,'-b',label='model')
plt.xlabel('RPM');plt.ylabel('Power (W)');plt.legend();plt.grid()
plt.savefig(os.path.join(root.saveData,'Electrical ('+ now.strftime("%Y-%m-%d")+').jpg'))

print('errors Thrust')
print(stdth)
print('errors torque')
print(stdto)
print('PWM')
print(meanR_to)
print('optical')
print(RPMoptical_m)
print('error power_pwm')
print(err_P_real_pwm)
print('error power_optical')
print(err_P_real_optical)

"""Final DataFrames"""
resulting_data = {'RPM_pwm':meanR_th , 'omega_pwm(rad/s)': omegath_pwm,'Error_RPM_pwm': stdR_th,
                'RPM_optico': RPMoptical_m,'omega_optico(rad/s)':omega_optical, 'Error_RPM_optico':RPMoptical_std,
                'Thrust (N)':meanth, 'Error_Thrust':stdth,
                'Torque(Nm)':meanto, 'Error_Torque':stdto,
                'Potencia_PWM (W)':P_real_pwm,'Error_Potencia_PWM':err_P_real_pwm,
                'Potencia_optica (W)':P_real_optical, 'Error_Potencia_optica': err_P_real_optical,
                'Velocidad_inducida (m/s)':RPMInduced_m, 'Error_Velocidad_inducida':RPMInduced_std,
                'Merito_PWM':Mm_pwm,'Merito_optica':Mm_optical,
                'Ct_pwm':Ct_pwm,'Ct_optico':Ct_optical,
                'Cp_pwm':Cp_pwm, 'Cp_optico':Cp_optical,
                'Cp_modificado_pwm':Cp_pwm_mod_ideal, 'Cp_modificado_optico':Cp_optical_mod_ideal,
                'Corriente(A)':current,'Potencia eléctrica (W)':P_elec,'Eficiencia_PWM':eff_pwm,
                'Eficiencia_optica':eff_optical}
dataResume=pd.DataFrame(data=resulting_data)
dataResume.to_csv(os.path.join(root.saveData,'Datos ('+ now.strftime("%Y-%m-%d")+').csv'))
print(root.filename)
print(k_pwm)
plt.show(block=True)

# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:27:32 2018
Frontend Pololu and RCBenchmark controller
@author: Juan Sebastián Villegas
"""
from tkinter import *
from PIL import Image, ImageTk
import serial
import sys
import glob

"""
Pololu
"""
"""Checking for ports"""
def serial_ports():
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in list(range(256))]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')
    result = []
    for port in ports:
        try:
            #print("checking port "+port)
            s = serial.Serial(port, rtscts=True, dsrdtr=True)
            #print("closing port "+port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result
"""Pololu Functions"""
class Controller:
    def __init__(self,ttyStr='/dev/ttyACM0',device=0x0c):
        #ttyStr=portName.get()
        self.usb = serial.Serial(ttyStr)
        self.PololuCmd = chr(0xaa) + chr(device)
        self.Targets = [0] * 24
        self.Mins = [0] * 24
        self.Maxs = [0] * 24
    def sendCmd(self, cmd):
        cmdStr = self.PololuCmd + cmd
        self.usb.write(bytes(cmdStr,'latin-1'))
    def runScriptSub(self, subNumber):
        cmd = chr(0x27) + chr(subNumber)
        # can pass a param with command 0x28
        # cmd = chr(0x28) + chr(subNumber) + chr(lsb) + chr(msb)
        self.sendCmd(cmd)
""" Connecting to Pololu and setting Target"""
def openPololu():
    global targ
    targ=s1.get()
    n=4*int(float(targ))
    if n>=1025*4:
        z=bin(n)
    elif n<1025*4:
        z=bin(1025*4)
    else:
        print('Si eres bueno en algo, no lo hagas gratis')
    f=open(serial_ports()[serial_ports().index(portNameC.get())],mode='wb')
    binary=[132,0,int(z[8:],2),int(z[2:8],2)]
    f.write(bytearray(binary))
    f.close()
    print (portNameC.get())
""" Running a Script"""
def runScript():
    m = Controller(portNameC.get())
    m.runScriptSub(2)

"""
RCBenchmark
"""



"""
Front End
"""
window=Tk()
window.title('Automation interface')
"""Seleccionar Puerto Serial Controlador"""
puertoPololu= Label(window,text='Puerto Controlador: '); puertoPololu.grid(row=0,column=0)

portNameC=StringVar()
COMPololu=OptionMenu(window,portNameC,*serial_ports());COMPololu.grid(row=0,column=1)


"""Seleccionar Adquisición"""
puertoAq= Label(window,text='Puerto Adquisición: '); puertoAq.grid(row=1,column=0)
portNameAq=StringVar()
COMAq=OptionMenu(window,portNameAq,*serial_ports());COMAq.grid(row=1,column=1)
dataB=Button(window,text='Save Data',width=20);dataB.grid(row=1,column=2)


"""Seleccionar Target del Ancho de Pulso"""
targetL=Label(window,text='Target: ');targetL.grid(row=2,column=0)

b1=Button(window,text='Set Target',width=20,
          command=openPololu,state='active');b1.grid(row=2,column=3)

s1=Scale(window,length=400,orient=HORIZONTAL,
         resolution=10,from_=1000,to=1500);s1.grid(row=2,column=1,columnspan=2)

"""Run Script"""
Script1L=Label(window,text='Scripts: ');Script1L.grid(row=3,column=0)

Script1B=Button(window,text='Full range RPM',width=20,
          command=runScript,state='active');Script1B.grid(row=3,column=1)

"""Post Processing"""
PostL=Label(window,text='PostProcessing: ');PostL.grid(row=4,column=0)
RawToB=Button(window,text='Plot Raw Torque',width=20);RawToB.grid(row=4,column=1)
RawThB=Button(window,text='Plot Raw Thrust',width=20);RawThB.grid(row=4,column=2)
fltToB=Button(window,text='Filtered Torque',width=20);fltToB.grid(row=5,column=1)
fltThB=Button(window,text='Filtered Thrust',width=20);fltThB.grid(row=5,column=2)
PwrB=Button(window,text='Power curve',width=20);PwrB.grid(row=6,column=1,columnspan=2)
ResB=Button(window,text='Save Results',width=20);ResB.grid(row=4,column=3,rowspan=3)


window.mainloop()

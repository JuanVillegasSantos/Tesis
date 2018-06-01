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
Functions
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
    f=open(serial_ports()[serial_ports().index(portName.get())],mode='wb')
    binary=[132,0,int(z[8:],2),int(z[2:8],2)]
    f.write(bytearray(binary))
    f.close()


"""
Front End
"""
window=Tk()
window.title('Automation interface')

l1= Label(window,text='Puerto: ')
l1.grid(row=0,column=0)

l2=Label(window,text='Target: ')
l2.grid(row=1,column=0)

portName=StringVar()
COMs=OptionMenu(window,portName,*serial_ports())
COMs.grid(row=0,column=1)

target=IntVar()
e=Entry(window,textvariable=target)
e.grid(row=1,column=1,columnspan=2)
 
b1=Button(window,text='Start',width=20,
          command=openPololu,state='active')
b1.grid(row=0,column=2)

s1=Scale(window,length=400,orient=HORIZONTAL,
         resolution=10,from_=1000,to=1500)
s1.grid(row=1,column=1,columnspan=2)


window.mainloop()

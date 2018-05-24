# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:27:32 2018

@author: user
"""
import tkinter
from tkinter import *
from PIL import Image, ImageTk


window=Tk()
window.title('Automation interface')
window.configure(background='white')

"""
img=ImageTk.PhotoImage(Image.open("./title.GIF"))
#photo=PhotoImage(image)
ph=Label(window,image=img,bg='white')
ph.image=photo
ph.pack()
"""
l1= Label(window,text='Title')
l1.grid(row=1,column=0)

l2= Label(window,text='Author')
l2.grid(row=1,column=1)

title_text=StringVar()
e1=Entry(window,textvariable=title_text)
e1.grid(row=1,column=1)

b1=Button(window,text='Start',width=20)
b1.grid(row=1,column=2)

b2=Button(window,text='Stop',width=20)
b2.grid(row=3,column=2)

window.mainloop()
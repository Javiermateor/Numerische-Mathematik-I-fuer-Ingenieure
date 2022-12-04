import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
import time
from IPython import display

#Erste Programmieraufgabe 

#Einstellungen für die Darstellung
plt.style.use('seaborn')
def f(t,y):
    f = np.cos(t)*y
    return f

def loeservergleich(h):
    k = int(50/h)
    # Anfangswerte:
    euler = np.array([[0,1]],dtype=float)
    collatz = np.array([[0,1]],dtype=float)
    heun = np.array([[0,1]],dtype=float)
    for i in range(k):
        # expliziertes Eulerverfahren
        euler = np.append(euler, [[euler[i,0]+h, euler[i,1]+h*f(euler[i,0], euler[i,1])]], axis=0)
        # Collatzverfahren
        collatz = np.append(collatz, [[collatz[i,0]+h, collatz[i,1]+h*f(collatz[i,0]+(h/2),collatz[i,1]+(h/2)*f(collatz[i,0], collatz[i,1]))]], axis=0)
        # Heunverfahren
        heun = np.append(heun, [[heun[i,0]+h, heun[i,1]+(h/2)*(f(heun[i,0], heun[i,1])+f(heun[i,0]+h,heun[i,1]+h*f(heun[i,0], heun[i,1])))]], axis=0)
    
    
    # Plot der Funktion und Annäherungen
    
    fig1 = plt.figure(figsize=(8,6))
    x = np.linspace(0,50,k)
    plt.plot(x,np.exp(np.sin(x)),  color='yellow', linewidth=7, label="Lösungsfunktion")
    plt.plot(euler[:,0], euler[:,1], color='green', linewidth=2, label="Euler (explizit)")
    plt.plot(collatz[:,0], collatz[:,1], color='red', linewidth=2, label="Collatz")
    plt.plot(heun[:,0], heun[:,1], color='blue', linewidth=2, label="Heun")
    plt.legend(loc="upper right")
    plt.xlim(0,50)
    plt.xlabel("t") 
    plt.ylabel("y")
    plt.title("Plot der Funktion und verschiedene Annäherungen", pad ='15')
    plt.rcParams["figure.figsize"] = (15,15)
    plt.show()

# #Test der Funktion
loeservergleich(0.5)

#Zweite Programmieraufgabe 

def doppelsternn(M1,M2,x1,x2,p,h):
    # Initialisierung der Werte
    global m1
    global m2
    global g
    g = 1 # Vorgabe aus VL
    m1 = M1
    m2 = M2

    global y
    y = np.array([[[x1,0],[x2,0],[0,p/m1],[0,-p/m2]]],dtype=float) # Anfangsort und Anfangsgeschwindigkeiten
    
    #Plot der Bahnen 
    while True:
        try:
            #print(y[-1,0,0], y[-1,0,1])
            #plt.subplot(1)
            plt.plot(y[-1,0,0], y[-1,0,1], 'ro')
            plt.plot(y[-1,1,0], y[-1,1,1], 'bo')
            plt.pause(0.01)
            plt.clf()
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            y = np.append(y, [ruku_schritt(y, h)],  axis=0)
            
        except KeyboardInterrupt:
            break
    plt.show()

def ruku_schritt(y,h):
    '''
    Input:, nx4x2 Vektoren, h = Schrittweite
    Output: nx4x2 Vektor für neue y- Wert (2xOrtskoordinate, 2x Geschwiendigkeit)
    '''
    k1 = gravi(y[-1])
    k2 = gravi(y[-1] + 0.5*h*k1)
    k3 = gravi(y[-1] + 0.5*h*k2)
    k4 = gravi(y[-1] + h*k3)
    yneu = y[-1] + h*(k1/6 + k2/3 + k3/3 + k4/6)
    return yneu

def gravi(y):
    '''
    Input: y = 4x2 also vier n=2 Vektoren
    Output:  Vekor der Kraft
    '''
    gravi = np.zeros((4,2))
    gravi[0] = y[2]
    gravi[1] = y[3]
    gravi[2] = F(y)/m1
    gravi[3] = -F(y)/m2
    return gravi

def F(y): # für f1 sonst -F(y) für f2
    F = ((g*m1*m2)/((LA.norm(y[1]-y[0],2))**3))*(y[1]-y[0])
    return F

m1 = 1
m2 = 5
x1 = -1
x2= 1
p= 1
h = 0.01

doppelsternn(m1,m2,x1,x2,p,h)

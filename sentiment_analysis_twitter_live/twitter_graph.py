#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#live graphing twitter sentiment

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    data = open('twitter_output.txt','r').read()
    lines = data.split('\n')[:-1]
    if len(lines) > 200:
        lines = lines[-200:]
    xarr,yarr = [],[]
    x,y =0,0
    for line in lines:
        x += 1
        s,c = line.split(',')
        c = float(c)
        if s == 'pos':
            y += c
        elif s == 'neg':
            y -= c
        xarr.append(x)
        yarr.append(y)
    
    ax1.clear()
    ax1.plot(xarr,yarr)
    
ani = animation.FuncAnimation(fig,animate,interval = 1000)
plt.show()

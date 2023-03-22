#
# Copyright 2016, 2021 Lars Pastewka (U. Freiburg)
#           2016 Adrien Gola (KIT)
#
# matscipy - Materials science with Python at the atomic-scale
# https://github.com/libAtoms/matscipy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


"""
 Usage : python plot.py [filename]
 
 This tool script allow to quicly plot data from text files arranged in column with space as separator. It uses a GUI through Tkinter
 "log.lammp" can be read and splited according to different "run" present in the log file
                                                                                            
 Adrien Gola // 4.11.2015                                                                        
"""

import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as p 

import os,sys
import os.path

import Tkinter as tk
import ttk

# ------------------------------------
# ------- Function definition --------
# ------------------------------------
def close_windows():
    root.destroy()

def plot_button():
    global xlabel,ylabel
    x = d[:,Xvar.get()]
    y = d[:,Yvar.get()]
  
    if xlabel.get() != "X-axis label":
        hx = xlabel.get()
    else:
        hx = h[Xvar.get()]
    if ylabel.get() != "Y-axis label":
        hy = ylabel.get()
    else:
        hy = h[Yvar.get()]
  
    p.plot(x, y, 'o')
    p.xlabel(hx)
    p.ylabel(hy)
    p.show() 
  
def process_file():
    global variables_frame,h,d,Xvar,Yvar,skipline,flag_loglammps
    selected_file = sorted(outlogfiles)[Fvar.get()]
    if flag_loglammps:
        h = open(selected_file,'r').readlines()[0].strip('#').strip().split(" ") # headers list 
        d = np.loadtxt(selected_file, skiprows=1) # data  
    else:
        h = filter(None,open(selected_file,'r').readlines()[0+int(skipline.get())].strip('#').strip().split(" ")) # headers list 
        d = np.loadtxt(selected_file, skiprows=1+int(skipline.get())) # data   
    try:
        for child in variables_frame.winfo_children():
            child.destroy()
    except:
        pass
    #Xvar
    tk.Label(variables_frame,text="Select X-axis variable to plot",anchor='n').grid(row=next_row+1)
    Xvar = tk.IntVar()
    for i,x in enumerate(h):
        x_row=i+2
        tk.Radiobutton(variables_frame,text=x,variable=Xvar, value=i).grid(row=next_row+x_row)
    #Yvar
    tk.Label(variables_frame,text="Select Y-axis variable to plot",anchor='n').grid(row=next_row+1,column=1)
    Yvar = tk.IntVar()
    for j,x in enumerate(h):
        y_row=j+2
        tk.Radiobutton(variables_frame,text=x,variable=Yvar, value=j).grid(row=next_row+y_row,column=1)


mypath = os.getcwd()
# -------------------------------------------------------
# ------- Reading and spliting the main log file --------
# -------------------------------------------------------
flag_loglammps=0
# --- file selection ---
try :
    in_file = sys.argv[1]
except:
    print("Usage : plot_general.py file_name")
    quit()

if in_file == "log.lammps":
    flag_loglammps=1
    data = open("log.lammps",'r').readlines()  # select log.lammps as main log file if it exists
      
    # --- spliting into "out.log.*" files ---
    j = 0
    flag = 0
    flag_out = 0
    step = ""
    for lines in data:
        lines = lines.strip()
        if lines.startswith("run:"):
            step = "."+lines.strip().strip("run: ")
        elif lines.startswith("Memory usage per processor"):
          if len(str(j)) == 1:
              J = "0"+str(j)
          else:
              J = str(j)
          out = open("out.log.%s%s-plt-tmp"%(J,step),'w')
          flag_out = 1
          flag = 1
        elif flag and lines.startswith("Loop") or lines.startswith("kill"):
            flag = 0
            j+=1
            out.close()
            step = ""
        elif flag:
            out.write(lines+'\n')
    if flag_out:
        out.close()
    outlogfiles = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) and f.startswith("out.log") and f.endswith("plt-tmp")] # list of "out.log.*" files
else:
    outlogfiles = [in_file]
    pass
  




###################
### MAIN script ###
###################

root=tk.Tk()
# Positioning frames
button_frame = tk.Frame(root,heigh=5)
button_frame.pack(side='right')

source_frame = tk.Frame(root)
source_frame.pack()

labels_frame = tk.Frame(root)
labels_frame.pack(side="bottom")

variables_frame = tk.Frame(root)
variables_frame.pack(side="bottom")

# Filling frames
tk.Label(source_frame,text="Select data source",anchor='n').grid(row=0)
Fvar = tk.IntVar()
for i,x in enumerate(outlogfiles):
    next_row=i+1
    tk.Radiobutton(source_frame,text=x,variable=Fvar, value=i, command = process_file).grid(row=next_row)
    if not flag_loglammps:
        skipline = tk.Entry(source_frame,width=5,justify="center")
        skipline.grid(row=next_row,column=1)
        skipline.insert(0,0)
    else:
        skipline=0
next_row+=1  
ttk.Separator(source_frame,orient='horizontal').grid(row=next_row,columnspan=2,sticky="ew") 

# Insert X,Y label text field
ttk.Separator(labels_frame,orient='horizontal').grid(columnspan=2,sticky="ew") 
xlabel = tk.Entry(labels_frame,width=50,justify="center")
xlabel.grid(row=next_row+1,column=0,columnspan=2)
xlabel.insert(0,"X-axis label")
ylabel = tk.Entry(labels_frame,width=50,justify="center")
ylabel.grid(row=next_row+2,column=0,columnspan=2)
ylabel.insert(0,"Y-axis label")

close = tk.Button(button_frame, text = "Close",width=10, command = close_windows)
close.grid(row=1)
plot = tk.Button(button_frame, text = "Plot",width=10, command = plot_button)
plot.grid(row=0)

# show GUI
root.mainloop()


# Cleaning tmp files
if flag_loglammps:
    for f in outlogfiles:
        os.remove(f)

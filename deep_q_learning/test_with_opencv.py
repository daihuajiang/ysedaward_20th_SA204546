from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import random
import os
from shutil import copyfile
import tkinter as tk
import tkinter.font as tkFont
import time
import cv2
import threading

from test_simulate_with_video import Simulation
from model import TestModel
from utils import import_test_configuration, set_sumo, set_test_path
from generator import TrafficGenerator


#計時器
class Clock():
    def __init__(self):
        self._actionlist_path = './action_list.txt'
        self.turngreen = [] #第n秒轉綠燈
        self.turnyellow = [] #第n秒轉黃燈
        self.turnred = [] #第n秒轉紅燈
        self.turnyellowcount = 0 #計算進入第幾次黃燈
        self.root = tk.Tk()
        self.font = tkFont.Font(family='microsoft yahei', size=40, weight='bold')
        self.bgcolor = '#DFFFDF'
        self.fgcolor = '#006000'
        self.timestr = tk.StringVar()
        self.running = False #計時器工作狀態
        self.inittime = time.time() #開始計時時間
        self.starttime = 0 #重製的開始計時時間
        self.maxtime = 0 #計時器維持時間
        self.elapsedtime = 0.0 #計時器統計的時間
        self.modelpredict = 0 #模型預測狀態
        self.predictstr = tk.StringVar()
        self.trafficflowstr = tk.StringVar()
        
        self.frame1 = tk.Frame(self.root)
        self.frame2 = tk.Frame(self.root)
        self.labelf1 = tk.Label(self.frame1, textvariable=self.timestr, bg=self.bgcolor, fg=self.fgcolor, font=self.font)
        self.labelf2 = tk.Label(self.frame2, textvariable=self.predictstr, bg=self.bgcolor, fg='#000000', font=self.font)
        #self.label = tk.Label(self.root, bg=self.bgcolor)
        #self.button1 = tk.Button(self.frame1, text='start', command=self.Start).pack(side=tk.LEFT)
        #self.button2 = tk.Button(self.frame1, text='stop', command=self.Stop).pack(side=tk.LEFT)
        #self.button3 = tk.Button(self.frame1, text='reset', command=self.Reset).pack(side=tk.LEFT)
        self.timer = None
        self.frame1.pack()
        self.frame2.pack()
        self.labelf1.grid(column=0, row=0)
        self.labelf2.grid(column=0, row=1)
        #self.label.pack()
        
        #self.timestr.set('燈號維持時間: 00:00')
        #self.predictstr.set('模型預測狀態: 綠燈')
        self.Check_action()
        self.Start()
        self.update()
        self.root.mainloop()
        
        
    #更新label內容
    def update(self):
        #變更開始時間及文字顏色
        if int(time.time() - self.inittime) in self.turngreen:
            self.modelpredict == 0
            self.predictstr.set('模型預測狀態: 綠燈')
            self.Reset()
        
        elif int(time.time() - self.inittime) in self.turnyellow:
            self.modelpredict == 1
            self.predictstr.set('模型預測狀態: 紅燈')
            self.Reset()
        
        #elif int(time.time() - self.inittime) in self.turnred:
            #self.turnyellowcount += 1
            #self.modelpredict == 1
            #self.predictstr.set('模型預測狀態: 紅燈')
            
            #self.Reset()
        
        #currenttime = int(time.time() - self.inittime)
        #if currenttime >= self.turnyellow[self.turnyellowcount] and currenttime < self.turnyellow[self.turnyellowcount]+3:
            #self.bgcolor = '#FFFFDF'
            #self.fgcolor = '#F75000'
            
        #elif currenttime >= self.turnyellow[self.turnyellowcount-1]+3 and currenttime < self.turnyellow[self.turnyellowcount-1]+90:
            #self.bgcolor = '#FFECEC'
            #self.fgcolor = '#FF0000'
        
        self.elapsedtime = time.time() - self.starttime
        m, s = divmod(int(self.elapsedtime), 60)
        min_sec_format = '燈號維持時間: {:02d}:{:02d}'.format(m, s)
        self.timestr.set(min_sec_format)
        self.timer = self.root.after(50, self.update)
        
        if int(time.time() - self.inittime) >= self.maxtime:
            self.root.destroy()

    #開始計時    
    def Start(self):
        if not self.running:
            self.starttime = time.time() - self.elapsedtime
            self.running = True
            self.update()
            pass

    #重製時間    
    def Reset(self):
        self.elapsedtime = 0.0
        m, s = divmod(int(self.elapsedtime), 60)
        min_sec_format = '燈號維持時間: {:02d}:{:02d}'.format(m, s)
        self.starttime = time.time()
        self.timestr.set(min_sec_format)
    
    #確認全部的action
    def Check_action(self):
        self.turngreen.append(0)
        
        with open(self._actionlist_path, 'r', newline='') as file:
            linecount = 0
            prevoiusaction = 0
            for line in file.readlines():
                if int(line) == 1:
                    prevoiusaction = 1
                    self.turnyellow.append((linecount*10))
                    self.turnred.append((linecount*10)+3)
                    
                if prevoiusaction == 1:
                    self.turngreen.append((linecount)*10+90)
                    prevoiusaction = 0
                    
                linecount += 1
            
            self.maxtime = (linecount + 9) *10
"""
def display_video():
    cap = cv2.VideoCapture('C:/Users/David01/traffic/objectdetect/runs/detect/object_tracking4/3.mp4')
    framecount = cap.get(5)
    
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        time.sleep(0.1/framecount)
        
    cap.release()
    cv2.destroyAllWindows()            
"""                   
if __name__ == "__main__":
    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )
    
    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )
    
    Simulation = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['red_duration'],
        config['num_states'],
        config['num_actions']
    )
    
    print('\n----- Test episode')
    simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    print('simulating finish')
    
    app = Clock()
    

    

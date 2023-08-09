import time
import tkinter
import tkinter.font as tkFont

root = tkinter.Tk()
f1 = tkFont.Font(family='microsoft yahei', size=40, weight='bold')
root.title("time")
root.geometry("400x400")
frame1 = tkinter.Frame(root)
frame1.pack()
timestr = tkinter.StringVar()
timestr.set('00:00')

running = False #計時器工作狀態
starttime = 0 #開始計時時間
elapsedtime = 0.0 #計時器統計的時間
timer = None

#更新label內容
def update():
    global elapsedtime, timestr, timer
    elapsedtime = time.time() - starttime
    m, s = divmod(int(elapsedtime), 60)
    min_sec_format = '{:02d}:{:02d}'.format(m, s)
    timestr.set(min_sec_format)
    timer = root.after(50, update)

#開始計時    
def Start():
    global running, starttime
    if not running:
        starttime = time.time() - elapsedtime
        running = True
        update()
        pass

#停止計時
def Stop():
    global running, timer
    if running:
        root.after_cancel(timer)
        elapsedtime = time.time() - starttime
        m, s = divmod(int(elapsedtime), 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        timestr.set(min_sec_format)
        running = False
        pass

#重製時間    
def Reset():
    global elapsedtime, timestr, starttime
    elapsedtime = 0.0
    m, s = divmod(int(elapsedtime), 60)
    min_sec_format = '{:02d}:{:02d}'.format(m, s)
    starttime = time.time()
    timestr.set(min_sec_format)

tkinter.Label(frame1, textvariable=timestr, bg='#CEFFCE', fg='#009100', font=f1).pack()
#tkinter.Button(frame1, text='start', command=Start).pack(side=tkinter.LEFT)
Start()
tkinter.Button(frame1, text='stop', command=Stop).pack(side=tkinter.LEFT)
tkinter.Button(frame1, text='reset', command=Reset).pack(side=tkinter.LEFT)

root.mainloop()
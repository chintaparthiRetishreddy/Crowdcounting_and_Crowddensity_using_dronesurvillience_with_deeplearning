import os
import cv2
import ctypes
import logging

from time import sleep

from tkinter import *
import tkinter.messagebox as tkMessageBox
import tkinter.filedialog as filedialog

from PIL import ImageTk, Image


from apscheduler.schedulers.background import BackgroundScheduler

import logging
logging.basicConfig()
logging.getLogger('apscheduler').setLevel(logging.ERROR)

from nwpu.nwpu_count import nwpu_count
import torch
from nwpu.config import cfg
from nwpu.models.CC import CrowdCounter

from classifier.classifier import predict_density



model_path = './nwpu/exp/MCNN-all_ep_907_mae_218.5_mse_700.6_nae_2.005.pth'
model = CrowdCounter(cfg.GPU_ID, 'MCNN')

model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

# ========================================================================== WINDOW ===================================================================
window = Tk()
window.title("Crowd Counting")

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
w = w-100
h = h-150
window.geometry(f"{w}x{h}+{25}+{25}")
#window.resizable(0,0)

loading_frame = Frame(window, bg="#FFFFFF", width=w, height=h)
select_frame = Frame(window, bg="#FFFFFF", width=w, height=h)
select_image_frame = Frame(window, bg="#FFFFFF", width=w, height=h)
select_video_frame = Frame(window, bg="#FFFFFF", width=w, height=h)
select_stream_frame = Frame(window, bg="#FFFFFF", width=w, height=h)
select_image_result = Frame(window, bg="#FFFFFF", width=w, height=h)
capture_image_result = Frame(window, bg="#FFFFFF", width=w, height=h)
select_video_result = Frame(window, bg="#FFFFFF", width=w, height=h)
capture_video_result = Frame(window, bg="#FFFFFF", width=w, height=h)
capture_stream_result = Frame(window, bg="#FFFFFF", width=w, height=h)


# ========================================================================== STREAM ===================================================================
def CaptureStreamResult(stream):
    global result, count_result
    result, count_result = "", 0
    capture_stream_result.place(x=0, y=0)
    
    image_frame = Frame(window, width=1080, height=720, borderwidth=4, bg='black')
    image_frame.place(x=370, y=80)
    densiity_label = Label(capture_stream_result, text=f"Density : {result}", font=("Comic Sans MS", 25, "bold"), bg="#FFFFFF")
    densiity_label.place(x=550, y=15)
    count_label = Label(capture_stream_result, text=f"Count : {int(count_result)}", font=("Comic Sans MS", 25, "bold"), bg="#FFFFFF")
    count_label.place(x=1050, y=15)

    cap = cv2.VideoCapture(stream)
    def show_frame():
        try:
            ret, frame = cap.read()
            if ret:
                global frame_copy
                frame_copy = frame.copy()
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                img = img.resize((1080, 720))
                imgtk = ImageTk.PhotoImage(image=img)

                display1.imgtk = imgtk
                display1.configure(image=imgtk)
                window.after(10, show_frame)
        except Exception as e:
            print(e)

    display1 = Label(image_frame)
    display1.grid(row=1, column=0)

    show_frame()

    if result == "":
        result = predict_density(frame_copy)
        densiity_label.config(text=f"Density : {result}")
        print(result)



    def start_detection():     
        count_result = nwpu_count(frame_copy, model)
        print(count_result)   
        count_label.config(text=f"Count : {int(count_result)}")

    scheduler = BackgroundScheduler()
    scheduler.add_job(start_detection, 'interval', seconds=3)
    scheduler.start()

    def back():
        result = ""
        count_result = 0
        cap.release()
        image_frame.place_forget()
        capture_stream_result.place_forget()
        scheduler.shutdown()
        densiity_label.place_forget()
        count_label.place_forget()
        SelectStreamScreen()

    Button(capture_stream_result, text = "BACK", font = ("Agency FB", 14, "bold"), relief = FLAT, bd = 0, width=20, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:back()).place(x=25, y = 25)


def SelectStreamScreen():
    select_stream_frame.place(x=0, y=0)

    Label(select_stream_frame, text="Crowd Counting", font=("Comic Sans MS", 75, "bold"), bg="#FFFFFF").place(x=150, y=200)

    def capture_stream_ip():
        stream_ip = stream_ip_entry.get()
        if len(stream_ip) == 0:
                result = tkMessageBox.showinfo("Crowd Counting", "Enter Stream IP!", icon= "warning")
        else:
            CaptureStreamResult(stream_ip)

    stream_ip_label = Label(select_stream_frame, text="Stream IP", font = ("Agency FB",20,"bold"),relief = FLAT, fg="black", bg="#FFFFFF")
    stream_ip_label.place(x = 70, y = 225)
    stream_ip_entry = Entry(select_stream_frame, font = ("Agency FB",20,"normal"), highlightthickness=2, bg="#FFFFFF", fg="black", highlightcolor="#006EFF", selectbackground="black", width=30)
    stream_ip_entry.place(x=200, y=225)
	    
    Button(select_stream_frame, text = "Start Stream", font = ("Agency FB", 28, "bold"), relief = FLAT, bd = 0, width=20, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:capture_stream_ip()).place(x=525, y = 10)

    def back():
        select_stream_frame.place_forget()
        SelectScreen()

    Button(select_stream_frame, text = "BACK", font = ("Agency FB", 14, "bold"), relief = FLAT, bd = 0, width=20, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:back()).place(x=25, y = 10)


# ========================================================================== VIDEO ===================================================================


def SelectVideoResult():
    select_video_result.place(x=0, y=0)
    global result, count_result
    result, count_result = "", 0

    selected_video = filedialog.askopenfilename(title="Select file", filetypes=( ("Video Files",(".mp4",".avi")),("All Files", "*.*")))
    if selected_video == "":
        select_video_result.place_forget()
        SelectVideoScreen()

    image_frame = Frame(window, width=1080, height=720, borderwidth=4, bg='black')
    image_frame.place(x=370, y=80)
    densiity_label = Label(select_video_result, text=f"Density : {result}", font=("Comic Sans MS", 25, "bold"), bg="#FFFFFF")
    densiity_label.place(x=550, y=15)
    count_label = Label(select_video_result, text=f"Count : {int(count_result)}", font=("Comic Sans MS", 25, "bold"), bg="#FFFFFF")
    count_label.place(x=1050, y=15)

    cap = cv2.VideoCapture(selected_video)

    def show_frame():
        try:
            ret, frame = cap.read()
            if ret:
                global frame_copy
                frame_copy = frame.copy()
                frame = cv2.flip(frame, 1)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                img = img.resize((1080, 720))
                imgtk = ImageTk.PhotoImage(image=img)

                display1.imgtk = imgtk
                display1.configure(image=imgtk)
                window.after(10, show_frame)
        except Exception as e:
            print(e)

    display1 = Label(image_frame)
    display1.grid(row=1, column=0)

    show_frame()
    
    if result == "":
        result = predict_density(frame_copy)
        densiity_label.config(text=f"Density : {result}")
        print(result)



    def start_detection():     
        count_result = nwpu_count(frame_copy, model)
        print(count_result)  
        count_label.config(text=f"Count : {int(count_result)}")


    scheduler = BackgroundScheduler()
    scheduler.add_job(start_detection, 'interval', seconds=3)
    scheduler.start()

    def back():
        result = ""
        count_result = 0
        cap.release()
        scheduler.shutdown()
        image_frame.place_forget()
        select_video_result.place_forget()
        densiity_label.place_forget()
        count_label.place_forget()
        SelectVideoScreen()

    Button(select_video_result, text = "BACK", font = ("Agency FB", 14, "bold"), relief = FLAT, bd = 0, width=20, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:back()).place(x=25, y = 25)


def CaptureVideoResult():
    global result, count_result
    result, count_result = "", 0
    capture_video_result.place(x=0, y=0)
    
    image_frame = Frame(window, width=1080, height=720, borderwidth=4, bg='black')
    image_frame.place(x=370, y=80)
    densiity_label = Label(capture_video_result, text=f"Density : {result}", font=("Comic Sans MS", 25, "bold"), bg="#FFFFFF")
    densiity_label.place(x=550, y=15)
    count_label = Label(capture_video_result, text=f"Count : {int(count_result)}", font=("Comic Sans MS", 25, "bold"), bg="#FFFFFF")
    count_label.place(x=1050, y=15)

    cap = cv2.VideoCapture(0)
    
    def show_frame():
        try:
            ret, frame = cap.read()
            if ret:
                global frame_copy
                frame_copy = frame.copy()
                frame = cv2.flip(frame, 1)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                img = img.resize((1080, 720))
                imgtk = ImageTk.PhotoImage(image=img)

                display1.imgtk = imgtk
                display1.configure(image=imgtk)
                window.after(10, show_frame)
        except Exception as e:
            print(e)

    display1 = Label(image_frame)
    display1.grid(row=1, column=0)

    show_frame()

    if result == "":
        result = predict_density(frame_copy)
        densiity_label.config(text=f"Density : {result}")
        print(result)

        
    def start_detection():        
        count_result = nwpu_count(frame_copy, model)
        print(count_result)
        count_label.config(text=f"Count : {int(count_result)}")


    scheduler = BackgroundScheduler()
    scheduler.add_job(start_detection, 'interval', seconds=3)
    scheduler.start()

    def back():
        result, count_result = "", 0
        cap.release()
        scheduler.shutdown()
        image_frame.place_forget()
        capture_video_result.place_forget()
        densiity_label.place_forget()
        count_label.place_forget()
        SelectVideoScreen()

    Button(capture_video_result, text = "BACK", font = ("Agency FB", 14, "bold"), relief = FLAT, bd = 0, width=20, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:back()).place(x=25, y = 25)


def SelectVideoScreen():
    select_video_frame.place(x=0, y=0)

    Label(select_video_frame, text="Crowd Counting", font=("Comic Sans MS", 75, "bold"), bg="#FFFFFF").place(x=100, y=100)

    Button(select_video_frame, text = "Browse Video", font = ("Agency FB", 28, "bold"), relief = FLAT, bd = 0, width=15, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:SelectVideoResult()).place(x=200, y = 400)

    Button(select_video_frame, text = "Start Video", font = ("Agency FB", 28, "bold"), relief = FLAT, bd = 0, width=15, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:CaptureVideoResult()).place(x=600, y = 400)

    def back():
        select_video_frame.place_forget()
        SelectScreen() 
        #chal gya count nai kr rha density shw kr rha h ... koi bi videp  chal gyawo bhi bs to fir dedeo
    Button(select_video_frame, text = "BACK", font = ("Agency FB", 14, "bold"), relief = FLAT, bd = 0, width=20, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:back()).place(x=25, y = 25)

# ========================================================================== IMAGE ===================================================================


def SelectImageResult():
    select_image_result.place(x=0, y=0)

    selected_image = filedialog.askopenfilename(title="Select file", filetypes=( ("Image Files",(".jpg",".png",".jpeg")),("All Files", "*.*")))
    if selected_image == "":
        select_image_result.place_forget()
        SelectImageScreen()

    image = cv2.imread(selected_image)
    
    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img = img.resize((1080, 720))
    imgtk = ImageTk.PhotoImage(image=img)

    image_frame = Frame(window, width=1080, height=720, borderwidth=4, bg='black')
    image_frame.place(x=370, y=80)
    
    display1 = Label(image_frame)
    display1.grid(row=1, column=0)
    display1.imgtk = imgtk
    display1.configure(image=imgtk)


    result = predict_density(image)   # PREDICT DENSITY
    density_label = Label(select_image_result, text=f"Density : {result}", font=("Comic Sans MS", 25, "bold"), bg="#FFFFFF")
    density_label.place(x=550, y=15)

    count_result = nwpu_count(image, model)

        
    count_label = Label(select_image_result, text=f"Count : {int(count_result)}", font=("Comic Sans MS", 25, "bold"), bg="#FFFFFF")
    count_label.place(x=1050, y=15)

    def back():

        image_frame.place_forget()
        select_image_result.place_forget()
        density_label.place_forget()
        count_label.place_forget()
        SelectImageScreen()

    Button(select_image_result, text = "BACK", font = ("Agency FB", 14, "bold"), relief = FLAT, bd = 0, width=20, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:back()).place(x=25, y = 25)


def CaptureImageResult():
    capture_image_result.place(x=0, y=0)
    
    image_frame = Frame(window, width=1080, height=720, borderwidth=4, bg='black')
    image_frame.place(x=370, y=80)
    
    cap = cv2.VideoCapture(0)
    def show_frame():
        try:
            ret, frame = cap.read()
            if ret:
                global frame_copy
                frame_copy = frame.copy()
                frame = cv2.flip(frame, 1)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                img = img.resize((1080, 720))
                imgtk = ImageTk.PhotoImage(image=img)

                display1.imgtk = imgtk
                display1.configure(image=imgtk)
                window.after(10, show_frame)
        except Exception as e:
            print(e)

    display1 = Label(image_frame)
    display1.grid(row=1, column=0)


    def capture_image():
        global density_label, count_label
        result = predict_density(frame_copy)   # PREDICT DENSITY
        density_label = Label(capture_image_result, text=f"Density : {result}", font=("Comic Sans MS", 25, "bold"), bg="#FFFFFF")
        density_label.place(x=550, y=15)

        count_result = nwpu_count(frame_copy, model)
        
        count_label = Label(capture_image_result, text=f"Count : {count_result}", font=("Comic Sans MS", 25, "bold"), bg="#FFFFFF")
        count_label.place(x=1050, y=15)

        capture_button.place_forget()
        cap.release()

    capture_button = Button(capture_image_result, text = "CAPTURE", font = ("Agency FB", 14, "bold"), relief = FLAT, bd = 0, width=20, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:capture_image())
    capture_button.place(x=280, y = 25)

    show_frame()

    def back():
        cap.release()
        image_frame.place_forget()
        capture_image_result.place_forget()
        density_label.place_forget()
        count_label.place_forget()
        SelectImageScreen()

    Button(capture_image_result, text = "BACK", font = ("Agency FB", 14, "bold"), relief = FLAT, bd = 0, width=20, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:back()).place(x=25, y = 25)


def SelectImageScreen():
    select_image_frame.place(x=0, y=0)

    Label(select_image_frame, text="Crowd Counting", font=("Comic Sans MS", 75, "bold"), bg="#FFFFFF").place(x=100, y=200)

    Button(select_image_frame, text = "Browse Image", font = ("Agency FB", 28, "bold"), relief = FLAT,  width=15, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:SelectImageResult()).place(x=150, y = 100)

    Button(select_image_frame, text = "Click Image", font = ("Agency FB", 28, "bold"), relief = FLAT, width=15, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:CaptureImageResult()).place(x=650, y = 100)

    def back():
        select_image_frame.place_forget()
        SelectScreen()

    Button(select_image_frame, text = "BACK", font = ("Agency FB", 14, "bold"), relief = FLAT, bd = 0, width=20, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:back()).place(x=25, y = 25)

# ========================================================================== SELECT ===================================================================

def SelectScreen():
	select_frame.place(x=0, y=0)

	Label(select_frame, text="Crowd Counting", font=("Comic Sans MS", 75, "bold"), bg="#FFFFFF").place(x=150, y=200)

	Button(select_frame, text = "From Image", font = ("Agency FB", 28, "bold"), relief = FLAT, bd = 0, width=15, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:SelectImageScreen()).place(x=70, y = 500)

	Button(select_frame, text = "From Video", font = ("Agency FB", 28, "bold"), relief = FLAT, bd = 0, width=15, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:SelectVideoScreen()).place(x=470, y = 500)

	Button(select_frame, text = "Live Stream", font = ("Agency FB", 28, "bold"), relief = FLAT, bd = 0, width=15, fg="#FFFFFF", bg='black', activebackground = "#E3E3E3",activeforeground = "#006EFF", command=lambda:SelectStreamScreen()).place(x=870, y = 500)


# ========================================================================== LOADING ===================================================================

def LoadingScreen():
	loading_frame.place(x=0, y=0)

	Label(loading_frame, text="Crowd Counting", font=("Comic Sans MS", 75, "bold"), bg="#FFFFFF").place(x=100, y=200)
	
	for i in range(28):
		Label(loading_frame, bg="#574D72",width=2,height=1).place(x=(i+4)*50,y=690) 

	def play_animation(): 
		for j in range(28):
			Label(loading_frame, bg= 'black',width=2,height=1).place(x=(j+4)*50,y=690) 
			sleep(0.07)
			loading_frame.update_idletasks()
		else:
			loading_frame.place_forget()
			SelectScreen()

	loading_frame.update()
	play_animation()
	


# ========================================================================== INTITIALIZE ===================================================================

LoadingScreen()
# SelectScreen()


window.configure(background='#FFFFFF')
window.mainloop()

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import messagebox
from camera_check import getCameraNum

root = Tk()
root.filename = None
root.textfiledir = None
root.videofiledir = None

# root.filename - path to video file to process / rbutton_file - True if video file and False if stream
# text_out_var - 1 if checked to output result json and 0 if no / vid_out_var - whether to output video.mp4
# root.textfiledir and root.videofiledir - paths to save files
# var_tracking_cb - vehicle tracker checkbox / entry_tracking_age - tracker max age
# var_person_det_cb - face detection checkbox / entry_face_det_prob - face detection probability >=
# var_color_cb - color recognition checkbox / var_color_method - string of method to use for color recognition
# entry_vehicle_det_prob - vehicle minimum probability >=

root.title("Vehicle and Passenger Recognition System")
# root.iconbitmap("c:/")

Label(root, text="Input and Output Settings", relief=SUNKEN, pady=3).grid(row=0, column=0, columnspan=5, sticky=W+E)

# ------------------- FILE OR CAMERA SELECTION
rbutton_file = BooleanVar()

Radiobutton(root, text="Video file", variable=rbutton_file, value=True).grid(row=1,column=0, sticky=W)
Radiobutton(root, text="Camera", variable=rbutton_file, value=False).grid(row=2, column=0, sticky=W)

vfile_label = Label(root, text=". . .")
vfile_label.grid(row=1, column=2)


def open_vfile():
    root.filename = filedialog.askopenfilename(initialdir="./", title="Select video file", filetypes=(
        ("mp4", "*.mp4"), ("avi", "*.avi"), ("mov", "*.mov"), ("all files", "*.*")))
    vfile_label.config(text=root.filename.split("/")[-1])


Button(root, text="Open file", command=open_vfile, padx=34).grid(row=1, column=1)

camera_dropdown = None


def camera_refresh():
    global camera_dropdown
    try:
        camera_dropdown.grid_forget()
    except:
        pass
    num = getCameraNum()
    if num > 0:
        camera_selected = IntVar()
        camera_selected.set(0)
        camera_dropdown = OptionMenu(root, camera_selected, *[i for i in range(num)])
        camera_dropdown.config(width=14)
        camera_dropdown.grid(row=2, column=1)
    else:
        camera_dropdown = Button(root, text="Refresh - no cameras detected", command=camera_refresh)
        camera_dropdown.grid(row=2, column=1)


# camera_dropdown = Button(root, text="Refresh - no cameras detected", command=camera_refresh) # Test if function works if no cameras detected
# camera_dropdown.grid(row=2, column=1)

camera_refresh()

# ---- OUTPUT SELECTION

outtext_file_label = Label(root, text=". . .")
outtext_file_label.grid(row=3, column=2)
outvid_file_label = Label(root, text=". . .")
outvid_file_label.grid(row=4, column=2)
text_out_var = IntVar()
vid_out_var = IntVar()
Checkbutton(root, text="Text output file", variable=text_out_var).grid(row=3, column=0, sticky=W)
Checkbutton(root, text="Video output file", variable=vid_out_var).grid(row=4, column=0, sticky=W)


def open_tfile_saveloc():
    root.textfiledir = filedialog.asksaveasfilename(initialdir="./", title="Select location to save at")
    outtext_file_label.config(text=root.textfiledir.split("/")[-1])


def open_vfile_saveloc():
    root.videofiledir = filedialog.asksaveasfilename(initialdir="./", title="Select location to save at")
    outvid_file_label.config(text=root.videofiledir.split("/")[-1])


Button(root, text="Select file", command=open_tfile_saveloc, padx=34).grid(row=3, column=1)
Button(root, text="Select file", command=open_vfile_saveloc, padx=34).grid(row=4, column=1)

# ------ Detection options
Label(root, text="Detection Settings", relief=SUNKEN, pady=3).grid(row=5, column=0, columnspan=5, sticky=W+E)

var_tracking_cb = IntVar()
var_person_det_cb = IntVar()
var_color_cb = IntVar()
Checkbutton(root, text="Vehicle tracking", variable=var_tracking_cb).grid(row=6, column=0, sticky=W)
Checkbutton(root, text="Passenger detection", variable=var_person_det_cb).grid(row=7, column=0, sticky=W)
Checkbutton(root, text="Color detection", variable=var_color_cb).grid(row=8, column=0, sticky=W)


entry_tracking_age = Entry(root, justify=CENTER)
entry_tracking_age.insert(0, 30)
entry_tracking_age.grid(row=6, column=1)

entry_face_det_prob = Entry(root, justify=CENTER)
entry_face_det_prob.insert(0, 0.5)
entry_face_det_prob.grid(row=7, column=1)

var_color_method = StringVar()
var_color_method.set("bincount")
om_color_method = OptionMenu(root, var_color_method, "bincount")
om_color_method.config(width=14)
om_color_method.grid(row=8, column=1)

entry_vehicle_det_prob = Entry(root, justify=CENTER)
entry_vehicle_det_prob.insert(0, 0.7)
entry_vehicle_det_prob.grid(row=9, column=1)


Label(root, text=" misses before track deletion").grid(row=6, column=2, sticky=W)
Label(root, text=" minimum probability (0.0-1.0)").grid(row=7, column=2, sticky=W)
Label(root, text="Vehicle detection").grid(row=9, column=0, sticky=W)
Label(root, text=" minimum probability (0.0-1.0)").grid(row=9, column=2, sticky=W)
# ------

# my_img = ImageTk.PhotoImage(Image.open("A:\\Desktop_HDD\\resident-sleeper.png"))
# my_label = Label(image=my_img)
# my_label.pack()
# my_label.grid_forget()

# canvas1 = Canvas(root, width = 400, height = 300, relief = 'raised')
# canvas1.pack
#
# label1 = Label(root, text='Admin Panel')
# label1.config(font=('helvetica', 14))
# canvas1.create_window(200,25,window=label1)

# e = Entry(root, width = 30)#borderwidth = 5) #width = 50, bg = "blue", fg ="white"
# e.insert(0, "Enter your name")
# e.pack()
# e.delet(0, END)

# def myClick():
#     myLabel = Label(root, text=e.get())
#     myLabel.pack()
#
# myButton = Button(root, text="Enter your name", command = myClick) #, fg="blue", bg="yellow")#state = DISABLED, padx=50, pady=50, fg="blue", bg="red")
# myButton.pack()

# # Label Widget
# myLabel1 = Label(root, text="Hello World!")
# myLabel2 = Label(root, text="My name is John Doe")
# myLabel3 = Label(root, text="----Hi-----")
#
# # Shove it onto the screen
# myLabel1.grid(row = 0, column = 0)
# myLabel2.grid(row = 1, column = 5)
# myLabel3.grid(row = 1, column = 1)


def exit_application():
    exit_message = messagebox.askquestion("Exit Application", "Are you sure you want to exit?", icon="warning")
    if exit_message == "yes":
        root.quit()
    else:
        pass


button_quit = Button(root, text="Exit Program", command=exit_application, borderwidth=5)
button_quit.grid(row=10, column=0, columnspan=5, sticky=W+E)

root.mainloop()

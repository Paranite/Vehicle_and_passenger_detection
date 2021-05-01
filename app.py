from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import messagebox
from camera_check import getCameraNum
from object_tracker import object_tracking

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

Label(root, text="Input and Output Settings", relief=SUNKEN, pady=3).grid(row=0, column=0, columnspan=5, sticky=W + E)

# ------------------- FILE OR CAMERA SELECTION
rbutton_file = BooleanVar()

Radiobutton(root, text="Video file", variable=rbutton_file, value=True).grid(row=1, column=0, sticky=W)
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
text_out_var = BooleanVar()
vid_out_var = BooleanVar()
Checkbutton(root, text="Text output file", variable=text_out_var).grid(row=3, column=0, sticky=W)
Checkbutton(root, text="Video output file", variable=vid_out_var).grid(row=4, column=0, sticky=W)


def open_tfile_saveloc():
    root.textfiledir = filedialog.asksaveasfilename(initialdir="./", title="Select location to save at",
                                                    initialfile="output", defaultextension=".csv",
                                                    filetypes=(("csv", "*.csv"), ("all files", "*")))
    outtext_file_label.config(text=root.textfiledir.split("/")[-1])


def open_vfile_saveloc():
    root.videofiledir = filedialog.asksaveasfilename(initialdir="./", title="Select location to save at",
                                                     initialfile="output", defaultextension=".mp4",
                                                     filetypes=(("mp4", "*.mp4"), ("all files", "*")))
    outvid_file_label.config(text=root.videofiledir.split("/")[-1])


Button(root, text="Select file", command=open_tfile_saveloc, padx=34).grid(row=3, column=1)
Button(root, text="Select file", command=open_vfile_saveloc, padx=34).grid(row=4, column=1)

# ------ Detection options
Label(root, text="Detection Settings", relief=SUNKEN, pady=3).grid(row=5, column=0, columnspan=5, sticky=W + E)

var_tracking_cb = BooleanVar()
var_person_det_cb = BooleanVar()
var_color_cb = BooleanVar()
Checkbutton(root, text="Vehicle tracking", variable=var_tracking_cb).grid(row=6, column=0, sticky=W)
Checkbutton(root, text="Passenger detection", variable=var_person_det_cb).grid(row=7, column=0, sticky=W)
Checkbutton(root, text="Color detection", variable=var_color_cb).grid(row=8, column=0, sticky=W)
print(var_color_cb.get())
print(var_color_cb.get())

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

def start_video_processing():
    if rbutton_file.get() and not root.filename:
        messagebox.showwarning("Missing param", "No video selected", icon="warning")
        return
    if vid_out_var.get() and not root.videofiledir:
        messagebox.showwarning("Missing param", "No video output selected", icon="warning")
        return
    if text_out_var.get() and not root.textfiledir:
        messagebox.showwarning("Missing param", "No text output selected", icon="warning")
        return

    print("Video path: {} | Vid/camera: {} | Video output: {} {} | Text output: {} {}|"
          " Tracking: {} | Tracking age: {} | Face det: {} | Face threshold: {} |"
          " Color: {} {} | Vehicle threshold: {}".format(root.filename, rbutton_file.get(), root.videofiledir,
                                                         vid_out_var.get(), root.textfiledir, text_out_var.get(),
                                                         var_tracking_cb.get(), entry_tracking_age.get(),
                                                         var_person_det_cb.get(), entry_face_det_prob.get(),
                                                         var_color_method.get(), var_color_cb.get(),
                                                         entry_vehicle_det_prob.get()))

    object_tracking(None, video_path=root.filename if rbutton_file.get() else None,
                    vid_output_path=root.videofiledir if vid_out_var.get() else None,
                    text_output_path=root.textfiledir if text_out_var.get() else None, tracking=var_tracking_cb.get(),
                    tracker_max_age=int(entry_tracking_age.get()), face_det=var_person_det_cb.get(),
                    face_score_threshold=float(entry_face_det_prob.get()),
                    color=var_color_method.get() if var_color_cb.get() else None,
                    score_threshold=float(entry_vehicle_det_prob.get()),
                    track_only=["car", "bus", "motorcycle", "truck", "bicycle"], show=True)


Button(root, text="Detect", borderwidth=3, command=start_video_processing).grid(row=10, column=0, columnspan=5, sticky=W+E)


def exit_application():
    exit_message = messagebox.askquestion("Exit Application", "Are you sure you want to exit?", icon="warning")
    if exit_message == "yes":
        root.quit()
    else:
        pass


button_quit = Button(root, text="Exit Program", command=exit_application, borderwidth=5)
button_quit.grid(row=11, column=0, columnspan=5, sticky=W+E)

root.mainloop()

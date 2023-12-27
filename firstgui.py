

from secondgui  import *
from Object_color_detection import *
from Green_screen import *
from Invisibility_cloak import *
from game import *
from pathlib import Path
from tkinter import  Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog,font
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)
#function upload_image() uploads an image from the user 
# and stores it in the global variable image_filtre

def upload_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

    if file_path:
        global image_filtre
        image_filtre = file_path
        image_8 = canvas.create_image(
        138.0,
        218.0,
         image=image_image_8
        )
    else:
        image_6 = canvas.create_image(
        138.0,
        218.0,
         image=image_image_6
        )
# function return_image() returns image_filtre

def return_image():
    return image_filtre

#new window named 'window' is created and given parametres 

window = Tk()

window.geometry("700x700")
window.configure(bg = "#C7C7C7")


canvas = Canvas(
    window,
    bg = "#C7C7C7",
    height = 700,
    width = 700,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    342.0,
    72.0,
    700.0,
    431.0,
    fill="#D9D9D9",
    outline="")

canvas.create_rectangle(
    0.0,
    0.0,
    816.0,
    73.0,
    fill="#312D62",
    outline="")
jolly_lodger_font = font.Font(family="Jolly Lodger", size=40)
canvas.create_text(
    23.0,
    12.0,
    anchor="nw",
    text="Vision Project dashboard",
    fill="#FFFFFF",
    font=jolly_lodger_font
)
jolly_lodger_font = font.Font(family="Jolly Lodger", size=70)
canvas.create_text(
    326.0,
    175.0,
    anchor="nw",
    text="Or",
    fill="#F88D0E",
    font=jolly_lodger_font
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    350.0,
    520.0,
    image=image_image_1
)
jolly_lodger_font = font.Font(family="Jolly Lodger", size=70)
canvas.create_text(
    220.0,
    464.0,
    anchor="nw",
    text="Try this out! ",
    fill="#F88D0E",
    font=jolly_lodger_font
)

#Button calls the upload_image() function to store the choosen image 

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=upload_image,
    relief="flat"
)
button_1.place(
    x=44.0,
    y=129.0,
    width=191.69668579101562,
    height=46.37255096435547
)

#Button calls the second GUI secondgui.py and gives the global variable 'image_filtre'
#as a parameter
 
button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda:newwind(image_filtre),
    relief="flat"
)
button_2.place(
    x=43.0,
    y=260.0,
    width=191.69668579101562,
    height=46.37255096435547
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    259.0,
    228.0,
    image=image_image_2
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: launch_object_color_detection(),
    relief="flat"
)
button_3.place(
    x=9.0,
    y=500.0,
    width=210.0,
    height=50.0
)

button_image_80= PhotoImage(
    file=relative_to_assets("button_75.png"))
button_80 = Button(
    image=button_image_80,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: Launch(),
    relief="flat"
)
button_80.place(
    x=9.0,
    y=587.0,
    width=210.0,
    height=50.0
)


image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    537.0,
    296.0,
    image=image_image_3
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    593.0,
    571.0,
    image=image_image_4
)
#Button calls the Launch_Invisibility_cloak() function 

button_image_4 = PhotoImage(
    file=relative_to_assets("button_101.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda:Launch_Invisibility_cloak(),
    relief="flat"
)
button_4.place(
    x=237.0,
    y=587.0,
    width=210.0,
    height=50.0
)

#Button calls the Launch_Green_Screen() function

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=lambda:  Launch_Green_screen(),
    relief="flat"
)
button_5.place(
    x=464.0,
    y=587.0,
    width=210.0,
    height=50.0
)

#Button calls the car game function

button_image_7 = PhotoImage(
    file=relative_to_assets("button_100.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=lambda:game(),
    relief="flat"
)
button_7.place(
    x=497.0,
    y=85.0,
    width=80.69668579101562,
    height=46.37255096435547
)


image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    71.0,
    382.0,
    image=image_image_5
)
image_image_6 = PhotoImage(
file=relative_to_assets("image_6.png"))
image_image_8 = PhotoImage(
file=relative_to_assets("button_8.png"))

window.resizable(False, False)
window.mainloop()
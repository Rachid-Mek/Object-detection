

from secondgui  import *
from Object_color_detection import *
from Green_screen import *
from Invisibility_cloak import *
from game import *
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog,font


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:/Users/User/Desktop/PROJETVISIONS/assets/frame0")




    

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)
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


def return_image():
    return image_filtre



window = Tk()

window.geometry("816x879")
window.configure(bg = "#E0E0E7")


canvas = Canvas(
    window,
    bg = "#E0E0E7",
    height = 879,#879
    width = 816,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    73.0,
    382.0,
    492.0,
    fill="#E0E0E7",
    outline="")

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    408.0,
    669.0,
    image=image_image_1
)

canvas.create_rectangle(
    382.0,
    73.0,
    816.0,
    545.0,
    fill="#FFFFFF",
    outline="")
jolly_lodger_font = font.Font(family="Jolly Lodger", size=70)
canvas.create_text(
    200,#294.0,
    590.0,
    anchor="nw",
    text="Try this out ! ",
    fill="#F88D0E",
    font=jolly_lodger_font
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: launch_object_color_detection(),
    relief="flat"
)
button_1.place(
    x=46.0,
    y=726.0,
    width=225.0,
    height=51.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: Launch_Invisibility_cloak(),
    relief="flat"
)
button_2.place(
    x=300.0,
    y=726.0,
    width=219.0,
    height=51.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: Launch_Green_screen(),
    relief="flat"
)
button_3.place(
    x=570.0,
    y=728.0,
    width=225.0,
    height=51.0
)

canvas.create_rectangle(
    0.0,
    0.0,
    816.0,
    73.0,
    fill="#312D62",
    outline="")
jolly_lodger_font = font.Font(family="Jolly Lodger", size=40)
canvas.create_text(
    44.0,
    12.0,
    anchor="nw",
    text="Project Dashboard",
    fill="#FFFFFF",
    font=jolly_lodger_font
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=upload_image,
    relief="flat"
)
button_4.place(
    x=46.0,
    y=135.0,
    width=191.0,
    height=36.0
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=lambda:newwind(image_filtre),
    relief="flat"
)
button_5.place(
    x=46.0,
    y=265.0,
    width=191.0,
    height=36.0
)


button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: game(),
    relief="flat"
)
button_7.place(
    x=523.0,
    y=107.0,
    width=159.0,
    height=57.0
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    603.0,
    384.0,
    image=image_image_2
)
jolly_lodger_font = font.Font(family="Jolly Lodger", size=90)
canvas.create_text(
    358.0,
    122.0,
    anchor="nw",
    text="or",
    fill="#F88D0E",
    font=jolly_lodger_font
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    275.0,
    348.0,
    image=image_image_3
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    279.0,
    534.0,
    image=image_image_4
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    662.0,
    859.0,
    image=image_image_5
)

image_image_6 = PhotoImage(
file=relative_to_assets("image_6.png"))
image_image_8 = PhotoImage(
file=relative_to_assets("button_8.png"))

window.resizable(False, False)
window.mainloop()

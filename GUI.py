from tkinter import *
from tkinter import filedialog


# Function for opening the
# file explorer window
# Create the root window
def open_window():
    window = Tk()
    # Set window title
    window.title('Music Genre Detector')
    # Set window size
    window.geometry("500x500")
    # Set window background color
    window.config(background="WHITE")
    # Create a File Explorer label
    label_file_explorer = Label(window,
                                text="Please select the music file to analyze",
                                width=100, height=4,
                                fg="blue")
    button_explore = Button(window,
                            text="Browse Files",
                            command=browseFiles)
    button_exit = Button(window,
                         text="Exit",
                         command=exit)
    label_file_explorer.grid(column=1, row=1)
    button_explore.grid(column=1, row=2)
    button_exit.grid(column=1, row=3)
    return window, label_file_explorer

def browseFiles():
    window, label_file_explorer = open_window()
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("Text files",
                                                      "*.txt*"),
                                                     ("all files",
                                                      "*.*")))

    label_file_explorer.configure(text="File Opened: " + filename)
    close_window(window)
    return filename

def open_results_window(results):
    res_window = Tk()
    # Set window title
    res_window.title('Music Genre Detector')
    # Set window size
    res_window.geometry("650x100")
    # Set window background color
    res_window.config(background="WHITE")
    # Create a File Explorer label
    label_file_explorer = Label(res_window,
                                text=f"The selected music is of the {results} genre",
                                width=35, height=1, font=("Arial", 25),
                                fg="blue")
    button_exit = Button(res_window,
                         text="Exit",
                         command=exit)
    label_file_explorer.grid(column=1, row=1)
    button_exit.grid(column=1, row=3)
    res_window.mainloop()
    return res_window

def close_window(window):
    window.destroy()



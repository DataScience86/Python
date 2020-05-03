# Learning how to use class with Tkinter

from Tkinter import *

class Application(Frame):
    """ A GUI application with three buttons. """

    def __init__(self, master):
        """ Initialize the Frame"""
        Frame.__init__(self, master)
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        """ Create 3 buttons that do nothing."""
        #create first button
        self.button1 =  Button(self,
                               text = "This is the first button")
        self.button1.grid()

        self.button2 =  Button(self)
        self.button2.grid()
        self.button2.configure(text = "This will show up the text")

        self.button3 =  Button(self)
        self.button3.grid()
        self.button3["text"] = "This is the third button"

root = Tk()
root.title("Lazy Buttons")
root.geometry("300x85")

app = Application(root)
root.mainloop()


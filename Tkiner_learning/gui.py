import Tkinter, Tkconstants, tkFileDialog
class TkApp(Tkinter.Frame):
    def __init__(self, root):
        Tkinter.Frame.__init__(self, root)
        # options for buttons
        button_opt = {'fill': Tkconstants.BOTH, 'padx': 5, 'pady': 5}
        #moved initiation of filename to here so it will print if no file has been selected
        self.filename = 'no-file'
        # define buttons
        Tkinter.Button(self, text='Open File', command=self.askopenfilename).pack(**button_opt)
        Tkinter.Button(self, text='Print File Path', command=self.printing).pack(**button_opt) #not the removal of variable.
        Tkinter.Button(self, text='Quit', command=self.quit).pack(**button_opt)

        # define options for opening or saving a file
        self.file_opt = options = {}
        options['defaultextension'] = '.twb'
        options['filetypes'] = [('All', '*')]
        options['initialdir'] = 'C:\\'
        options['parent'] = root
        options['title'] = 'Select a File'

    def askopenfilename(self):
        """Returns an opened file in read mode.
        This time the dialog just returns a filename and the file is opened by your own code.
        """       
        # get filename - edited to be part of self
        self.filename = tkFileDialog.askopenfilename(**self.file_opt)

        # open file on your own
        if self.filename:
            #print "askopenfilename value: " + self.filename
            return self.filename

    def printing(self):
        print "Value on click: " + self.filename


    def quit(self):
        root.destroy()

if __name__=='__main__':
        root = Tkinter.Tk()
        root.title("Path Printer")  
        TkApp(root).pack()
        root.mainloop()

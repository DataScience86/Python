from Tkinter import *

# To initialize the window
root = Tk()

""" Setting the button widget, it take many arguments like text. They are also
called as the properties associated with the widgets"""
button = Button(root, text = "Click Me!")
# The button will not show up until we use pack() method 
button.pack()

# To check the property associated we can use the following command
button['text'] = 'Press Me'

# Change the property of the widget, use the following
button['text'] = 'Press Me'

# We can also use config command to change/modify the property.
button.config(text = 'Push Me')

""" note: When using the square brackets keyword argument is passed in as a
string IE inside inverted commas. whereas with config command it is passed
as it is."""

# To View all of the properties associated with the widget use
button.config()

# To get the underline button name
str(button)

""" To View the results put print in from the above commands"""

root.mainloop()

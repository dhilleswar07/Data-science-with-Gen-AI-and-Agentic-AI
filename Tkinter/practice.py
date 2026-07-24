import tkinter as tk

#Create an main application window
root = tk.Tk()
root.title("Simple Tkinter App")
root.geometry("300x100") #set window size

# function to print helloworld in the console when button is clicked
def say_hello():
    print("Hello, World!")
    
# create a button that trigger say_hello function when clicked
hello_button = tk.Button(root, text="Click Me", command=say_hello)
hello_button.pack(pady=20) # pack the button into window  

# start the Tkinter event loop
root.mainloop()
    
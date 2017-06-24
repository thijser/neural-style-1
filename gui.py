import Tkinter as Tk
from tkFileDialog import askopenfilename
from PIL import Image, ImageTk
import os
filename="in.jpg"
########################################################################
class MyApp(object):
    """"""
    
    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        self.root = parent
        self.root.title("Main frame")
        self.frame = Tk.Frame(parent)
        self.frame.pack()
 
        btn = Tk.Button(self.frame, text="SelectImg", command=self.LoadImage)
        btn.pack()

        btn2 = Tk.Button(self.frame,text="run",command=self.runAll)
        btn2.pack()


        img = ImageTk.PhotoImage(Image.open(filename))
        panel = Tk.Label(root, image = img)
        panel.pack(side = "bottom", fill = "both", expand = "yes")
        root.mainloop()

	
    #----------------------------------------------------------------------
    def hide(self):
        """"""
        self.root.withdraw()
 
    #----------------------------------------------------------------------
    def LoadImage(self):
		filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
		print(filename)
		img = ImageTk.PhotoImage(Image.open(filename))
		panel = Tk.Label(root, image = img)
		panel.pack(side = "bottom", fill = "both", expand = "yes")
		root.mainloop()
    #----------------------------------------------------------------------
    def onCloseOtherFrame(self, otherFrame):
        """"""
        otherFrame.destroy()
        self.show()

    def ReadInreconOutput (self):
        fp = open('recon.out') # open file on read mode
        lines = fp.read().split("\n") # create a list containing all lines
        fp.close() # close file
        print(lines)
        l=""


 
    #----------------------------------------------------------------------
    def show(self):
        """"""
        self.root.update()
        self.root.deiconify()
 
    def runAll(self):

        os.system("python ./zhang/colorization/colorize.py -img_in "+filename+" img_out out/prepro.png")

        print("executing everything")
        tmp = os.popen("th imageRecon.lua -target_image prepro.png" ).read()
        print("ran recon")
        self.ReadInreconOutput()


        
        

		 
#----------------------------------------------------------------------
if __name__ == "__main__":
    root = Tk.Tk()
    root.geometry("800x600")
    app = MyApp(root)
    root.mainloop()

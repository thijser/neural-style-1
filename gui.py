import Tkinter as Tk
from tkFileDialog import askopenfilename
from PIL import Image, ImageTk
import os
from time import sleep

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


    def ReadInSelectorOutput(self):
        fp = open('selector.out')
        tmp=fp.read()
        fp.close()
        print(tmp)
        return tmp

    def ReadInreconOutput (self):
        fp = open('selector.out') # open file on read mode
        lines = fp.read().split("\n") # create a list containing all lines
        fp.close() # close file
        return lines
        


 
    #----------------------------------------------------------------------
    def show(self):
        """"""
        self.root.update()
        self.root.deiconify()
 

    def runAll(self):

        os.system("python ./zhang/colorization/colorize.py -img_in "+filename+" -img_out out/prepro.png > /dev/null" )

        print("executing everything")
        os.system("th imageRecon.lua -target_image out/prepro.png")
        print("ran recon")
        lines=self.ReadInreconOutput()
        contentstr=""
        for line in lines: 
            contentstr=contentstr+getNNNumStringName(line)

        contentstr=contentstr.replace(",","")
        contentstr=contentstr.replace(" ","_")
        os.system("rm t/Pictures/* -r")
        print("./runSelector.sh " + contentstr[1:] + " 2")
        os.system("./runSelector.sh " + contentstr[1:] + " 100")   
        SelectedStr=self.ReadInSelectorOutput()


        print("th neural_stylerun.lua -content_image out/prepro.png" + " -style_image "+ SelectedStr)
        os.system("th neural_stylerun.lua -content_image out/prepro.png" + " -style_image "+ SelectedStr)

        
        		 
#----------------------------------------------------------------------


def getNNNumStringName(num):
    fp = open('models/imagenet1000_clsid_to_human.txt')
    link = fp.read().split("\n") # create a list containing all lines
    for lin in link: 
       if num in lin:
           return lin.replace(num+': ','').replace("'",'')


if __name__ == "__main__":
    root = Tk.Tk()
    root.geometry("800x600")
    app = MyApp(root)
    root.mainloop()




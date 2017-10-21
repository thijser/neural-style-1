
import sys
from PIL import Image, ImageTk
import os
def ReadInSelectorOutput():
    fp = open('selector.out')
    tmp=fp.read()
    fp.close()
    print(tmp)
    return tmp

def ReadInreconOutput ():
    fp = open('selector.out') # open file on read mode
    lines = fp.read().split("\n") # create a list containing all lines
    fp.close() # close file
    return lines
        

def getNNNumStringName(num):
    fp = open('models/imagenet1000_clsid_to_human.txt')
    link = fp.read().split("\n") # create a list containing all lines
    for lin in link: 
       if num in lin:
           return lin.replace(num+': ','').replace("'",'')


def runal(filename,cnt):
        fnm=filename.replace("/","")
        fnm=filename.replace("\\","")

        os.system("python ./zhang/colorization/colorize.py -img_in "+filename+" -img_out out/prepro.png > /dev/null" )

        print("executing everything")
        os.system("th imageRecon.lua -target_image out/prepro.png")
        print("ran recon")
        lines=ReadInreconOutput()
        contentstr=""
        for line in lines: 
            contentstr=contentstr+getNNNumStringName(line)

        contentstr=contentstr.replace(",","")
        contentstr=contentstr.replace(" ","_")
        os.system("rm t/Pictures/* -r")
        print("./runSelector.sh " + contentstr[1:] + " 300" +" "+str(cnt*30000))
        os.system("./runSelector.sh " + contentstr[1:] + " 300"+ " "+str(cnt)) 
        #os.system("./runSelector.sh ")  
        SelectedStr=ReadInSelectorOutput()
        
                
        
        strcmd="th neural_stylerun.lua -content_image out/prepro.png " + " -style_image "+ SelectedStr + " -output_image " +fnm+"cnt="+str(cnt)+"_out.png"
        print(strcmd)
        os.system(strcmd)


        with open("executelog.txt", "a") as myfile:
             myfile.write(strcmd)
             myfile.write("\n")



 

for imgname in sys.argv:
  for cnt in range(0,6):
   if ".png" in imgname or ".jpg" in imgname:
     runal(imgname,cnt)



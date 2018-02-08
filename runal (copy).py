
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


def runal(filename,cnt,clv):
        print("runal:" + str(cnt) +" " +str(clv))
        fnm=filename.replace("/","")
        fnm=filename.replace("\\","")

        os.chdir("zhang/colorization")
        os.system("python colorize.py -img_in ../../"+filename+" -img_out ../../out/prepro.png" )
        os.chdir("../..")
        
        print("executing everything")
        os.system("th imageRecon.lua -target_image out/prepro.png")
        print("ran recon")
        lines=ReadInreconOutput()
        contentstr=""
        for line in lines: 
            contentstr=contentstr+getNNNumStringName(line)
        print(contentstr)

        contentstr=contentstr.replace(",","")
        contentstr=contentstr.replace(" ","_")
        #os.system("rm t/Pictures/* -r")
        print("./runSelector.sh " + contentstr[1:] + " 300"+ " "+str(clv*30000)+" "+str(cnt))

        os.system("./runSelector.sh " + contentstr[1:] + " 300"+ " "+str(clv*30000)+" "+str(cnt)) 
        SelectedStr=ReadInSelectorOutput()
        
                
        
        strcmd="th neural_stylerun.lua -content_image out/prepro.png " + " -style_image "+ SelectedStr + " -output_image "+fnm+"cnt="+str(cnt)+"clv="+str(clv*30000)+"_out.png"
        print(strcmd)
        os.system(strcmd)


        with open("executelog.txt", "a") as myfile:
             myfile.write(strcmd)
             myfile.write("\n")



 

for imgname in sys.argv:
  for cnt in range(1,20):
    for clv in range(0,20):
      if ".png" in imgname or ".jpg" in imgname:
        runal(imgname,cnt,clv)



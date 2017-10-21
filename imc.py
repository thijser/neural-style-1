import sys

import numpy as np
import cv2



def main (orig,colloc):


  orig = cv2.imread(orig,1)
  #orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
  height,width = orig.shape[:2]

  col = cv2.imread(colloc,1)
  col=cv2.resize(col,(width,height),interpolation=cv2.INTER_CUBIC)
  cv2.imwrite('messigray3.png',orig)
  print(colloc)
  res=cv2.pyrMeanShiftFiltering(orig,20,45,3)
  res=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
  


  cv2.imwrite('messigray.png',res)
  sects=list_seg_regs(res,col,orig)
  cv2.imwrite('messigray2.png',sects)




def alterImage(mask,image,out,confidence):

  m=mask[0].astype(np.uint8)
  avg=cv2.mean(image,m)
  mask=getSelection(mask[0],confidence)
  avg=avg[:-1]
  out[mask>0]=avg

  
def getSelection(mask,confidenc):
  m=np.array(mask, copy=True)  
  confidence=np.array(confidenc,copy=True)
 
  confidence[m == 0] = float('NaN')
#  v=np.nanpercentile(confidence,0.1)
#  confidence=np.nan_to_num(confidence)

#  m[confidence>v]=0
  return m
  
  
def getConfidence(orig,target):
  #print(orig)
  go=cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
  gt=cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
  confidence=np.absolute(go-gt)
  cv2.imwrite('confidence.png',confidence)
  return confidence
  

def list_seg_regs(a,colimage,orig): 
    height,width = orig.shape[:2]
    result=orig
    confidence=255-getConfidence(orig,colimage).astype(float)
    for i in np.unique(a):
        print(i)
        ret, l = cv2.connectedComponents((a==i).astype(np.uint8))
        for j in range(1,ret):
            out = []
            out.append((l==j).astype(int)) #skip .astype(int) for bool
            alterImage(out,colimage,result,confidence)
    return result


main(sys.argv[1],sys.argv[2])


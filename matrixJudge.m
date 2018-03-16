function[res] = matrixJudge(loc)
raw=csvread(loc)
v1=squeeze(raw(:,2))
v2=squeeze(raw(:,3)/30000)
v3=squeeze(raw(:,1))


res=zeros(max(v1),max(v2))
res(res==0)=NaN
for v=0:max(v3) 
   
   res(v1(v+1),v2(v+1)+1)=v3(v+1) 
end


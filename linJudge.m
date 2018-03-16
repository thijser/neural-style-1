function[res] = linJudge(loc)
raw=csvread(loc)
max=41
v2=squeeze(raw(:,2))
v1=squeeze(raw(:,1))

res=zeros(41,1)

res(v1+1)=v2
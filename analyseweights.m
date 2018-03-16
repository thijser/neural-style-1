a2=linJudge('/home/thijser/Desktop/tvloss/out/spitfireevalres.txt')

a=filter(w,1,a2)
w=gausswin(2)
ag=filter(w,1,a)
ag(1)=ag(1)*4
avg=ag+0.11


x=linspace(0,40,41)
xaxis=((1.4).^x)/1400
avg=rot90(avg)
avg=avg/mean(avg)*20
yaxis=avg
pt=semilogx(xaxis,avg)
xlim([0,max(xaxis)])

figure
plot(avg)
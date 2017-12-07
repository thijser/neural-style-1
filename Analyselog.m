[values1,times1]=makelog('log1');
[values2,times2]=makelog('log2');
[values3,times3]=makelog('log3');
[values4,times4]=makelog('log4');

values3(1)=values3(2)

[values5,times5]=makelog('log5');

figure(2)
plot(times1,values1)
hold on
plot(times2,values2)
plot(times3,values3)
plot(times4,values4)
xlabel('cpu time(s)')
ylabel('combined loss')
legend('top','top mate','random','roullete mate' , 'r2')
x =rand(9,1)*10
y = x + 0.1*randn(size(x));

p = polyfit(x,y,10)
t2 = 0:0.01:max(x)
y2 = polyval(p,t2);
figure

plot(x,y,'o',t2,y2)
hold on
pp = polyfit(x,y,1)
t3 = 0:0.01:max(x)
y3 = polyval(pp,t2);
plot(t3,y3)

legend('datapoints','10th order polynomial','linear equation')
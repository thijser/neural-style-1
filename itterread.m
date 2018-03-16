a1=csvread('/home/thijser/Desktop/itterlog/iterlog0.7637540931584.txt',' ')
a2=csvread('/home/thijser/Desktop/itterlog/iterlog0.013412262348514.txt' ,' ')
a3=csvread('/home/thijser/Desktop/itterlog/iterlog0.15482201618382.txt',' ')
a4=csvread('/home/thijser/Desktop/itterlog/iterlog0.023201430189723.txt',' ')
a5=csvread('/home/thijser/Desktop/itterlog/iterlog0.27453782964684.txt',' ')
a6=csvread('/home/thijser/Desktop/itterlog/iterlog0.040255252321508.txt',' ')
a7=csvread('/home/thijser/Desktop/itterlog/iterlog0.72095777953231.txt',' ')
a8=csvread('/home/thijser/Desktop/itterlog/iterlog0.79712806119799.txt',' ')
a9=csvread('/home/thijser/Desktop/itterlog/iterlog0.89375257899052.txt',' ')


a=a2+a4+a6+a7+a8+a9
figure
loglog(a/6)
xlabel('iterations')
ylabel('transfer loss')
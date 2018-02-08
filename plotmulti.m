logs = dir('log1/*.txt');

hold off
for log = logs' 
    t=load(log.name, ' ')
    plot(t(2:end,1),t(2:end,3),'g')
    hold on
end



logs = dir('log2/*.txt');

for log = logs' 
    t=load(log.name, ' ')
    plot(t(2:end,1),t(2:end,3),'r')
    hold on
end


logs = dir('log3/*.txt');

for log = logs' 
    t=load(log.name, ' ')
    plot(t(2:end,1),t(2:end,3),'m')
    hold on
end


logs = dir('log4/*.txt');

for log = logs' 
    t=load(log.name, ' ')
    plot(t(2:end,1),t(2:end,3),'b')
    hold on
end


logs = dir('log5/*.txt');

for log = logs' 
    t=load(log.name, ' ')
    plot(t(2:end,1),t(2:end,3),'y')
    hold on
end


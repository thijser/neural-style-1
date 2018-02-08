function [values,times] = makelog(log )
logs = dir(strcat(log,'/*.txt'))
k=0
for log = logs' 
    k=k+1
    
    t{k}=load(log.name, ' ')
end

stamp=3
data=1
times=t{1}(:,stamp)
values=t{1}(:,data)

for i=2:k 
    all_times = union(times, t{i}(:,stamp))';
    values1_interp = interp_left(times,values, all_times);
    values2_interp = interp_left(t{i}(:,stamp), t{i}(:,data), all_times);
    v_sum = values1_interp + values2_interp;
    times=all_times
    values=v_sum
end


end

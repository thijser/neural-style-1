a1=matrixJudge("/home/thijser/Desktop/evalres/kingfisherevalresRO.txt")
a2=matrixJudge("/home/thijser/Desktop/evalres/chickadeeevalreska.txt")
a3=matrixJudge("/home/thijser/Desktop/evalres/kingfisherevalreska.txt")
a4=matrixJudge("/home/thijser/Desktop/evalres/queenevalreska1.txt")
a5=matrixJudge("/home/thijser/Desktop/evalres/soccerevalreska.txt")
a6=matrixJudge("/home/thijser/Desktop/evalres/bouqetteevalreska.txt")

all=cat(3,a1,a2,a3,a4,a5,a6)
result=nanmean(all,3)

hm = heatmap(round(imgaussfilt(result,1)));
% Change x axis tick labels to some array
% hm.XDisplayLabels = [0:3e4:(numel(hm.XDisplayLabels)-1)*3e4];
% Change x axis tick labels to the current labels * 3e4 (same result)
hm.XDisplayLabels = str2double(hm.XDisplayData)*3e4 - 3e4;

% To add axes labels to *most* chart types in MATLAB, use `XLabel` and `YLabel` properties
hm.XLabel = 'colour weight';
hm.YLabel = 'number of images';
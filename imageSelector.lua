require 'torch'
require 'nn'
require 'image'
require 'optim'
require('image')
require("neural_style.lua")
require 'loadcaffe'
require 'EMDCriterion'

local cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-image_size', 64, 'Maximum height / width of generated image')


cmd:option('-target_image', 'out/prepro.png')
cmd:option('-avaible_images', 'in.jpg,tankbw.jpg,hawaii.jpg,aeaecb2791801e2bfeb37f281599a885.jpg,tt.png,tt2.jpg')
cmd:option('image_count',5)
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers-deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-selectorGenerations', 10)
cmd:option('-selectorWidth', 5)
cmd:option('-selectorDepth', 6)
cmd:option('neural_Content_eval_Layer' ,'conv5_4') 



--function from https://gist.github.com/MihailJP/3931841
function copy (t) -- shallow-copy a table
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do target[k] = v end
    setmetatable(target, meta)
    return target
end


 function randomSelectImages(avImages, count)
	local notSelected=copy(avImages)
	local selected={}

	for i=1, count do
		rnd=math.random(#notSelected)
		selected[i]=notSelected[rnd]
		notSelected[rnd]=nil

	end
	return selected
	
	
end


cache={}
 function evaluate(params,selectedImages)

if(cache[selectedImages~=nil]) then
	return cache[selectedImages]
end
if(#selectedImages~=params.image_count) then
	return 999999999999999999999999999999999999999999999999999999999999999
	end

	coleval=evalHueImages(params,selectedImages) *5000*18
    neuroval=neuralEval(params,selectedImages)
	evalValue=coleval+neuroval
	cache[selectedImages]=evalValue
    print("coleval="..coleval )
    print("neuroval="..neuroval )
		
    return evalValue
	
end

 function SelectTop(params,images,count)
	local values={}  
	for k,v in pairs(images) do
		local noError,res=pcall(evaluate,params,v)
		--res=evaluate(params,v) noError=true
	if noError then 
		values[#values+1]= {v, res}
	else

    values[v] = 9999999999999999999999999999999999999999999999999999999999
  	end
  end
function compare(a,b)
  return a[2] < b[2]
end
	ret={}
  table.sort(values,compare)

	for  i=1,count do
		ret[i]=values[i][1]
	end

	print(values[1])
	return ret
end


function contains(array,value)

	for i=1,#array do
		if(array[i]==value) then
			return true
		end
	end
	return false
end

function addRange(array1,array2)
	for i=1,#array2 do
		array1[#array1+1]=array2[i]
	end
end

function mutate(params,selected,avImages)
	for x = 1,	params.selectorWidth do
		addRange(selected,mutateSingle(params.selectorDepth,selected[x],avImages))	
	end
	return selected
end

function mutateSingle(count,orig,avImages)

	ret={}
	for i=1,count do
		rnd=math.random(#avImages)

		while(contains(orig,avImages[rnd])) do
				rnd=math.random(#avImages)

		end
		ret[i]=copy(orig)
		orig[math.random(#orig)]=avImages[rnd]
	end	
	return ret
end

 function evolve(params,avImages)
	
	local selected={}
	for i=1,params.selectorDepth*params.selectorDepth do
		selected[i]=randomSelectImages(avImages,params.image_count)
	end
	
	for i=1,params.selectorGenerations do
		selected=SelectTop(params,selected,params.selectorWidth)
		selected=mutate(params,selected,avImages)
	end
	return SelectTop(params,selected,1)
end
 function main(params)
	local avImages = params.avaible_images:split(',')
	local selectedImages=evolve(params,avImages)
    print('selected:')
	print(tabtostr(selectedImages[1]))
    print("writing selected")
 	file= io.open("selector.out","w")
	file:write(tabtostr(selectedImages[1]))
	print("done writing selector")
	file:close()
		

end


function tabtostr(tab)
    str=""
	for k,v in pairs(tab) do 
        print(v)
	    str=str..','..v	
			
	end
	str=str:sub(2,str:len())
    str=string.gsub(str, "\n", "")
    return str
end
local cnn = nil
function buildSelectorNetworkOrGet(params)

	if cnn~=nil then
		return cnn
	end
	  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'clnn'
      require 'cltorch'
      cltorch.setDevice(params.gpu + 1)
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end


  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
  cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      cnn:cuda()
    else
      cnn:cl()
    end
  end
  return cnn
end

 function neuralEval(params, selectedImages)
    collectgarbage()
	cnn=buildSelectorNetworkOrGet(params)
  targetImage_caffe = image.load(params.target_image, 3)
  targetImage_caffe = image.scale(targetImage_caffe, 224,224, 'bilinear')
  targetImage_caffe = preprocess(targetImage_caffe):float()

  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      targetImage_caffe = targetImage_caffe:cuda()
    
    else
      targetImage_caffe = targetImage_caffe:cl()
    end
  end


	if(gram==nil) then
     	gram = GramMatrix():float()
	end
	cnn:forward(targetImage_caffe)

        

        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            gram = gram:cuda()
          else
            gram = gram:cl()
          end
        end

	
		local targetStructure=nil
		
	  for i = 1, #cnn do
		layer=cnn:get(i)
		if layer.name=='conv5_4' then
			gram:forward(layer.output)			
			targetStructure=gram.output:clone()

		end
	end

	local contentDist=0
	
	for imageIndex =1, #selectedImages do
		    collectgarbage()
			
  selectedImage_caffe = image.load(selectedImages[imageIndex], 3)
  selectedImage_caffe = image.scale(selectedImage_caffe, 224,224, 'bilinear')
  selectedImage_caffe = preprocess(selectedImage_caffe):float()

  		if params.gpu >= 0 then
  		  if params.backend ~= 'clnn' then
    		  selectedImage_caffe = selectedImage_caffe:cuda()
    
   			  else
      			 selectedImage_caffe = selectedImage_caffe:cl()
    		end
		  end
			cnn:forward(selectedImage_caffe)
			
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            gram = gram:cuda()
          else
            gram = gram:cl()
          end
        end
			 for i = 1, #cnn do
		layer=cnn:get(i)
		if layer.name=='conv5_4' then
			gram:forward(layer.output)			
			contentDist=contentDist+torch.sum(torch.abs(gram.output-targetStructure))


		end
	end
		
	end



	return contentDist
end



  
function evalHueImages(params,tarImages)

  
  local images={}
  for i = 1,#tarImages do

  local targetImage_caffe = image.load(tarImages[i], 3)
    targetImage_caffe = image.scale(targetImage_caffe, params.image_size, 'bilinear')
	images[i]=targetImage_caffe
	
  end

 return ColourCompareHistEMD(images)

end

function ColourCompareHistEMD (images)

	sumdistance=0
    r1={}
	r2={}
	r3={}
	for i=1,#images do 

		v1=images[i][{{},1}]
		v2=images[i][{{},2}]
		v3=images[i][{{},3}]
	    local hist = torch.histc(v1,20,-1,1)
	    r1[i] = hist:reshape(1,hist:nElement())
	    local hist = torch.histc(v2,20,-1,1)
	    r2[i] = hist:reshape(1,hist:nElement())
	    local hist = torch.histc(v3,20,-1,1)
	    r3[i] = hist:reshape(1,hist:nElement())

	end
   
   criterion = nn.EMDCriterion()
    for i=1,#images do 
       for j=1,#images do 

			local loss = criterion:forward(r1[i],r1[j])
			local loss2 = criterion:forward(r2[i],r2[j])
			local loss3 = criterion:forward(r3[i],r3[j])
			sumdistance = sumdistance + loss+loss2+loss3
			
       end
   end
return sumdistance
end

function ColourCompareHSL(images)
	sumdistance=0;
	k={}
	for i = 1, #images do 
		local imagesum=torch.sum(torch.sum(images[i],2),3)
		k[i]=imagesum

	end
   
	for i = 1, #images do 
        print(torch.max(images))
        print(torch.min(images))
		hsvA=RGBToHSV(k[i][1][1][1],k[i][2][1][1],k[i][3][1][1]) 
    	for j = 1, #images do
			hsvB=RGBToHSV(k[j][1][1][1],k[j][2][1][1],k[j][3][1][1]) 
			hueDistance = math.min(math.abs(hsvB[2]-hsvA[2]), 360-math.abs((hsvB[2]-hsvA[2])));
			lightDistance=math.abs(hsvA[1]-hsvB[1])
			satDistance=math.abs(hsvA[2]-hsvB[2])
			sumdistance=sumdistance+hueDistance+lightDistance+satDistance

        end
    end

	return sumdistance
end






function RGBToHSV( red, green, blue )
	-- Returns the HSV equivalent of the given RGB-defined color https://gist.github.com/GigsD4X/8513963
	-- (adapted from some code found around the web)

	local hue, saturation, value;

	local min_value = math.min( red, green, blue );
	local max_value = math.max( red, green, blue );

	value = max_value;

	local value_delta = max_value - min_value;

	-- If the color is not black
	if max_value ~= 0 then
		saturation = value_delta / max_value;

	-- If the color is purely black
	else
		saturation = 0;
		hue = 0;
		return {hue, saturation, value};
	end;

	if red == max_value then
		hue = ( green - blue ) / value_delta;
	elseif green == max_value then
		hue = 2 + ( blue - red ) / value_delta;
	else
		hue = 4 + ( red - green ) / value_delta;
	end;

	hue = hue * 60;
	hue=hue%360;
	if value ~= value then
	 value=0
	end if hue ~= hue then
 	hue=0
	end if saturation ~= saturation then
 	saturation=0
	end
	return {value,hue, saturation};
end


math.randomseed(os.time())
local params = cmd:parse(arg)
return main(params)



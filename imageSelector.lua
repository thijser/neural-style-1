require 'torch'
require 'nn'
require 'image'
require 'optim'
require('image')
require 'loadcaffe'
local cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-image_size', 64, 'Maximum height / width of generated image')


cmd:option('-target_image', 'tank.jpg')
cmd:option('-avaible_images', 'tank.jpg,tankbw.jpg,hawaii.jpg,aeaecb2791801e2bfeb37f281599a885.jpg,tt.png,tt2.jpg')
cmd:option('image_count',3)
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers-deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-selectorGenerations', 100)
cmd:option('-selectorWidth', 5)
cmd:option('-selectorDepth', 6)



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


 function evaluate(params,selectedImages)
	coleval=evalHueImages(params,selectedImages) 
    neuroval=neuralEval(params,selectedImages)
	return coleval+neuroval
end

 function SelectTop(params,images,count)
	local values={}  
	for k,v in pairs(images) do
		local noError,res=pcall(evaluate,params,v)

	if noError then 
		values[v]=res
	else

    values[v] = 9999999999999999999999999999999999999999999999999999999999
  	end
  end
function compare(a,b)
  return a[2] < b[2]
end

  table.sort(values,compare)
  print(values)
end

 function evolve(params,avImages)
	
	local selected={}
	for i=1,params.selectorDepth*params.selectorDepth do
		selected[i]=randomSelectImages(avImages,params.image_count)
	end
SelectTop(params,selected,5)
	
end
 function main(params)
	local avImages = params.avaible_images:split(',')
	evolve(params,avImages)
	
	

end

 function neuralEval(params, selectedImages)
    collectgarbage()
	if(true) then
		return 1
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
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      cnn:cuda()
    else
      cnn:cl()
    end
  end
 
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

 netimage=cnn:forward(targetImage_caffe)

end

function evalHueImages(params,tarImages)

  
  local images={}
  for i = 1,#tarImages do

  local targetImage_caffe = image.load(tarImages[i], 3)
    targetImage_caffe = image.scale(targetImage_caffe, params.image_size, 'bilinear')
	images[i]=targetImage_caffe
	
  end

 return ColourCompareHSL(images)

end

function ColourCompareHSL(images)
	sumdistance=0;
	k={}
	for i = 1, #images do 
		local imagesum=torch.sum(torch.sum(images[i],2),3)
		k[i]=imagesum

	end

	for i = 1, #images do 
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




function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
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



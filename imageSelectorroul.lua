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
cmd:option('-image_count',5)
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers-deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-selectorGenerations', 200999)
cmd:option('-timestop',180)
cmd:option('-selectorWidth', 4)
cmd:option('-selectorDepth', 5)
cmd:option('-neural_Content_eval_Layer' ,'conv5_4') 
cmd:option('-colweight' , 90000)
cmd:option('mode','topmate') --roulmate,topmate,top,roul,rand
cmd:option('mutatechance',1)


--function from https://gist.github.com/MihailJP/3931841
function copy (t) -- shallow-copy a table
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do target[k] = v end
    setmetatable(target, meta)
    return target
end

NumEval=0

starttime=os.time()

math.randomseed(starttime)
rndsavename=math.random()


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
		return 99999999999999999999999999999999999999999999999999999999999999999999
	end
	coleval=evalHueImages(params,selectedImages) *params.colweight
    neuroval=neuralEval(params,selectedImages)
	evalValue=coleval+neuroval
	cache[selectedImages]=evalValue
	NumEval=NumEval+1
    print("neuro:")
	print(neuroval)
    print("col:")
	print(coleval)
    print("blub")
    return evalValue
	
end
	function compare(a,b)

	  if a==nil  then return false end
	  if a[2] ==nil then return false end

	  if b==nil  then return false end
	  if b[2] ==nil then return false end
	  
	 
	  return a[2] < b[2]
	end

function SelectTop(params,images,count)

	local values={}  
	for k,v in pairs(images) do
		local noError,res=pcall(evaluate,params,v)
		--res=evaluate(params,v) noError=true
		if noError and res~=nil then 

			values[#values+1]= {v, res}
		else
		
	   	 values[v] = {v,999999999999999999999999999999999999999999999999999999999999999999999}
	  	end
    
 	 end

	ret={}
    table.sort(values,compare)


	for  i=1,count do
		ret[i]=values[i][1]
	end

	score=values[1][2]
	return ret
end
function randgen(params,images,avImages)


		rnd=randomSelectImages(avImages,params.image_count)
        rndr={}
		local noError,res=pcall(evaluate,params,rnd)
		--res=evaluate(params,v) noError=true
		if noError and res~=nil then 

			rndr[0]={rnd, res}

		else
		
	   	 rndr[0] = {rnd,99999999999999999999999999999999999999999999999999999999999999999999999999999}

	  	end
    
  print(rndr)
 print(" = rnd , images=")
	print(images)
	if images[0]==nil or rndr[0][2]<images[0][2] then 

	score=rndr[0][2]
		return rndr
	else 
	score=images[0][2]
		return images
	end
end

function roulrun(params,images)
	local values={}  
	for k,v in pairs(images) do
		local noError,res=pcall(evaluate,params,v)
		--res=evaluate(params,v) noError=true
		if noError and res~=nil then 

			values[#values+1]= {v, res}
		else
		
	   	 values[v] = {v,99999999999999999999999999999999999999999999999999999999999999999999999999999}
	  	end
    
 	 end

		ret={}
    table.sort(values,compare)
    ret[1]=values[1][1]
	score=values[1][2]


	for i=2, params.selectorDepth*params.selectorDepth  do 

	firstpick=rollwheel(values)


		secondpick=rollwheel(values)
        mated=mate(firstpick,secondpick)
		print("go")
		print(#ret+1)
		if math.random()<params.mutatechance then
			mated=mutateSingle(1,mated,params.avaible_images:split(','))[1]

		end
		print(#ret+1)
		ret[#ret+1]=mated
		
	end

    return ret
end
function accres(zerovals)
    local sum=0
    local ret={} 
    index=0
    for a,b in pairs(zerovals) do
		
		if type(b)~='number'  then
			sum=sum+1/b[2]
			
			ret[index]={b[1],sum}
		end
	end
	return ret

end


function roll(t)
   max=t{#t-1}
   local r = math.random()*max
   for i,n in pairs(t) do
       if r <= n{2} then
           return n{1}
       end
   end
end

function sumtrans (v)
  return 1/v
end
function rollwheel(t)
   max=0
   for i,n in pairs(t) do
        if(type(n)=='number') then
           print("number made it into wrong table loc, ignored  >>>")
           print(n)
			else
		max=max+sumtrans(n[2])
		end
   end


   sum=0
   local r = math.random()*max
   for i,n in pairs(t) do
        if(type(n)~='number') then
		sum=sum+sumtrans(n[2])
   	       if r <= sum then
    	       return n[1]
    	   end
		end
   end
end


score=0

function log(params)

    logfile = assert(io.open('epochlog'..params.mode..rndsavename..'.txt', "a"))
    print("log: " .. score )
    logfile:write(score,' ',NumEval,' ',os.time()-starttime,"\n")
    logfile:close()
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

function mutate(selected,avImages,num)

	for x = 1,	#selected do
		addRange(selected,mutateSingle(num,selected[x],avImages))	
	end


	return selected
end

function mutateSingle(count,orig,avImages)

	res={}
	for i=1,count do
		rnd=math.random(#avImages)

		while(contains(orig,avImages[rnd])) do
				rnd=math.random(#avImages)

		end
		res[i]=copy(orig)
		orig[math.random(#orig)]=avImages[rnd]
	end	
	return res
end

function evolve(params,avImages)
	
	local selected={}
	if params.mode=='rand' then
       initialSize=1
    else    
		initialSize=params.selectorDepth*params.selectorDepth
    end
	for i=1,initialSize do
		selected[i]=randomSelectImages(avImages,params.image_count)
	end
	
	for i=1,params.selectorGenerations do

        if(os.time()-starttime>params.timestop)  then
			break
        end
 		if params.mode=='top' then
			selected=SelectTop(params,selected,params.selectorWidth)
			selected=mutate(selected,avImages,params.selectorDepth)

		end
		if params.mode=='topmate'  then
			selected=SelectTop(params,selected,params.selectorWidth)
			for i=1,#selected do 
				for j=1,#selected do
					selected[#selected+1]=mate(selected[i],selected[j])
				end
			end
			mutate(params,selected,avImages,1)

		end

		if params.mode=="roul" or params.mode=='roulmate' then
			selected=roulrun(params,selected)
		end
		if params.mode=="rand"  then
			selected=randgen(params,selected,avImages)
		end


    log(params)
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

function mate(parentA, parentB)
   A=copy(parentA)
   B=copy(parentB)
   res=intersection(A,B)
   for k,v in pairs(res) do 
      removeByValue(A,v)
      removeByValue(B,v)
   end

   for i=1,#A do 
	sel=nil
     if(math.random(2)==1) then 
		sel=A
	else
		sel=B
	end
     res[#res+1]=sel[i]
   end

 return res
   
end

function ElementwiseMin(a,b)
	b[a:lt(b)]=a
	return b
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
	local distmat=torch.Tensor(targetStructure:size())
	distmat=distmat:fill(1e308):cuda()
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
			distmat=ElementwiseMin(distmat,torch.abs(gram.output-targetStructure))

            
		end
	end
		
	end
   contentDist=torch.sum(distmat)


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
   sumdistance=sumdistance/(#images*#images)/3
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



local params = cmd:parse(arg)


--from http://www.phailed.me/2011/02/common-set-operations-in-lua/
local function find(a, tbl)
	for _,a_ in ipairs(tbl) do if a_==a then return true end end
end
--from http://www.phailed.me/2011/02/common-set-operations-in-lua/
function intersection(a, b)
	local ret = {}
	for _,b_ in ipairs(b) do
		if find(b_,a) then table.insert(ret, b_) end
	end
	return ret
end


function removeByValue(table1,value)
  for k,v in pairs(table1)do
    if v == value then
       table.remove(table1, k)
     break
   end
 end
end
return main(params)



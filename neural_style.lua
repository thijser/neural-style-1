require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'


local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'examples/inputs/seated-nude.jpg',
           'Style target image')
cmd:option('-style_blend_weights', 'nil')
cmd:option('-content_image', 'examples/inputs/tubingen.jpg',
           'Content target image')
cmd:option('-image_size', 768, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Optimization options
cmd:option('-content_weight', 5e0)

cmd:option('-style_weight', 2e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 5000)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random', 'random|image')
cmd:option('-optimizer', 'adam', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 5)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-original_colors', 0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

function main(params)
    collectgarbage()
print(collectgarbage("count")*1024)

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
      collectgarbage()
  content_image = image.load(params.content_image, 3)
  content_image = image.scale(content_image, params.image_size, 'bilinear')
  local content_image_caffe = preprocess(content_image):float()
  
  local style_size = math.ceil(params.style_scale * params.image_size)
  local style_image_list = params.style_image:split(',')
  local style_images_caffe = {}
  for _, img_path in ipairs(style_image_list) do
    local img = image.load(img_path, 3)
    img = image.scale(img, style_size, 'bilinear')
    local img_caffe = preprocess(img):float()
    table.insert(style_images_caffe, img_caffe)
  end

  -- Handle style blending weights for multiple style inputs
  local style_blend_weights = nil
  if params.style_blend_weights == 'nil' then
    -- Style blending not specified, so use equal weighting
    style_blend_weights = {}
    for i = 1, #style_image_list do
      table.insert(style_blend_weights, 1.0)
    end
  else
    style_blend_weights = params.style_blend_weights:split(',')
    assert(#style_blend_weights == #style_image_list,
      '-style_blend_weights and -style_images must have the same number of elements')
  end
  -- Normalize the style blending weights so they sum to 1
  local style_blend_sum = 0
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = tonumber(style_blend_weights[i])
    style_blend_sum = style_blend_sum + style_blend_weights[i]
  end
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = style_blend_weights[i] / style_blend_sum
  end
  

  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      content_image_caffe = content_image_caffe:cuda()
      for i = 1, #style_images_caffe do
        style_images_caffe[i] = style_images_caffe[i]:cuda()
      end
    else
      content_image_caffe = content_image_caffe:cl()
      for i = 1, #style_images_caffe do
        style_images_caffe[i] = style_images_caffe[i]:cl()
      end
    end
  end
  
  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")

  -- Set up the network, inserting style and content loss modules
  local content_losses, style_losses = {}, {}
  local next_content_idx, next_style_idx = 1, 1
  local net = nn.Sequential()
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    if params.gpu >= 0 then
      if params.backend ~= 'clnn' then
        tv_mod:cuda()
      else
        tv_mod:cl()
      end
    end
    net:add(tv_mod)
  end
  for i = 1, #cnn do

    if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            avg_pool_layer:cuda()
          else
            avg_pool_layer:cl()
          end
        end
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
        local target = net:forward(content_image_caffe):clone()
        local norm = params.normalize_gradients
        local loss_module = nn.ContentLoss(params.content_weight, target, norm):float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            loss_module:cuda()
          else
            loss_module:cl()
          end
        end
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end
      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local gram = GramMatrix():float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            gram = gram:cuda()
          else
            gram = gram:cl()
          end
        end
        local target = nil
        for i = 1, #style_images_caffe do
          local target_features = net:forward(style_images_caffe[i]):clone()
          local target_i = gram:forward(target_features):clone()
          target_i:div(target_features:nElement())
          target_i:mul(style_blend_weights[i])
          if i == 1 then
            target = target_i
          else
            target:add(target_i)
          end
        end
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight, target, norm):float()
        if params.gpu >= 0 then
          if params.backend ~= 'clnn' then
            loss_module:cuda()
          else
            loss_module:cl()
          end
        end
        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1
      end
    end
end

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  gram = nil
  target=nil

  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()
  
  -- Initialize the image
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
  end
  local img = nil
  if params.init == 'random' then
    img = torch.randn(content_image:size()):float():mul(0.001)
  elseif params.init == 'image' then
    img = content_image_caffe:clone():float()
  else
    error('Invalid init type')
  end
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      img = img:cuda()
    else
      img = img:cl()
    end
  end
  content_imageprep= content_image--preprocess(content_image)
  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = params.num_iterations,
      verbose=true,
    }
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,

    }
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  local function maybe_print(t, loss)
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
      print(string.format('Iteration %d / %d', t, params.num_iterations))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style %d loss: %f', i, loss_module.loss))
      end
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function maybe_save(t)
    local should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save then
      local disp = deprocess(img:double())
      disp = image.minmax{tensor=disp, min=0, max=1}
      local filename = build_filename("post",params.output_image, t)
      local prefilename = build_filename("prepost",params.output_image, t)
      if t == params.num_iterations then
        filename = params.output_image
      end

      -- Maybe perform postprocessing for color-independent style transfer
      if params.original_colors == 1 then
        disp = original_colors(content_image, disp)
      end
	image.save(prefilename,disp)
	disp=deprocess(TransferBlack(preprocess(disp),content_image):double())
	
      image.save(filename, disp)
    end
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this function many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  local num_calls = 0
  local function feval(x)

    num_calls = num_calls + 1
    net:forward(x)
	loss=0
    local grad = net:updateGradInput(x, dy)
	if num_calls%3==-10 then
		
	grey=GreyGradientLAB(content_imageprep,img)
	loss=torch.sum(torch.abs(grey))
	grad=grey+grad 
	end

	img=TransferBlack(img,content_imageprep)
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    maybe_print(num_calls, loss)
    maybe_save(num_calls)

    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = optim.lbfgs(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    for t = 1, params.num_iterations do
      local x, losses = optim.adam(feval, img, optim_state)
			--img=postProccess(img,content_imageprep,params)
			--img=TransferBlack(img,content_imageprep)
			--optim_state.v=nil
			--optim_state.m=nil

    end
  end
end
  

function build_filename(anote,output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)


  return string.format('%s/out/%s_%s_%d.%s',directory, basename,anote, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
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


-- Combine the Y channel of the generated image and the UV channels of the
-- content image to perform color-independent style transfer.
function original_colors(content, generated)
  local generated_y = image.rgb2yuv(generated)[{{1, 1}}]
  local content_uv = image.rgb2yuv(content)[{{2, 3}}]
  return image.yuv2rgb(torch.cat(generated_y, content_uv, 1))
end


-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
end

function ContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

function GreyGradient (orig,img)

	d=orig:double()
	i=preprocess(img)

	g=d-i
	g=g/90
	g=torch.sum(g,1)

	g=torch.Tensor(g,g,g)
	r=torch.zeros(3,485,768)
	r[1]=g
	r[2]=g
	r[3]=g

	return r:cuda()
end


function GreyGradientLAB (orig,img)
	imgdepros=deprocess(img:double())
	origdouble=orig:double()
	imglab=image.rgb2lab(imgdepros)
	origlab=image.rgb2lab(origdouble)
	imglab[1]=origlab[1]
	imgtarget=image.lab2rgb(imglab):cuda()

	return imgtarget-img

	
	
end
-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = target
  self.loss = 0
  
  self.gram = GramMatrix()
  self.G = nil
  self.crit = nn.MSECriterion()
    collectgarbage()
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target)
  dG:div(input:nElement())
  self.gradInput = self.gram:backward(input, dG)
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

local greyLoss, parent = torch.class('nn.greyLoss', 'nn.Module')
function greyLoss:__init(strength,target)
	parent.__init(self)
    self.x_diff = torch.Tensor()
    self.y_diff = torch.Tensor()
    self.target=target
end

function greyLoss:updateOutput(input)
  self.output = input
  return self.output
end


function greyLoss:updateGradInput(input, gradOutput)
	
end


local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


function postProccess(image,origimage,params)	
	return  TransferBlack(image,origimage,params)
end

function getAverageChannel(image,channel)
	total =0
	image_width=image:size()[2]
	image_height=image:size()[3]
     	
	for i=1,image_width,1 do
		for j=1,image_height,1 do
			total=total+image[channel][i][j]
		end
	end

	return total/(image_width*image_height)
end

function transferAverageChannel (imag,channel,ratio)
	image_width=image:size()[2]
	image_height=image:size()[3]
	for i=1,image_width,1 do
		for j=1,image_height,1 do

			image[channel][i][j]=image[channel][i][j]*ratio
		end
	end
	return imag
end

function TransferBlackViaYIQ(imag,bwimage)
	image_width=imag:size()[2]
	image_height=imag:size()[3]

	for i=1,image_width,1 do
		for j=1,image_height,1 do
			imYIQ=RGBToYIQ(imag[1][i][j],imag[2][i][j],imag[3][i][j])	
			bwYIQ=RGBToYIQ(bwimage[1][i][j],bwimage[2][i][j],bwimage[3][i][j])
			
			Y=bwYIQ[1]
			I=imYIQ[2]
			Q=imYIQ[3]
			RGB=YIQtoRGB(Y,I,Q)
			imag[1][i][j]=RGB[1]
			imag[2][i][j]=RGB[2]
			imag[3][i][j]=RGB[3]
		end
	end	

return imag;
end

function campTo1or0 (Tensor3d)

	tensor_width=Tensor3d:size()[2]
	tensor_height=Tensor3d:size()[3]
	tensor_depth=Tensor3d:size()[1]
	for i=1,tensor_width,1 do
		for j=1,tensor_height,1 do
			for k=1,tensor_depth,1 do 
				if(Tensor3d[k][i][j])>1 then

					Tensor3d[k][i][j]=1
			
				end	
				if(Tensor3d[k][i][j]<0.0) then
					Tensor3d[k][i][j]=0.0			
				end
			end
		end
	end
	return Tensor3d
end
function TransferBlackViaHSV(imag,bwimage)
	imag=campTo1or0(imag)
	bwimage=campTo1or0(bwimage)

	image_width=imag:size()[2]
	image_height=imag:size()[3]

	for i=1,image_width,1 do
		for j=1,image_height,1 do
			imHSV=RGBToHSV(imag[1][i][j],imag[2][i][j],imag[3][i][j])	
			bwHSV=RGBToHSV(bwimage[1][i][j],bwimage[2][i][j],bwimage[3][i][j])

			H=bwHSV[1]
			S=imHSV[2]
			V=imHSV[3]

			RGB=HSVToRGB(H,S,V)
			
			imag[1][i][j]=RGB[1]
			imag[2][i][j]=RGB[2]
			imag[3][i][j]=RGB[3]
			for k=1,3,1 do 
				if(imag[k][i][j])>1 then

					imag[k][i][j]=1
			
				end	
				if(imag[k][i][j]<0.01) then
					imag[k][i][j]=0.0			
				end
			end
		end
	end	


	return imag
end


function TransferBlack(img,orig)

--	print(image.rgb2lab(orig:double())[3])
	img=deprocess(img:double())
	print("set:")
--
--	imgt=img
	imgo=orig:double()
	imgt=image.rgb2lab(img)
	imgo=image.rgb2lab(imgo)
	imgt[1]=imgo[1]
	img=image.lab2rgb(imgt)
	
	print(img)
	img=preprocess(img)

return img:cuda()
	
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
function HSVToRGB(value, hue, saturation )
	-- Returns the RGB equivalent of the given HSV-defined color https://gist.github.com/GigsD4X/8513963
	-- (adapted from some code found around the web)


	-- Get the hue sector
	local hue_sector = math.floor( hue / 60 );
	local hue_sector_offset = ( hue / 60 ) - hue_sector;

	local p = value * ( 1 - saturation );
	local q = value * ( 1 - saturation * hue_sector_offset );
	local t = value * ( 1 - saturation * ( 1 - hue_sector_offset ) );

	if hue_sector == 0 then
		return {value, t, p};
	elseif hue_sector == 1 then
		return {q, value, p};
	elseif hue_sector == 2 then
		return {p, value, t};
	elseif hue_sector == 3 then
		return {p, q, value};
	elseif hue_sector == 4 then
		return {t, p, value};
	elseif hue_sector == 5 then
		return {value, p, q};
	end;
end;


function RGBToYIQ(R,G,B)
--    Y=(0.299*R)+(0.587*G)+(0.114*B)
--    I=(0.596*R)-(0.274*G)-(0.322*B)
--    Q=(0.211*R)-(0.523*G)-(0.312*B)

	Y=0.299*R+0.596*G+0.211*B
	I=0.587*R-0.274*G-0.322*B
	Q=0.114*R-0.322*G+0.312*B
    return {Y,I,Q}
end
function YIQtoRGB (Y,I,Q)
    R= (1*Y)+(0.956*I)+(0.621*Q)
    G= (1*Y)-(0.272*I)-(0.647*Q)
    B= (1*Y)-(1.106*I)+(1.703*Q)
    return {R,G,B}
end
--local params = cmd:parse(arg)
--main(params)

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
cmd:option('-target_image', 'in.jpg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers-deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('label_file', 'models/imagenet1000_clsid_to_human.txt')

function Main (params)
	collectgarbage()
	net=buildNN(params)

	Timage= image.load(params.target_image, 3)
	Timage = image.scale(Timage, 224,224, 'bilinear')
	Timage=preprocess(Timage)
  	if params.gpu >= 0 then
  	  if params.backend ~= 'clnn' then
     	 Timage = Timage:cuda()
   	  else
    	 Timage = Timage:cl()
      end
	 end
	output=net:forward(Timage)
	v,i=torch.max(output,1)
	vv,ii=torch.topk(output,3,1,true)
 	file= io.open("recon.out","w")
	file:write((i[1]-1))
	file:close()

end




local cnn

function buildNN (params)


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



function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


local params = cmd:parse(arg)
return Main(params)


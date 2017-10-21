require 'torch'
require 'optim'
require 'image'
local cmd = torch.CmdLine()
cmd:option('-orig', '4093254375_062cabefb1.jpg')
cmd:option('-col', 'out_prepost_1000.png')


function main(params)
	imga = image.load(params.col, 3)
	imgb = image.load(params.orig, 3)
    print(imga:size())
	imbg=imgb:resize(imga:size()[2],imga:size()[3],3)
	r=torch.sum(torch.abs(imga-imgb))
    print(r)
	return r
end




local params = cmd:parse(arg)
return main(params)


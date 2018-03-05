for i in {0..40}
do
	echo $i
sleep 1
killall -9  /home/thijser/torch/install/bin/luajit
killall -9 /home/thijser/torch-cl/install/bin/luajit
	exc="th neural_stylerun.lua"
	contentweight="-content_weight 5"
	styleweight="-style_weight $((2 ^ $i))"
	content="-content_image in/crown.jpg"
	out="-output_image out/"$i"out.png"
	style="-style_image t/Pictures/crown/ActiOn_77.png,t/Pictures/crown/ActiOn_16.jpg,t/Pictures/crown/ActiOn_81.jpg,t/Pictures/crown/ActiOn_46.jpg,t/Pictures/crown/ActiOn_1.jpg,t/Pictures/crown/ActiOn_61.jpg,t/Pictures/crown/ActiOn_63.jpg,t/Pictures/crown/ActiOn_90.jpg,t/Pictures/crown/ActiOn_47.jpg,t/Pictures/crown/ActiOn_96.jpg,t/Pictures/crown/ActiOn_66.png
"

 	exc="$exc $content $style $styleweight $contentweight $out" 
	echo $exc
	$exc > nul

	
done

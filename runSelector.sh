echo "killing all other torch instances"
killall -9  /home/thijser/torch-cl/install/bin/luajit

cd t 

if [ "$2" -eq "0" ]; then 
	echo "using pre-existing images"
else 
#	python imsearch.py $1 $2
    echo "skipping step"
fi 
cd ..

echo $PATH
c="/home/nfs/thijsboumans/distro/install/bin/th imageSelectorroul.lua -avaible_images $(find t/Pictures -type f \( -iname \*.jpg -o -iname \*.png \) -printf '%p,' | sed 's/,$//') -colweight $3 -image_count $4 " 

echo $c
eval $c





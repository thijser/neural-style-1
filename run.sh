cd t 
rm Pictures/* -r
python imsearch.py $1 $2
cd ..


c="th neural_stylerun.lua -content_image in.jpg -style_image $(find t/Pictures -type f \( -iname \*.jpg -o -iname \*.png \) -printf '%p,' | sed 's/,$//')"
echo $c
eval $c

mkdir images -p && mkdir results -p;
rm images/content3.png -rf;
rm images/style3.png -rf;
rm results/content3_seg.pgm -rf;
rm results/style3_seg.pgm -rf;
rm results/stylization_with_auto_segmentation.png -rf;
export PYTHONPATH=$PYTHONPATH:segmentation
cd images;
# axel -n 1 https://pre00.deviantart.net/f1a6/th/pre/i/2010/019/0/e/country_road_hdr_by_mirre89.jpg --output=content3.png;
# axel -n 1 https://nerdist.com/wp-content/uploads/2017/11/Stranger_Things_S2_news_Images_V03-1024x481.jpg --output=style3.png;
# convert -resize 100% content3.png content3.png;
# convert -resize 100% style3.png style3.png;
convert -resize 50% ../content/4_13_2021_vid1_frame_00001.jpg content3.png
convert -resize 50% ../styles/vlcsnap-2022-09-02-16h26m19s408.jpg style3.png
cd ..;
python demo_with_ade20k_ssn.py;
convert -resize 200% results/example3.png results/example3-upscaled.png
cp images/content3.png results/content3.png
cp images/style3.png results/style3.png

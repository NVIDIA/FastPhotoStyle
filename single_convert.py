from photo_wct import PhotoWCT
import torch
import process_stylization
import glob
import os

styles_dir = "styles"
content_dir = "content"
output_dir = "results/result"
percent_resized_style = 50
percent_resized_content = 50
fast = False
use_cuda = True

style_files = glob.glob(f"{styles_dir}/*")
content_files = glob.glob(f"{content_dir}/*")
print(f"Using style files: {style_files}")

p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load('./PhotoWCTModels/photo_wct.pth'))

style_file = style_files[0]
content_file = content_files[0]

if fast:
    from photo_gif import GIFSmoothing
    p_pro = GIFSmoothing(r=35, eps=0.001)
else:
    from photo_smooth import Propagator
    p_pro = Propagator()
if use_cuda:
    p_wct.cuda(0)

os.system(f"convert -resize {percent_resized_style}% {style_file} ./images/stylesingle1.png")
os.system(f"convert -resize {percent_resized_content}% {content_file} ./images/contentsingle1.png")

output_file = f"{output_dir}_styled.png"

process_stylization.stylization(
    stylization_module = p_wct,
    smoothing_module = p_pro,
    content_image_path = "./images/contentsingle1.png",
    style_image_path = "./images/stylesingle1.png",
    content_seg_path = [],
    style_seg_path = [],
    output_image_path = output_file,
    cuda = use_cuda,
    save_intermediate = False,
    no_post = False
)

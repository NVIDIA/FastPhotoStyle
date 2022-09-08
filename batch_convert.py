from photo_wct import PhotoWCT
import torch
import process_stylization
import glob
import os

styles_dir = "styles"
content_dir = "content"
output_dir = "results"
percent_resized_style = 50
percent_resized_content = 25
fast = False
use_cuda = True

style_files = glob.glob(f"{styles_dir}/*.jpg")
content_files = glob.glob(f"{content_dir}/*.jpg")
print(f"Using style files: {style_files}")

p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load('./PhotoWCTModels/photo_wct.pth'))

if fast:
    from photo_gif import GIFSmoothing
    p_pro = GIFSmoothing(r=35, eps=0.001)
else:
    from photo_smooth import Propagator
    p_pro = Propagator()
if use_cuda:
    p_wct.cuda(0)

style_num = 0
for style_file in style_files:
    print(f"Style file {style_file} converting")
    # os.system(f"convert -resize {percent_resized_style}% {style_file} ./images/style1.png")
    for content_file in content_files:
        print(f"Content file file {content_file} converting")
        # os.system(f"convert -resize {percent_resized_content}% {content_file} ./images/content1.png")

        output_file = f"{output_dir}/{content_file.split('/')[-1].split('.')[0]}-{style_file.split('/')[-1].split('.')[0]}.png"
        print(
            f"Stylizing {output_file} from style {style_file} and content {content_file}"
        )
        process_stylization.stylization(
            stylization_module = p_wct,
            smoothing_module = p_pro,
            content_image_path = content_file,
            style_image_path = style_file,
            # content_image_path = "./images/content1.png",
            # style_image_path = "./images/style1.png",
            content_seg_path = f"{content_file.split('.')[0]}.json",
            style_seg_path = f"{style_file.split('.')[0]}.json",
            output_image_path = output_file,
            cuda = use_cuda,
            save_intermediate = False,
            no_post = False
        )

        style_num += 1

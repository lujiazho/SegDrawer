from fastapi import FastAPI, status, File, Form, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

import os
import cv2
import time
import torch
import zipfile
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image
from base64 import b64encode, b64decode

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

sam_checkpoint = "sam_vit_l_0b3195.pth" # "sam_vit_l_0b3195.pth" or "sam_vit_h_4b8939.pth"
model_type = "vit_l" # "vit_l" or "vit_h"
# device = "cuda" # "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('CUDA not available. Please connect to a GPU instance if possible.')
    device = 'cpu'

print("Loading model")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
print("Finishing loading")
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a palette for video segmentation
import random
palette = [random.randint(0, 255) for _ in range(256*3)]

input_point = []
input_label = []
masks = []

segmented_mask = []
interactive_mask = []
mask_input = None

GLOBAL_IMAGE = None
GLOBAL_MASK = None
GLOBAL_ZIPBUFFER = None

@app.post("/image")
async def process_images(
    image: UploadFile = File(...)
):
    global segmented_mask, interactive_mask
    global GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_ZIPBUFFER
    import os
    cache_path = "./embedding_cached.pt"

    segmented_mask = []
    interactive_mask = []

    # Read the image and mask data as bytes
    image_data = await image.read()

    image_data = BytesIO(image_data)
    img = np.array(Image.open(image_data))
    print("get image", img.shape)
    GLOBAL_IMAGE = img[:,:,:-1]
    GLOBAL_MASK = None
    GLOBAL_ZIPBUFFER = None
    # produce an image embedding by calling SamPredictor.set_image

    # if os.path.exists(cache_path):
    #     predictor.load_embedding(cache_path)
    #     print(predictor.is_image_set)
    # else:
    #     predictor.set_image(GLOBAL_IMAGE)
    #     predictor.save_embedding(cache_path)
    #     print("finish setting image")
    predictor.set_image(GLOBAL_IMAGE)

    # Return a JSON response
    return JSONResponse(
        content={
            "message": "Images received successfully",
        },
        status_code=200,
    )

from XMem import XMem, InferenceCore, image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

torch.set_grad_enabled(False)

def seg_propagation(video_name, mask_name):
    # default configuration
    config = {
        'top_k': 30,
        'mem_every': 5,
        'deep_update_every': -1,
        'enable_long_term': True,
        'enable_long_term_count_usage': True,
        'num_prototypes': 128,
        'min_mid_term_frames': 5,
        'max_mid_term_frames': 10,
        'max_long_term_elements': 10000,
    }

    network = XMem(config, './XMem/saves/XMem.pth').eval().to(device)

    im = Image.open(mask_name).convert('L')
    im.putpalette(palette)
    mask = np.array(im)
    acc = 0
    for i in range(256):
        if np.sum(mask==i) == 0:
            acc += 1
            mask[mask==i] -= acc-1
        else:
            mask[mask==i] -= acc
    print(np.unique(mask))
    num_objects = len(np.unique(mask)) - 1

    st = time.time()
    # torch.cuda.empty_cache()

    processor = InferenceCore(network, config=config)
    processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
    cap = cv2.VideoCapture(video_name)

    # You can change these two numbers
    frames_to_propagate = 1500
    visualize_every = 1

    current_frame_index = 0

    masked_video = []

    with torch.cuda.amp.autocast(enabled=True):
        while (cap.isOpened()):
            # load frame-by-frame
            _, frame = cap.read()
            if frame is None or current_frame_index > frames_to_propagate:
                break

            # convert numpy array to pytorch tensor format
            frame_torch, _ = image_to_torch(frame, device=device)
            if current_frame_index == 0:
                # initialize with the mask
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
                # the background mask is not fed into the model
                prediction = processor.step(frame_torch, mask_torch[1:])
            else:
                # propagate only
                prediction = processor.step(frame_torch)

            # argmax, convert to numpy
            prediction = torch_prob_to_numpy_mask(prediction)

            if current_frame_index % visualize_every == 0:
                visualization = overlay_davis(frame[...,::-1], prediction)
                masked_video.append(visualization)

            current_frame_index += 1
    ed = time.time()

    print(f"Propagation time: {ed-st} s")

    from moviepy.editor import ImageSequenceClip, AudioFileClip
    
    audio = AudioFileClip(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_dir = f'./XMem/output/{video_name.split("/")[-1].split(".")[0]}.mp4'
    if not os.path.exists('./XMem/output/'):
        os.mkdir('./XMem/output/')
    clip = ImageSequenceClip(sequence=masked_video, fps=fps)
    # Set the audio of the new video to be the audio from the original video
    clip = clip.set_audio(audio)
    clip.write_videofile(output_dir, fps=fps, audio=True)

    return output_dir

VIDEO_NAME = ""

@app.post("/video")
async def obtain_videos(
    video: UploadFile = File(...)
):
    # Read the video data as bytes
    video_data = await video.read()

    # Write the video data to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_data)
    temp_file.close()

    print(temp_file.name)

    global VIDEO_NAME
    if VIDEO_NAME != "":
        os.unlink(VIDEO_NAME)
    VIDEO_NAME = temp_file.name

    return JSONResponse(
        content={
            "message": "upload video successfully",
        },
        status_code=200,
    )

@app.post("/ini_seg")
async def process_videos(
    ini_seg: UploadFile = File(...)
):
    global VIDEO_NAME

    ini_seg_data = await ini_seg.read()

    tmp_seg_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_seg_file.write(ini_seg_data)
    tmp_seg_file.close()

    print(tmp_seg_file.name)

    if VIDEO_NAME == "":
        raise HTTPException(status_code=204, detail="No content")
    
    res_path = seg_propagation(VIDEO_NAME, tmp_seg_file.name)

    os.unlink(tmp_seg_file.name)
    # os.unlink(VIDEO_NAME)
    # VIDEO_NAME = ""

    # Return a FileResponse with the processed video path
    return FileResponse(
        res_path,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="{VIDEO_NAME.split("/")[-1].split(".")[0]}.mp4"',
        },
    )

@app.post("/undo")
async def undo_mask():
    global segmented_mask
    segmented_mask.pop()

    return JSONResponse(
        content={
            "message": "Clear successfully",
        },
        status_code=200,
    )


from fastapi import Request


@app.post("/click")
async def click_images(
    request: Request,
):  
    global mask_input, interactive_mask

    form_data = await request.form()
    type_list = [int(i) for i in form_data.get("type").split(',')]
    click_list = [int(i) for i in form_data.get("click_list").split(',')]
    # x_list = [int(i) for i in form_data.get("x").split(',')]
    # y_list = [int(i) for i in form_data.get("y").split(',')]

    point_coords = np.array(click_list, np.float32).reshape(-1, 2)
    point_labels = np.array(type_list).reshape(-1)

    print(point_coords)
    print(point_labels)

    if (len(point_coords) == 1):
        mask_input = None

    masks_, scores_, logits_ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=mask_input,
        multimask_output=True,
    )

    best_idx = np.argmax(scores_)
    res = masks_[best_idx]
    mask_input = logits_[best_idx][None, :, :]

    len_prompt = len(point_labels)
    len_mask = len(interactive_mask)
    if len_mask == 0 or len_mask < len_prompt:
        interactive_mask.append(res)
    else:
        interactive_mask[len_prompt-1] = res

    # Return a JSON response
    res = Image.fromarray(res)
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(res),
            "message": "Images processed successfully"
        },
        status_code=200,
    )

@app.post("/finish_click")
async def finish_interactive_click(
    mask_idx: int = Form(...),
):
    global segmented_mask, interactive_mask

    segmented_mask.append(interactive_mask[mask_idx])
    interactive_mask = list()

    return JSONResponse(
        content={
            "message": "Finish successfully",
        },
        status_code=200,
    )
    

@app.post("/rect")
async def rect_images(
    start_x: int = Form(...), # horizontal
    start_y: int = Form(...), # vertical
    end_x: int = Form(...), # horizontal
    end_y: int = Form(...)  # vertical
):
    masks_, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array([[start_x, start_y, end_x, end_y]]),
        multimask_output=False
    )
    
    res = Image.fromarray(masks_[0])
    # res.save("res.png")
    print(masks_[0].shape)
    # res.save("res.png")

    # Return a JSON response
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(res),
            "message": "Images processed successfully"
        },
        status_code=200,
    )

@app.post("/everything")
async def seg_everything():
    """
        segmentation : the mask
        area : the area of the mask in pixels
        bbox : the boundary box of the mask in XYWH format
        predicted_iou : the model's own prediction for the quality of the mask
        point_coords : the sampled input point that generated this mask
        stability_score : an additional measure of mask quality
        crop_box : the crop of the image used to generate this mask in XYWH format
    """
    global GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_ZIPBUFFER
    if type(GLOBAL_MASK) != type(None):
        return JSONResponse(
            content={
                "masks": pil_image_to_base64(GLOBAL_MASK),
                "zipfile": b64encode(GLOBAL_ZIPBUFFER.getvalue()).decode("utf-8"),
                "message": "Images processed successfully"
            },
            status_code=200,
        )


    masks = mask_generator.generate(GLOBAL_IMAGE)
    assert len(masks) > 0, "No masks found"

    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    print(len(sorted_anns))

    # Create a new image with the same size as the original image
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]), dtype=np.uint8)
    for idx, ann in enumerate(sorted_anns, 0):
        img[ann['segmentation']] = idx % 255 + 1 # color can only be in range [1, 255]
    
    res = Image.fromarray(img)
    GLOBAL_MASK = res

    # Save the original image, mask, and cropped segments to a zip file in memory
    zip_buffer = BytesIO()
    PIL_GLOBAL_IMAGE = Image.fromarray(GLOBAL_IMAGE)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Cut out the segmented regions as smaller squares
        for idx, ann in enumerate(sorted_anns, 0):
            left, top, right, bottom = ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]
            cropped = PIL_GLOBAL_IMAGE.crop((left, top, right, bottom))

            # Create a transparent image with the same size as the cropped image
            transparent = Image.new("RGBA", cropped.size, (0, 0, 0, 0))

            # Create a mask from the segmentation data and crop it
            mask = Image.fromarray(ann["segmentation"].astype(np.uint8) * 255)
            mask_cropped = mask.crop((left, top, right, bottom))

            # Combine the cropped image with the transparent image using the mask
            result = Image.composite(cropped.convert("RGBA"), transparent, mask_cropped)

            # Save the result to the zip file
            result_bytes = BytesIO()
            result.save(result_bytes, format="PNG")
            result_bytes.seek(0)
            zip_file.writestr(f"seg_{idx}.png", result_bytes.read())

    # move the file pointer to the beginning of the file so we can read whole file
    zip_buffer.seek(0)
    GLOBAL_ZIPBUFFER = zip_buffer

    # Return a JSON response
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(GLOBAL_MASK),
            "zipfile": b64encode(GLOBAL_ZIPBUFFER.getvalue()).decode("utf-8"),
            "message": "Images processed successfully"
        },
        status_code=200,
    )

@app.get("/assets/{path}/{file_name}", response_class=FileResponse)
async def read_assets(path, file_name):
    return f"assets/{path}/{file_name}"

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return read_content('segDrawer.html')

import uvicorn
uvicorn.run(app, host="0.0.0.0", port=7860)

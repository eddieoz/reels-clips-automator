import random
import sys
import textwrap
import time
import numpy as np
from pytube import YouTube
import cv2
import subprocess
import openai
import json
from datetime import datetime
import os
from os import path
import shutil

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["GGML_CUDA_NO_PINNED"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

scale = 4
target_face_size = 345
        
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

## Added Real-ESRGAN to utils
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
# restorer
upsampler = RealESRGANer(
    scale=scale,
    model_path="weights/realesr-general-x4v3.pth",
    dni_weight=1,
    model=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=not True,
    gpu_id=0)

from gfpgan import GFPGANer
face_enhancer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
    upscale=scale,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=upsampler)

import argparse

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Download video
def download_video(url, filename):
    yt = YouTube(url)
    video = yt.streams.filter(file_extension='mp4').get_highest_resolution()

    # Download the video
    video.download(filename=filename, output_path='tmp/')


#Segment Video function
def generate_segments(response):
  
  for i, segment in enumerate(response):
        print(i, segment)

        start_time = segment.get("start_time", 0).split('.')[0]
        end_time = segment.get("end_time", 0).split('.')[0]
        title = segment.get("title", 0)

        pt = datetime.strptime(start_time,'%H:%M:%S')
        start_time = pt.second + pt.minute*60 + pt.hour*3600

        pt = datetime.strptime(end_time,'%H:%M:%S')
        end_time = pt.second + pt.minute*60 + pt.hour*3600

        if end_time - start_time < 50:
            end_time += (50 - (end_time - start_time))

        output_file = f"{str(title).replace(' ','_')}{str(i).zfill(3)}.mp4"
        command = f"ffmpeg -y -hwaccel cuda -i tmp/input_video.mp4 -vf scale='1920:1080' -qscale:v '3' -b:v 6000k -ss {start_time} -to {end_time} tmp/{str(output_file)}"
        subprocess.call(command, shell=True)

def find_smile(frame, output_file):
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the smile
    smile = smile_cascade.detectMultiScale(img_gray, scaleFactor=1.8, minNeighbors=25, flags=cv2.CASCADE_SCALE_IMAGE)
    # print(f"smile: {smile}")

    if len(smile) > 0:
        for x, y, w, h in smile:
            print("[INFO] Smile found. Saving locally.")
            # resize image
            # resizeimg = cv2.resize(frame, (400, 400), interpolation = cv2.INTER_CUBIC)
            # cv2.imwrite(f"./tmp/smiling-{output_file}.png", frame)
            return True, frame
    else:
        return False, ''

def generate_thumbnail(face, text, thumb_dir):
    from PIL import Image, ImageFont, ImageDraw 

    # Create a blank image 
    W, H = (1080, 1920)
    newthumb = Image.new('RGBA', size=(W, H), color=(0, 0, 0, 0))

    bg_dir = "./backgrounds-reels/"
    background = Image.open(bg_dir + random.choice(os.listdir(bg_dir))).convert("RGBA")
    background = background.resize((W, H))
    # Paste background
    newthumb.paste(background, (0,0))

    # resizing image
    img_w, img_h = (500, 500)
    face = cv2.resize(face, (img_w, img_h))

    # Define the border
    top = bottom = left = right = 20
    color = [255, 255, 255] 
    bordered_face = cv2.copyMakeBorder(face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    bordered_face = cv2.cvtColor(bordered_face, cv2.COLOR_BGR2RGB)

    bordered_image_pil = Image.fromarray(bordered_face)
 
    # add title
    title_font = ImageFont.truetype(font='fonts/Arial_Black.ttf', size=100)
    title_text = textwrap.fill(text, width=15)
    image_editable = ImageDraw.Draw(newthumb)

    # get sizes
    text_w, text_h = image_editable.textsize(title_text, font=title_font)
    print(f"1) text w: {text_w}, text h: {text_h}")

    # adjust text size
    if (text_h > 400):
        title_font = ImageFont.truetype(font='fonts/Arial_Black.ttf', size=85)
        title_text = textwrap.fill(text, width=20)
        image_editable = ImageDraw.Draw(newthumb)

        # get sizes
        text_w, text_h = image_editable.textsize(title_text, font=title_font)
        print(f"2) text w: {text_w}, text h: {text_h}")
    
    if (text_h > 400):
        title_font = ImageFont.truetype(font='fonts/Arial_Black.ttf', size=70)
        title_text = textwrap.fill(text, width=25)
        image_editable = ImageDraw.Draw(newthumb)

        # get sizes
        text_w, text_h = image_editable.textsize(title_text, font=title_font)
        print(f"3) text w: {text_w}, text h: {text_h}")

    face_text_h = img_h + 80 + text_h

    # New positions
    new_face_h = int((H-face_text_h)/2)
    new_face_w = int((W-img_w-left-right)/2)
    new_text_h = (new_face_h + face_text_h - text_h)
    new_text_w = int((W-text_w)/2)

    # Paste face, write text
    newthumb.paste(bordered_image_pil, box=(new_face_w, new_face_h))
    image_editable.text((new_text_w, new_text_h), title_text, (255, 255, 255), align='center', font=title_font, stroke_width=10 ,stroke_fill=(0,0,0))

    # Save thumbnail
    newthumb.save(f"./{thumb_dir}/thumb-{text}.png","PNG")



def generate_short(input_file, output_file, title = '', upscale = False, enhance = False, thumb = False, thumb_dir = ''):
    try:

        # Interval to switch faces (in frames) (ex. 150 frames = 5 seconds, on a 30fps video)
        switch_interval = 150

        # Frame counter
        frame_count = 0

        # Index of the currently displayed face
        current_face_index = 0
        
        # Constants for cropping    
        CROP_RATIO_BIG = 1 # Adjust the ratio to control how much of the image (around face) is visible in the cropped video
        CROP_RATIO_SMALL = 0.5
        VERTICAL_RATIO = 9 / 16  # Aspect ratio for the vertical video

        # Load pre-trained face detector from OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Open video file
        cap = cv2.VideoCapture(f"tmp/{input_file}")

        # Get the frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Image frame_height {frame_height}, frame_width {frame_width}")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"tmp/{output_file}", fourcc, 30, (1080, 1920))  # Adjust resolution for 9:16 aspect ratio
        face_positions = []
        smile_found = False
        # success = False
        while(cap.isOpened()):
            # Read frame from video
            ret, frame = cap.read()

            if ret == True:
                
                # If we don't have any face positions, detect the faces
                # Switch faces if it's time to do so
                if frame_count % switch_interval == 0:
                    # Convert color style from BGR to RGB
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Perform face detection
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100))
           
                    if len(faces) > 0:
                        # Initialize trackers and variable to hold face positions
                        trackers = cv2.legacy.MultiTracker_create()
                        face_positions.clear()
                        
                        for (x, y, w, h) in faces:
                            face_positions.append((x, y, w, h))
                            tracker = cv2.legacy.TrackerKCF_create()
                            tracker.init(frame, (x, y, w, h))
                            trackers.add(tracker, frame, (x, y, w, h))            

                        # Update trackers and get updated positions
                        try:
                            success, boxes = trackers.update(frame)
                        except Exception as e:
                            print (f"Erro update trackers: {e}")

                    # Switch faces if it's time to do so
                    current_face_index = (current_face_index + 1) % len(face_positions)
                    x, y, w, h = [int(v) for v in boxes[current_face_index]]

                    print (f"Frame: {frame_count}, Current Face index {current_face_index} heigth {h} width {w} total faces {len(face_positions)}, Upscale: {upscale}, Enhance: {enhance}, Thumb: {thumb}")

                    face_center = (x + w//2, y + h//2)

                    if w * 16 > h * 9:
                        w_916 = w
                        h_916 = int(w * 16 / 9)
                    else:
                        h_916 = h
                        w_916 = int(h * 9 / 16)

                    #Calculate the target width and height for cropping (vertical format)
                    if max(h, w) < target_face_size:
                        target_height = int(frame_height * CROP_RATIO_SMALL)
                        target_width = int(target_height * VERTICAL_RATIO)
                    else:
                        target_height = int(frame_height * CROP_RATIO_BIG)
                        target_width = int(target_height * VERTICAL_RATIO)

                # Calculate the top-left corner of the 9:16 rectangle
                x_916 = (face_center[0] - w_916 // 2)
                y_916 = (face_center[1] - h_916 // 2)

                crop_x = max(0, x_916 + (w_916 - target_width) // 2)  # Adjust the crop region to center the face
                crop_y = max(0, y_916 + (h_916 - target_height) // 2)
                crop_x2 = min(crop_x + target_width, frame_width)
                crop_y2 = min(crop_y + target_height, frame_height)


                # Crop the frame to the face region
                crop_img = frame[crop_y:crop_y2, crop_x:crop_x2]

                # Upscale the cropped image if the face is too small
                if upscale or enhance:
                    if max(h, w) < target_face_size:
                        if upscale:
                            crop_img, _ = upsampler.enhance(crop_img, outscale=scale)
                        if enhance:
                            _, _, crop_img = face_enhancer.enhance(crop_img, has_aligned=False, only_center_face=False, paste_back=True)

                resized = cv2.resize(crop_img, (1080, 1920), interpolation = cv2.INTER_AREA)

                if thumb and not smile_found:
                    smile_found, smile_frame = find_smile(frame[y:y+h, x:x+w], output_file)
                
                out.write(resized)

                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Extract audio from original video
        command = f"ffmpeg -y -hwaccel cuda -i tmp/{input_file} -vn -acodec copy tmp/output-audio.aac"
        subprocess.call(command, shell=True)

        # Merge audio and processed video
        command = f"ffmpeg -y -hwaccel cuda -i tmp/{output_file} -i tmp/output-audio.aac -c copy tmp/final-{output_file}"
        subprocess.call(command, shell=True)

        if thumb and smile_found:
            generate_thumbnail(smile_frame, title, thumb_dir)

    except Exception as e:
        print(f"Error during video cropping: {str(e)}")

def generate_viral(transcript): # Inspiredby https://github.com/NisaarAgharia/AI-Shorts-Creator 

    json_template = '''
        { "segments" :
            [
                {
                    "start_time": 00:00:00.00, 
                    "end_time": 00:00:00.00,
                    "title": "Title of the reels",
                    "duration":00.00,
                },    
            ]
        }
    '''

    title_language = 'Portuguese'
    prompt = f"Given the following video transcript, analyze each part for potential virality and identify 3 most viral segments of maximum 59 seconds each segment, from the transcript. Each segment should have a minimum of 50 seconds and maximum of 59 seconds in duration. The provided transcript is as follows: {transcript}. Based on your analysis, return a JSON document containing the timestamps (start and end), the engaging but objective title for the viral part in {title_language} language, and its duration. The JSON document should follow this format: {json_template}. Please replace the placeholder values with the actual results from your analysis."
    system = f"You are a Viral Segment Identifier, an AI system that analyzes a video's transcript and predict which segments might go viral on social media platforms. You use factors such as emotional impact, humor, unexpected content, and relevance to current trends to make your predictions. You return a structured JSON document detailing the start and end times, the title in {title_language} language, and the duration of the potential viral segments."
    messages = [
        {"role": "system", "content" : system},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=512,
        n=1,
        stop=None
    )
    return response.choices[0]['message']

def generate_subtitle(input_file, output_folder, results_folder):
    command = f"auto_subtitle tmp/{input_file} -o {results_folder}/{output_folder} --model medium"
    print (command)
    subprocess.call(command, shell=True)

def generate_transcript(input_file):
    command = f"auto_subtitle tmp/{input_file} --srt_only True --output_srt True -o tmp/ --model medium"
    subprocess.call(command, shell=True)
    
    # Read the contents of the input file
    try:
        with open(f"tmp/{os.path.basename(input_file).split('.')[0]}.srt", 'r', encoding='utf-8') as file:
            transcript = file.read()
    except IOError:
        print("Error: Failed to read the input file.")
        sys.exit(1)
    return transcript

def __main__():

    # Check command line argument    
    parser = argparse.ArgumentParser(description='Create 3 reels or tiktoks from Youtube video')
    parser.add_argument('-v', '--video_id', required=False, help='Youtube video id. Ex: Cuptv7-A4p0 in https://www.youtube.com/watch?v=Cuptv7-A4p0')
    parser.add_argument('-f', '--file', required=False, help='Video file to be used')
    parser.add_argument('-u', '--upscale', action='store_true', default=False, required=False, help='Upscale small faces')
    parser.add_argument('-e', '--enhance', action='store_true', default=False, required=False, help='Upscale and enhance small faces')
    parser.add_argument('-t', '--thumb', action='store_true', default=False, required=False, help='Create thumbnail')
    args = parser.parse_args()
    print (args)
    
    if not args.video_id and not args.file: 
        print('Needed at least one argument. <command> --help for help')
        sys.exit(1)

    if args.upscale and args.enhance:
        print('You can use --upcale or --enhance. Not both')
        sys.exit(1)

    if args.upscale or args.enhance:
        if not os.path.exists("weights") or not os.path.exists("gfpgan") or not os.path.exists("realesrgan"):
            args.upscale = args.enhance = False
            print('Upscale and Enhance require utils/Real-ESRGAN installed and working')
            sys.exit(1)

    if args.thumb:
        if not os.path.exists("backgrounds-reels"):
            args.thumb = False
            print('Thumbnail require utils/thumbnail_generator installed and working')
            sys.exit(1)
    
    if args.video_id and args.file:
        print('use --video_id or --file')
        sys.exit(1)

    # Create temp folder
    try: 
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")
        os.mkdir('tmp') 
    except OSError as error: 
        print(error)

    filename = 'input_video.mp4'
    if args.video_id:
        video_id=args.video_id
        url = 'https://www.youtube.com/watch?v='+video_id  # Replace with your video's URL
        # Download video
        download_video(url,filename)
    
    if args.file:
        video_id = os.path.basename(args.file).split('.')[0]
        print(video_id)
        if (path.exists(args.file) == True):
            command = f"cp {args.file} tmp/input_video.mp4"
            subprocess.call(command, shell=True)
        else:
            print(f"File {args.file} does not exist")
            sys.exit(1)

    output_folder = 'results'
    
    # Create outputs folder
    try: 
        os.mkdir(f"{output_folder}") 
    except OSError as error: 
        print(error)
    try: 
        os.mkdir(f"{output_folder}/{video_id}") 
    except OSError as error: 
        print(error)

    # Verifies if output_file exists, or create it. If exists, it doesn't call OpenAI APIs
    output_file = f"{output_folder}/{video_id}/content.txt"
    transcript_file = f"{output_folder}/{video_id}/transcript.txt"
    if (path.exists(output_file) == False):
        # generate transcriptions
        transcript = generate_transcript(filename)
        print (transcript)
        
        viral_segments = generate_viral(transcript)
        content = viral_segments["content"]
        try:
            with open(transcript_file, 'w', encoding='utf-8') as file:
                file.write(transcript)
        except IOError:
            print("Error: Failed to write the output file.")
            sys.exit(1)
        print("Full transcription written to ", output_file)

        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(content)
        except IOError:
            print("Error: Failed to write the output file.")
            sys.exit(1)
        print("Segments written to ", output_file)
    else:
        # Read the contents of the input file
        try:
            with open(output_file, 'r', encoding='utf-8') as file:
                content = file.read()
        except IOError:
            print("Error: Failed to read the input file.")
            sys.exit(1)

    parsed_content = json.loads(content)
    generate_segments(parsed_content['segments'])
    
    # Loop through each segment
    for i, segment in enumerate(parsed_content['segments']):  # Replace xx with the actual number of segments
        thumbs_dir = f"{output_folder}/{video_id}/"
        input_file = f"{str(segment['title']).replace(' ','_')}{str(i).zfill(3)}.mp4"
        output_file = f"{str(segment['title']).replace(' ','_')}_cropped{str(i).zfill(3)}.mp4"
        generate_short(input_file, output_file, str(segment['title']), args.upscale, args.enhance, args.thumb, thumbs_dir)
        generate_subtitle(f"final-{output_file}", video_id, output_folder)

__main__()

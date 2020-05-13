import cv2
import os
import glob
from pytube import YouTube
#pip install fuzzywuzzy
from fuzzywuzzy import fuzz
import pytesseract

curr_url = "https://www.youtube.com/watch?v=VO5vKowfMOQ&t=7s"

# download from here https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_images_from_video(video, target_path, frequency=15, name="file", max_images=20, silent=False):  
    vidcap = cv2.VideoCapture(video)
    frame_count = 0
    time_sec = 0
    num_images = 0
    folder = target_path
    label = 0
    success = True
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    list = []
    success, image = vidcap.read()
    while success and num_images < max_images:
        num_images += 1
        label += 1
        file_name = name + "_" + str(num_images) + ".jpg"
        list.append((file_name,time_sec))
        path = os.path.join(folder, file_name)
        print(path)
        cv2.imwrite(path, image)
        if cv2.imread(path) is None:
            os.remove(path)
        else:
            if not silent:
                print(f'Image successfully written at {path}')
        frame_count += frequency * fps #skips secends
        time_sec += frequency
        vidcap.set(1, frame_count)
        success, image = vidcap.read()
    return (list);

def extract_time_titles_changed(listi,change_threshold):
    wantedTimes=[]
    index = 0
    (file_name,time) = listi[index]
    img = cv2.imread(file_name)
    last_text = pytesseract.image_to_string(img)
    index += 1
    while index < len(listi):
        (file_name,time) = listi[index]
        img = cv2.imread(file_name)
        text = pytesseract.image_to_string(img)
        if text:
            # compare similarity between the last text and the new one
            if fuzz.ratio(text.lower(),last_text.lower()) > change_threshold:
                wantedTimes.append(time)
            
        index+=1
        last_text = text
    return wantedTimes

def get_time_titles_changed_with_downlode(video_url, target_path, wanted_frequency, fileName ):
    print(video_url)
    youtube = YouTube(video_url)
    # download youtube video
    
    youtube.streams.first().download(target_path,filename=fileName)
    arr = []
    #videoPath = "C:/Users/Sarit/Desktop/final_proj/The Apriori algorithm.mp4"
    wanted_file_path=target_path+'/'+fileName+'.mp4'
    print(wanted_file_path)
    arr = extract_images_from_video(wanted_file_path,target_path, frequency=wanted_frequency, name="The Apriori algorithm.mp4", max_images=1500, silent=False)
    print(extract_time_titles_changed(arr))

def get_time_titles_changed(video_path, wanted_frequency, change_threshold):
    target_path = os.getcwd();
    print(target_path)
    video_name=video_path.rsplit('/',1)[1] # take from right the first / to get the name
    print(video_name)

    images = extract_images_from_video(video_path, target_path, frequency = wanted_frequency, name=video_name, max_images=1500, silent=True)
    return extract_time_titles_changed(images, change_threshold)








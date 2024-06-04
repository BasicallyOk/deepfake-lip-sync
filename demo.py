import subprocess
import tensorflow as tf
import os
import cv2
import numpy as np

from models.decoder import image_decoder
from models.encoder import *
from models.gan import *
import utils.audio as audio

def crop_audio_window(spectrogram, start_frame_num):
    """
    Get the audio spectrogram window corresponding to the frame.
    """
    # I did some testing, there are 8 samples per second
    start_idx = int(8 * (start_frame_num // hp.fps))
    
    end_idx = start_idx + 4 # half a second to provide some padding
    if (spectrogram.shape[0] < end_idx):
        return spectrogram[-4:, :]

    return spectrogram[start_idx : end_idx, :]

FRONT_FACE_DETECTOR = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml'))
RESIZE_SIZE = (64, 64)  # Resize size for the cropped face

# For profile picture detection (including side faces... We might need it later)...
# profile_face_detector = cv2.CascadeClassifier("cascade-files/haarcascade_profileface.xml")
if FRONT_FACE_DETECTOR.empty():
    print("Unable to open the haarcascade mouth detection xml file...")
    exit(1)

# maybe only load the generator?
discriminator = Discriminator(identity_encoder_disc, audio_encoder_disc)
generator = Generator(masked_id_encoder, identity_encoder, audio_encoder)
gan = DeepfakeGAN(generator, discriminator)
gan.load_weights('') # TODO model name

video_path = '' # set to an integer for live webcam footage (need to test live generation)
audio_path = '' # live audio generation?

# Extract audio from videos
ffmpeg = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
subprocess.call(ffmpeg.format(video_path, audio_path), shell=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) # for synchronizing with the audio
counter = 0

# video writer
out = cv2.VideoWriter('output.avi', -1, fps, (64, 64))

# calculate the audio spectrogram for the entire video
wav = audio.load_wav(audio_path, hp.sample_rate)
spectrogram = audio.melspectrogram(wav)

# define reference image
reference_image = None

# Check if the video can be turned into a stream successfully.
# If not, probably check to make sure tche destination is correct.
if cap.isOpened() is False:
    print("Error opening video stream or file")
    exit(2)

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if ret is True:
        faces = FRONT_FACE_DETECTOR.detectMultiScale(frame_bgr, minNeighbors=6,
                                                 minSize=(125, 125), scaleFactor=1.15)

        if faces is not None:
            for (x, y, w, h) in faces:
                # Cropping only the faces. Please read documentation for it....
                frame_bgr = cv2.rectangle(img=frame_bgr, pt1=(x - 1, y - 1), pt2=(x + w, y + h),
                                            color=(0, 255, 0), thickness=1)
                if not reference_image:
                    reference_image = frame_bgr

                cropped = frame_bgr[y:y + h, x:x + w]
                cropped = cv2.resize(cropped, RESIZE_SIZE)

                # inference
                audio_window = crop_audio_window(spectrogram, counter)
                masked_image = frame_bgr[:32, :, :]

                generated_image = gan(masked_image, reference_image, audio_window)
                
                out.write(generated_image)
                break # only get the first face found.

        else:
            # if face not detected, show a black screen
            out.write(np.zeros((64, 64,3), np.uint8))
            
        # Wait for 25 miliseconds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        counter += 1 # determine audio clip
    else:
        break

# combine new video and original audio
subprocess.call("ffmpeg -i {} -i {} -c copy {}".format("./output.avi", audio_path, "./output.avi"), shell=True)
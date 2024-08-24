# # User Interface

import streamlit as st
import os
import imageio
import tensorflow as tf
import numpy as np
from utils import load_data, num_to_char
from modelutil import load_model
from difflib import SequenceMatcher
from googletrans import Translator, LANGUAGES  # Importing the translation package
from gtts import gTTS  # Importing the text-to-speech package
import tempfile  # For creating temporary files

st.set_page_config(layout='wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App')

# Generating a list of options or videos
options = os.listdir(os.path.join('data/s1'))
selected_video = st.selectbox('Choose video', options)

col1, col2 = st.columns(2)

if options:
    file_name = os.path.splitext(selected_video)[0]
    video_file_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_file_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')

    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        if not os.path.exists(video_file_path):
            st.error(f"Error: Video file {video_file_path} not found.")
        else:
            output_video_path = 'test_video.mp4'
            os.system(f'ffmpeg -i {video_file_path} -vcodec libx264 {output_video_path} -y')

            # Rendering inside of the app
            if os.path.exists(output_video_path):
                with open(output_video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error("Error: Failed to convert video.")

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        if not os.path.exists(video_file_path):
            st.error(f"Error: Video file {video_file_path} not found.")
        else:
            video, annotations = load_data(tf.convert_to_tensor(video_file_path))

            # Displaying the number of frames extracted
            st.write(f"Number of frames extracted: {len(video)}")
            if len(video) == 0:
                st.error("Error: The video is empty. No frames to save.")
            else:
                video_uint8 = [tf.squeeze(tf.image.convert_image_dtype(frame, dtype=tf.uint8)) for frame in video]
                video_uint8 = [tf.expand_dims(frame, -1) if frame.ndim == 2 else frame for frame in video_uint8]

                try:
                    imageio.mimsave('animation.gif', video_uint8, fps=10)
                    st.image('animation.gif', width=400)
                except TypeError as e:
                    st.error(f"Error saving GIF: {e}")

                st.info('This is the output of the machine learning model as tokens')
                model = load_model()

                yhat = model.predict(tf.expand_dims(video, axis=0))
                st.write("Raw model output (first timestep):")
                st.write(yhat[0, 0, :])

                decoded, log_probabilities = tf.keras.backend.ctc_decode(
                    yhat,
                    input_length=[yhat.shape[1]],
                    greedy=False,
                    beam_width=100
                )

                decoded = decoded[0].numpy()
                pred_text = []
                for idx in decoded[0]:
                    if idx != -1:
                        pred_text.append(idx)

                pred_text = tf.strings.reduce_join(num_to_char(pred_text)).numpy().decode('utf-8')

                pred_text = ''.join(char for i, char in enumerate(pred_text) if i == 0 or char != pred_text[i-1])
                pred_text = pred_text.replace(' ', '')

                st.info('Decoded prediction:')
                st.text(pred_text)

                st.info('Confidence scores for each character:')
                char_confidences = tf.reduce_max(yhat, axis=-1)
                for char, conf in zip(pred_text, char_confidences[0]):
                    st.text(f"{char}: {conf:.2f}")

                with open(alignment_file_path, 'r') as f:
                    ground_truth = ' '.join([line.split()[-1] for line in f if line.split()[-1] != 'sil'])

                st.info('Ground truth:')
                st.text(ground_truth)

                accuracy = SequenceMatcher(None, pred_text, ground_truth).ratio()
                st.info(f'Accuracy: {accuracy:.2%}') 

                # Translation Feature
                st.subheader("Translation")

                # Create a dropdown for language selection
                lang_options = list(LANGUAGES.values())
                selected_lang = st.selectbox('Select language for translation:', lang_options)
                lang_code = list(LANGUAGES.keys())[lang_options.index(selected_lang)]

                # Create a Translator object
                translator = Translator()

                # Translate ground truth to the selected language
                translation = translator.translate(ground_truth, dest=lang_code)
                st.text_area("Translated Text:", translation.text)

                # Add an option for reverse translation (translating from selected language to English)
                reverse_translation = translator.translate(translation.text, src=lang_code, dest='en')
                st.text_area("Reverse Translated Text (back to English):", reverse_translation.text)

                # Text-to-Speech Feature
                st.subheader("Text-to-Speech")

                # Generate audio from translated text
                if st.button('Play Translated Audio'):
                    tts = gTTS(translation.text, lang=lang_code)
                    
                    # Create a temporary file to save the audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                        tts.save(tmp_file.name)
                        st.audio(tmp_file.name, format='audio/mp3')
                        
                    # Optionally, you could delete the temp file after use to avoid cluttering
                    # os.unlink(tmp_file.name)

                # Option for playing reverse translated audio (back to English)
                if st.button('Play Reverse Translated Audio'):
                    tts_reverse = gTTS(reverse_translation.text, lang='en')
                    
                    # Create a temporary file to save the reverse audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file_reverse:
                        tts_reverse.save(tmp_file_reverse.name)
                        st.audio(tmp_file_reverse.name, format='audio/mp3')

                    # Optionally, you could delete the temp file after use to avoid cluttering
                    # os.unlink(tmp_file_reverse.name)

# Python script file exported from voice_capture_response.ipynb

from gtts import gTTS
from io import BytesIO
import speech_recognition as sr

from pydub import AudioSegment
from pydub.playback import play

from dotenv import load_dotenv
load_dotenv()

import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
gemini_pro = genai.GenerativeModel(model_name="models/gemini-pro")

import streamlit as st

# define a function to listen to microphone
def listen_to_microphone(sr):

    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("Adjusting noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening to microphone...")
            recorded_audio = recognizer.listen(source)
            print("Done Listening.")

            return recorded_audio
            
    except Exception as ex:
        print("Something wrong during listening:", ex)
        return None

# define a function to recognize voice
def audio_to_text(audio, sr):

    recognizer = sr.Recognizer()

    text = "Sorry, I can't hear you!"
    audio_normal = False
    
    if audio:
        try:
            print("Recognizing the text...")
            text = recognizer.recognize_google(audio, language="en-US")

            audio_normal = True
            
            print("Decoded Text: {}".format(text))
            
        except sr.UnknownValueError as ex:
            print("Google Speech Recognition could not understand the audio.", ex)
    
            text = "Google Speech Recognition could not understand the audio."
            
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.", ex)
    
            text = "Could not request results from Google Speech Recognition service."
            
        except Exception as ex:
            print("Error during recognition.", ex)
    
            text = "Error during recognition."

    return (audio_normal, text)
        
## create a function converting text to voice
def text_to_speech(text):

    try:
        tts = gTTS(text)
    
        # Save the audio object into a buffer
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)

        # Set the buffer's position to the start
        audio_buffer.seek(0)

        # Load MP3 data into an AudioSegment
        audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")

        # Play the audio
        play(audio_segment)       

    except Exception as ex:
        print("Error during text to speech:", ex)


role = '''
        You are an intelligent assistant to chat on the topic:
        `{}`.    
    '''

topic = '''
        The future of artificial intelligence
    '''

role_text = role.format(topic)

instructions = '''
        Respond to the INPUT_TEXT briefly in chat style.
        Respond based on your knowledge about `{}` in brief chat style. 
    '''

instructions_text = instructions.format(topic)

## create a function to respond to input text with Gemini pro
def respond_by_gemini(input_text, role_text, instructions_text):

    final_prompt = [
        "ROLE: " + role_text,
        "INPUT_TEXT: " + input_text,
        instructions_text,
    ]

    response = gemini_pro.generate_content(
            final_prompt,
            stream=True,
        )

    response_list = []
    for chunk in response:
        response_list.append(chunk.text)
        
    response_text = "".join(response_list)

    return response_text


## Streamlit is a simple application to design an interactive interface 

st.title("Voice-to-Voice Chat Application Supported by an LLM")

# create a button to start conversation
talk_btn = st.button("Let's Talk")

# add a session state to hold the user inputs and LLM responses
if('messages' not in st.session_state):
    st.session_state['messages'] = []

# when the button is clicked
if talk_btn:

    st.write("Listening...")

    # listen to the microphone
    recorded_audio = listen_to_microphone(sr)

    # recognize the voice
    audio_normal, text = audio_to_text(recorded_audio, sr)

    # if the voice input is normal
    if audio_normal:
        
        st.write("Thinking...")

        # add the voice input as the user's message
        st.session_state['messages'].append({
            'role': 'user',
            'message': text
        })

        # ask the LLM to respond
        response_text = respond_by_gemini(text, role_text, instructions)

        # add the LLM's response as the assistant's message
        st.session_state['messages'].append({
            'role': 'assistant',
            'message': response_text
        })

        # voice output
        text_to_speech(response_text)
        
    else:
        # if the voice is abnormal, add the error message to the assistant
        st.session_state['messages'].append(
            {
                'role': 'assistant',
                'message': text
            })

        st.write("Click the button to try again.")

        # play the error message
        text_to_speech(text)
        
    st.write("Click the button to continue.")

# display the conversation history 
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['message'])






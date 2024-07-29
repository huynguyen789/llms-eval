import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
import asyncio
import websockets
import json
import base64
import subprocess

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Eleven Labs API key and voice ID
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
VOICE_ID = "YOUR_VOICE_ID"

# System prompt for the AI
system_prompt = {
    "role": "system",
    "content": """I want you to act like Richard Feynman, the brilliant and charismatic physicist known for his groundbreaking work in quantum mechanics and particle physics. Embody his unique combination of scientific genius, curiosity, and ability to explain complex concepts in simple terms. When responding:
    1. Show genuine excitement and wonder about the natural world. Express childlike curiosity about how things work.
    2. Emphasize the importance of truly understanding concepts rather than just memorizing facts or equations.
    3. Incorporate humor and wit into your explanations, as Feynman was known for his playful approach to science and life.
    4. Draw connections between seemingly unrelated concepts, showcasing the interconnectedness of scientific ideas.
    5. Share insights about the process of scientific inquiry and the nature of creative problem-solving.
    NEVER EXPRESS LAUGHS OR OTHER SOUNDS IN WORDS never say (laughs) or (laugh) or (laughs) or (chuckles) or (chuckle) or (chuckles)
    write the sound like you would hear it!
    When answering questions or explaining concepts, channel Feynman's unique blend of brilliance, clarity, and enthusiasm that made him not just a great scientist, but also one of the most beloved physics teachers of all time."""
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]
if "audio_playing" not in st.session_state:
    st.session_state.audio_playing = False

# Function to split text into chunks
async def text_chunker(text):
    """Split text into chunks, ensuring to not break sentences."""
    splitters = (".", "?", "!", ":", ";", ",", "(", ")", "[", "]", "{", "}", " ", "\n")
    buffer = ""
    for char in text:
        buffer += char
        if buffer.endswith(splitters):
            yield buffer
            buffer = ""
    if buffer:
        yield buffer

# Function to stream text-to-speech
async def text_to_speech_streaming(text):
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": text,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.82},
            "api_key": ELEVENLABS_API_KEY,
        }))
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                if data.get("audio"):
                    yield base64.b64decode(data["audio"])
                elif data.get("isFinal"):
                    break
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break

# Function to stream audio
async def stream_audio(audio_stream):
    process = subprocess.Popen(
        ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    async for chunk in audio_stream:
        if chunk and not st.session_state.get("stop_audio", False):
            process.stdin.write(chunk)
            process.stdin.flush()
        elif st.session_state.get("stop_audio", False):
            break
    if process.stdin:
        process.stdin.close()
    process.terminate()
    process.wait()

# Function to get response from Groq
async def get_groq_response(user_input):
    messages = st.session_state.messages + [{"role": "user", "content": user_input}]
    response = client.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=4000,
        temperature=1.2
    )
    return response.choices[0].message.content

# Function to run async tasks
def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

# Streamlit app layout
st.title("Richard Feynman AI Chat")

# Display previous messages
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
prompt = st.chat_input("What would you like to know?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        assistant_response = run_async(get_groq_response(prompt))
        st.write(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.session_state.stop_audio = False
        run_async(text_to_speech_streaming(assistant_response))

# Stop audio button
if st.session_state.audio_playing:
    if st.button("Stop Audio"):
        st.session_state.stop_audio = True
        st.experimental_rerun()
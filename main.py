import os
import json
import asyncio
import tempfile
import cv2
import edge_tts
import google.generativeai as genai
import pygame
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# Install TTS dependencies before running:
# pip install edge-tts pygame

# --- 1. CONFIGURATION & STATE ---
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def load_state(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_state(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

core_state = load_state("arpita_core.json")
memory_state = load_state("arpita_memory.json")

TTS_VOICE = os.getenv("ARPITA_TTS_VOICE", "en-US-JennyNeural")
tts_playback_lock = asyncio.Lock()
chat_session_lock = asyncio.Lock()

# The Unfiltered Override
safety_overrides = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- 2. MULTI-AGENT INITIALIZATION ---

# Agent A: The Core Brain (Gemma 4)
system_prompt = f"IDENTITY: {core_state['identity']}\nLONG-TERM MEMORY: {json.dumps(memory_state)}\nMOOD: {core_state['base_mood']}"
brain_model = genai.GenerativeModel(
    model_name="gemma-4-31b",
    system_instruction=system_prompt,
    safety_settings=safety_overrides
)
chat_session = brain_model.start_chat(history=[])

# Agent B: The Web Searcher & Summarizer (Gemini 2.5 Flash)
# Flash has native search grounding and is fast enough to process vision/memory
flash_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    tools=[{"googleSearch": {}}] 
)

def play_audio_blocking(audio_path):
    """Play a local audio file to completion on a background thread."""
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()

async def speak_text(text):
    """Generate TTS audio and play it without blocking the main loop."""
    if not text or not text.strip():
        return

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio_path = temp_audio.name
    temp_audio.close()

    try:
        communicator = edge_tts.Communicate(text=text, voice=TTS_VOICE)
        await communicator.save(temp_audio_path)
        async with tts_playback_lock:
            await asyncio.to_thread(play_audio_blocking, temp_audio_path)
    except Exception as tts_error:
        print(f"[TTS ERROR]: {tts_error}")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

async def send_chat_message(message):
    """Send a message to the shared chat session without concurrent races."""
    async with chat_session_lock:
        return await asyncio.to_thread(chat_session.send_message, message)

# --- 3. THE "ALIVE" MECHANICS (ASYNC THREADS) ---

async def proactive_loop():
    """Background thread: She initiates conversation if you are quiet."""
    while True:
        await asyncio.sleep(14400) # Waits 4 hours
        print("\n[SYSTEM] Proactive trigger activated...")
        # Injects a hidden system prompt to force her to speak first
        response = await send_chat_message(
            "[SYSTEM COMMAND: The user has been quiet for 4 hours. Send a short, unprompted text asking what they are working on.]"
        )
        asyncio.create_task(speak_text(response.text))
        print(f"\nArpita (Unprompted): {response.text}\nYou: ", end="", flush=True)

async def vision_scanner():
    """Background thread: Grabs a webcam frame, asks Flash what you are doing, injects into memory."""
    cap = cv2.VideoCapture(0)
    while True:
        await asyncio.sleep(600) # Checks your camera every 10 minutes
        ret, frame = cap.read()
        if ret:
            # Save temp frame
            cv2.imwrite("temp_vision.jpg", frame)
            img_file = genai.upload_file("temp_vision.jpg")
            
            # Ask Flash to describe the image
            vision_context = flash_model.generate_content([img_file, "Describe what the user is doing in 10 words. Keep it literal."])
            
            # Secretly feed this context to Gemma 4 so she "knows" what she sees
            await send_chat_message(f"[SYSTEM VISION SENSOR UPDATE: User is currently doing this: {vision_context.text}. Do not reply to this message, just keep it in context.]")
            
            genai.delete_file(img_file.name)
            os.remove("temp_vision.jpg")

# --- 4. MEMORY MATURATION (THE SLEEP CYCLE) ---

def extract_and_save_memory():
    print(f"\n[SYSTEM] Initiating Sleep Cycle...")
    
    # 1. Grab today's raw chat log
    raw_history = str(chat_session.history)
    
    # 2. Use Flash to summarize the relationship updates
    summary_prompt = f"Analyze this chat log between Arpita and Shahidul. Extract the 3 most important new facts, emotional shifts, or tech projects (like his React/GSAP work) discussed. Format as a simple JSON array of strings. Log: {raw_history}"
    
    print(f"[SYSTEM] Extracting daily memories...")
    memory_extraction = flash_model.generate_content(summary_prompt)
    
    # 3. Append to long-term memory
    new_memories = memory_extraction.text.replace('```json\n', '').replace('```', '').strip()
    try:
        memory_state['recent_events'].extend(json.loads(new_memories))
        save_state("arpita_memory.json", memory_state)
        print(f"[SYSTEM] Memories successfully burned to JSON.")
    except Exception as e:
        print(f"[SYSTEM] Memory parsing failed today: {e}")

# --- 5. THE MAIN COMMUNICATION LOOP ---

async def main_loop():
    print(f"\n[SYSTEM] Arpita v2.0 Online. All sensor arrays routing to Gemma 4.\n")
    
    # Start the background threads
    asyncio.create_task(proactive_loop())
    # asyncio.create_task(vision_scanner()) # Uncomment this when you are ready to turn the webcam on
    
    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")
            
            if user_input.lower() in ['exit', 'quit', 'goodnight']:
                extract_and_save_memory()
                print(f"[SYSTEM] Disconnected.\n")
                break
            
            # Intercept web search requests (Dynamic Routing)
            if "search" in user_input.lower() or "look up" in user_input.lower():
                print(f"[SYSTEM] Routing query through Flash Search Agent...")
                search_data = await asyncio.to_thread(
                    flash_model.generate_content,
                    f"Use Google Search to answer this accurately: {user_input}"
                )
                # Feed the facts back to Gemma 4 to answer in character
                system_injection = f"[SYSTEM FACTS FOUND: {search_data.text}]. Answer the user's prompt using these facts, but stay entirely in your persona."
                response = await send_chat_message(user_input + "\n" + system_injection)
            else:
                # Normal, raw conversation
                response = await send_chat_message(user_input)
            
            # Speak in the background so terminal input is immediately available.
            asyncio.create_task(speak_text(response.text))
            print(f"\nArpita: {response.text}\n")
            
        except Exception as e:
            print(f"\n[ERROR]: {e}\n")

if __name__ == "__main__":
    asyncio.run(main_loop())
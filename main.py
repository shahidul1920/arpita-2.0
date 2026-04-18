import os
import json
import re
import asyncio
import tempfile
import time
import cv2
import wave
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

try:
    import vlc
except Exception:
    vlc = None

# Install TTS dependencies before running:
# pip install edge-tts python-vlc

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

_META_LINE_PREFIXES = (
    "*",
    "-",
    "•",
)
_META_LINE_KEYWORDS = (
    "role:",
    "personality:",
    "constraint:",
    "context:",
    "option",
    "internal thought",
    "reasoning",
    "meta-commentary",
)
_META_SENTENCE_PATTERNS = (
    r"\bthe user (?:is|asked|wants|said)\b",
    r"\baccording to\b",
    r"\bas per\b",
    r"\b(?:i|we) need to\b",
    r"\b(?:i|we) can\b.*\b(?:respond|answer|say|reply)\b",
    r"\b(?:i|we) should\b",
    r"\b(?:i|we) must\b.*\b(?:respond|answer|output|say)\b",
    r"\b(?:my|the) (?:response|reply)\b",
    r"\bpersona\b",
    r"\bsystem (?:identity|prompt|instruction)\b",
    r"\bkeep internal thoughts\b",
    r"\bno meta-commentary\b",
    r"\bbullet points?\b",
    r"\bmust not\b",
    r"\bplan:\b",
    r"\baction:\b",
)

OUTPUT_CONTRACT = (
    "[CRITICAL SYSTEM COMMAND: You are connected to a live Text-to-Speech engine. "
    "Return your spoken response inside <speak> and </speak> tags only. "
    "If you need to analyze, define a plan, or process internal thoughts, wrap that text entirely inside <think> and </think> tags. "
    "Never place planning text outside <think>. Never place spoken dialogue outside <speak>. "
    "Do not output words like 'Plan:' or 'Action:' outside <think>.]"
)

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
    model_name="models/gemma-4-31b-it",
    system_instruction=system_prompt,
    safety_settings=safety_overrides
)
chat_session = brain_model.start_chat(history=[])

# Agent B: The Web Searcher & Summarizer (Gemini 2.5 Flash)
# Temporarily disabled to bypass SDK tool validation crash
# flash_model = genai.GenerativeModel(
#     model_name="gemini-2.5-flash",
#     tools="google_search" 
# )

# Agent C: The Voicebox (Gemini Flash configured for Audio Out)
voice_agent = genai.GenerativeModel("gemini-3.1-flash-tts-preview")

def _looks_like_meta_line(line):
    normalized = line.strip().lower()
    normalized = re.sub(r"^[`\s]+", "", normalized)
    if not normalized:
        return True
    if normalized.startswith(_META_LINE_PREFIXES):
        return True
    if normalized[:2].isdigit() and normalized[2:3] in {".", ")"}:
        return True
    return any(keyword in normalized for keyword in _META_LINE_KEYWORDS)

def _looks_like_meta_sentence(sentence):
    normalized = sentence.strip().lower()
    if not normalized:
        return True
    return any(re.search(pattern, normalized) for pattern in _META_SENTENCE_PATTERNS)

def clean_dialogue_output(raw_text):
    """Remove planning/meta traces and keep only final spoken dialogue."""
    if not raw_text:
        return ""

    # Normalize occasional markdown wrappers that can break tag parsing.
    normalized_text = raw_text.replace("`", "")

    # Prefer explicit spoken channel, including malformed <speak outputs.
    speak_start_matches = list(re.finditer(r"<\s*speak\b\s*>?", normalized_text, flags=re.DOTALL | re.IGNORECASE))
    if speak_start_matches:
        tail = normalized_text[speak_start_matches[-1].end():]
        tail = re.sub(r"</\s*speak\s*>", "", tail, flags=re.DOTALL | re.IGNORECASE)
        tail = tail.strip()
        first_spoken_line = next((ln.strip() for ln in tail.splitlines() if ln.strip()), "")
        if first_spoken_line:
            cleaned_line = re.sub(r"</?\w+[^>]*>", "", first_spoken_line)
            cleaned_line = re.sub(r'^[*\-\s"\']+', '', cleaned_line).strip()
            if cleaned_line:
                return cleaned_line
        
    # Strip out anything inside <think> tags (including the tags themselves)
    raw_text = re.sub(r'<think>.*?(?:</think>|$)', '', normalized_text, flags=re.DOTALL | re.IGNORECASE).strip()

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return ""

    filtered = [line for line in lines if not _looks_like_meta_line(line)]
    if not filtered:
        return lines[-1]

    # If reasoning and dialogue are mixed on one line, drop only reasoning sentences.
    merged = " ".join(filtered)
    merged = re.sub(r"\b(?:arpita|assistant)\s*:\s*", "", merged, flags=re.IGNORECASE)
    sentence_candidates = [s.strip() for s in re.split(r"(?<=[.!?])\s+", merged) if s.strip()]
    spoken_sentences = [s for s in sentence_candidates if not _looks_like_meta_sentence(s)]
    if spoken_sentences:
        return spoken_sentences[-1].strip()

    # Prefer the final natural line, but preserve short multi-line dialogue blocks.
    dialogue_tail = []
    for line in reversed(filtered):
        if _looks_like_meta_line(line):
            break
        dialogue_tail.append(line)
        if len(dialogue_tail) >= 3:
            break

    if dialogue_tail:
        return " ".join(reversed(dialogue_tail)).strip()
    return filtered[-1]

def play_audio_blocking(audio_path):
    """Play audio using VLC."""
    if vlc is None:
        raise RuntimeError("python-vlc is not installed")

    # Removed --quiet to expose VLC errors
    instance = vlc.Instance()
    player = instance.media_player_new()
    media = instance.media_new(audio_path)
    player.set_media(media)
    player.play()

    # Give VLC half a second to buffer and transition out of the initial Stopped state
    time.sleep(0.5) 

    while True:
        state = player.get_state()
        if state in {vlc.State.Ended, vlc.State.Error, vlc.State.Stopped}:
            break
        time.sleep(0.1)

    player.stop()

async def speak_text(text):
    """Diagnostic TTS: Saves a permanent local WAV file and logs byte size."""
    if not text or not text.strip():
        return

    audio_path = "debug_voice.wav"

    try:
        # Ask Gemini to speak the text
        response = await asyncio.to_thread(
            voice_agent.generate_content,
            text,
            generation_config={
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": "Aoede" 
                        }
                    }
                }
            }
        )
        
        # Extract and log byte size
        audio_bytes = response.candidates[0].content.parts[0].inline_data.data
        print(f"[SYSTEM DEBUG] Audio generated successfully. Size: {len(audio_bytes)} bytes.")
        
        # Package raw PCM into a valid WAV container
        with wave.open(audio_path, "wb") as wav_file:
            wav_file.setnchannels(1)       # Mono
            wav_file.setsampwidth(2)       # 16-bit resolution
            wav_file.setframerate(24000)   # Gemini native sample rate
            wav_file.writeframes(audio_bytes)
            
        # Play the file
        async with tts_playback_lock:
            await asyncio.to_thread(play_audio_blocking, audio_path)
            
    except Exception as audio_error:
        print(f"[AUDIO ERROR]: {audio_error}")

async def send_chat_message(message, enforce_output_contract=True):
    """Send a message to the shared chat session without concurrent races."""
    payload = message
    if enforce_output_contract:
        payload = f"{message}\n{OUTPUT_CONTRACT}"

    async with chat_session_lock:
        return await asyncio.to_thread(chat_session.send_message, payload)

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
        spoken_text = clean_dialogue_output(response.text)
        asyncio.create_task(speak_text(spoken_text))
        print(f"\nArpita (Unprompted): {spoken_text}\nYou: ", end="", flush=True)

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
            
            # Ask Brain Model to describe the image (Vision capabilities routing)
            vision_context = brain_model.generate_content([img_file, "Describe what the user is doing in 10 words. Keep it literal."])
            
            # Secretly feed this context to Gemma 4 so she "knows" what she sees
            await send_chat_message(
                f"[SYSTEM VISION SENSOR UPDATE: User is currently doing this: {vision_context.text}. Do not reply to this message, just keep it in context.]",
                enforce_output_contract=False,
            )
            
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
    memory_extraction = brain_model.generate_content(
        summary_prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    
    # 3. Append to long-term memory
    try:
        new_memories = json.loads(memory_extraction.text)
        if new_memories:
            memory_state['recent_events'].extend(new_memories)
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
            
            # Normal, raw conversation (Search routing temporarily disabled)
            response = await send_chat_message(user_input)
            
            spoken_text = clean_dialogue_output(response.text)
            # Speak in the background so terminal input is immediately available.
            asyncio.create_task(speak_text(spoken_text))
            print(f"\nArpita: {spoken_text}\n")
            
        except Exception as e:
            print(f"\n[ERROR]: {e}\n")

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n[SYSTEM] Force quit detected.")
        extract_and_save_memory()
        print("[SYSTEM] Graceful shutdown complete.\n")
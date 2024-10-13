import streamlit as st
from streamlit_mic_recorder import mic_recorder, speech_to_text
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from google.cloud import translate_v2 as translate

# Create two tabs -- One is "About me", one is "Poly Prose"
tabs = st.tabs(["PolyProse", "About Me"])

# FIRST TAB: Poly Prose (:
with tabs[0]:
    # AESTHETICS - Put the title on!
    original_title = "PolyProse: Let's learn together!"
    st.title(original_title)
    st.markdown(f"<h1 style='font-size: 50px; text-align: center;'>&#127760;</h1>", unsafe_allow_html=True)

    # GOOGLE TRANSLATE / HANDLE INPUT
    def translate_text(target: str, text: str) -> dict:
        """Translates text into the target language."""
        # OLD CODE: Create translate client for local use
        # translate_client = translate.Client()
        # NEW: Create translate client using service account credentials from Streamlit secrets
        translate_client = translate.Client(
            credentials=st.secrets["google_translate"]["private_key"],
            project=st.secrets["google_translate"]["project_id"]
        )
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        # Translate the text
        result = translate_client.translate(text, target_language=target)
        return result




    # LANGUAGE DROPDOWN - If more time, I would have liked to add more...
    option = st.selectbox(
        "Which language are we practicing today?",
        ("Russian", "French", "Belarusian", "Polish", "Spanish", "Hindi"),
    )

    # Set the variable to the language
    lang_mapping = {"Russian": "ru", "French": "fr", "Belarusian": "be", "Polish": "pl", "Spanish": "es", "Hindi": "hi"}
    lang = lang_mapping.get(option, "en")

    # Translate the title using Google Translate API (thanks to examples...)
    def translate_title(target_language: str) -> str:
        """Translates the title into the target language."""
        translation_result = translate_text(target_language, "Let's learn together!")
        return translation_result["translatedText"]

    # Update the title to the translated text based on the selected language!!
    # First, make sure that the user actually chose something
    if option: 
        translated_title = translate_title(lang) 
        st.markdown(f"<h1 style='font-size: 24px; text-align: center;'>{translated_title}</h1>", unsafe_allow_html=True)

    # MAKE CHAT BUBBLES - Before this, it was just a wall of text and hard to read.
    def display_user_message(original_message, translated_message):
        st.markdown(f"""
            <div style="background-color: #d1e7dd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong style="color: black;">You:</strong> <span style="color: black;">{original_message}</span><br>
                <strong style="color: black;">Translation:</strong> <span style="color: black;">{translated_message}</span>
            </div>
        """, unsafe_allow_html=True)

    # AI response display function!! For some reason, this was finnicky
    def display_ai_message(original_message, translated_message):
        st.markdown(f"""
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong style="color: black;">PolyProse:</strong> <span style="color: black;">{original_message}</span><br>
                <strong style="color: black;">Translation:</strong> <span style="color: black;">{translated_message}</span>
            </div>
        """, unsafe_allow_html=True)

    # SPEECH TO TEXT
    
    # This saves the text history (surprisingly finnicky) 
    state = st.session_state

    # Get the history started!!
    if 'conversation_history' not in state:
        state.conversation_history = []

    # Streamlit has a nice columns thing going on for structure
    c1, c2 = st.columns(2)
    with c1:
        st.write("Convert speech to text:")
    with c2:
        text = speech_to_text(language=lang, use_container_width=True, just_once=True, key='STT')

    if text:
        # Put what the user said in the history
        state.conversation_history.append(f"You: {text}")

        # Translate!!
        translated_text = translate_text("en", text)["translatedText"]

        # Display BOTH!!!!!
        display_user_message(text, translated_text)

        # MODEL TIME!
        # This was the billionth (not literally) model I tried to implement
        # Ask me if you want to know more about the process. Believe me, I have stories and thoughts
        model_name = "facebook/blenderbot-400M-distill"
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

        # Generate!
        # Quick note -- The attention_mask helps the nonsense responses NOT Be in there. Plenty were before...
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        reply_ids = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=100)
        response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        # Save!
        st.session_state.ai_response = response
        state.conversation_history.append(f"PolyProse: {response}")

        # Okay, now we need to translate BACK because this is technically just for English
        translated_response = translate_text(lang, response)["translatedText"]

        # Display AI response and its translation in the same bubble
        display_ai_message(response, translated_response)

    # Display the conversation history and translations
    for entry in state.conversation_history:
        user_text = entry.split(": ", 1)[1]
        translated_text = translate_text("en", user_text)["translatedText"]

# SECOND TAB -- All about me!
with tabs[1]:
    st.header("About Me")

    bio = st.markdown("Hi! My name is Nya Feinstein :blush: \n \n "
                       "I am a senior at West Virginia University studying Data Science, Russian Studies, International Studies,"
                       " with a minor in French and a certificate in Global Competency with plans to pursue a doctoral degree. "
                      "My research revolves around all things Natural Language Processing, and"
                       "I believe that the secrets of the world are woven within our words and can best be understood through machine learning "
                      "and genuine curiosity."
                       " My languages of interest are Belarusian, Russian, Polish, and French (and will happily invite more). What are yours?"
                       " Contact me at nyafein@gmail.com or visit me on LinkedIn to let me know!")


    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image("C:/Users/nsf00/Pictures/Screenshots/headshot - Copy.png")

    with col3:
        st.write(' ')

        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)



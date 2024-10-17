import streamlit as st
from streamlit_mic_recorder import mic_recorder, speech_to_text
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import argostranslate.package
import argostranslate.translate
import time

def load_language_package(from_code, to_code):
    """Load the language package from Argos Translate."""
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code,
            available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

def translate_text(from_language: str, to_language: str, text: str) -> dict:
    """Translates text from one language to another using Argos Translate."""
    load_language_package(from_language, to_language)
    translated_text = argostranslate.translate.translate(text, from_language, to_language)
    return {"translatedText": translated_text}

# Create two tabs -- One is "About me", one is "PolyProse"
tabs = st.tabs(["PolyProse", "About Me", "Sources"])

# FIRST TAB: Poly Prose
with tabs[0]:
    original_title = "PolyProse: Let's learn together!"
    st.title(original_title)
    st.markdown(f"<h1 style='font-size: 50px; text-align: center;'>&#127760;</h1>", unsafe_allow_html=True)

    option = st.selectbox(
        "Which language are we practicing today?",
        ("Russian", "French", "Polish", "Spanish", "Hindi"),
    )

    lang_mapping = {"Russian": "ru", "French": "fr", "Polish": "pl", "Spanish": "es", "Hindi": "hi"}
    lang = lang_mapping.get(option, "en")

    def translate_title(target_language: str) -> str:
        """Translates the title from English to the target language."""
        translation_result = translate_text("en", target_language, "Let's learn together!")
        return translation_result["translatedText"]

    if option:
        translated_title = translate_title(lang)
        st.markdown(f"<h1 style='font-size: 24px; text-align: center;'>{translated_title}</h1>", unsafe_allow_html=True)

    def display_user_message(original_message, translated_message):
        st.markdown(f"""
            <div style="background-color: #d1e7dd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong style="color: black;">You:</strong> <span style="color: black;">{original_message}</span><br>
                <strong style="color: black;">Translation:</strong> <span style="color: black;">{translated_message}</span>
            </div>
        """, unsafe_allow_html=True)

    def display_ai_message(original_message, translated_message):
        st.markdown(f"""
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong style="color: black;">PolyProse:</strong> <span style="color: black;">{original_message}</span><br>
                <strong style="color: black;">Translation:</strong> <span style="color: black;">{translated_message}</span>
            </div>
        """, unsafe_allow_html=True)

    # Handle session state safely
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'ai_response' not in st.session_state:
        st.session_state.ai_response = ""

    try:
        c1, c2 = st.columns(2)
        with c1:
            st.write("Convert speech to text:")
        with c2:
            text = speech_to_text(language=lang, use_container_width=True, just_once=True, key='STT')

        if text:
            st.session_state.conversation_history.append(f"You: {text}")

            with st.spinner('Pondering...'):
                translated_text = translate_text(lang, "en", text)["translatedText"]
                display_user_message(text, translated_text)

                # Model integration (BlenderBot)
                model_name = "facebook/blenderbot-400M-distill"
                tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
                model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                reply_ids = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=100)
                response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

                st.session_state.ai_response = response
                st.session_state.conversation_history.append(f"PolyProse: {response}")

                translated_response = translate_text("en", lang, response)["translatedText"]
                display_ai_message(response, translated_response)

        # Display conversation history
        for entry in st.session_state.conversation_history:
            st.write(entry)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# SECOND TAB: About Me
with tabs[1]:
    st.header("About Me")
    bio = st.markdown("Hi! My name is Nya Feinstein :blush: \n \n "
                      "I am a senior at West Virginia University studying Data Science, Russian Studies, International Studies,"
                      " with a minor in French and a certificate in Global Competency with plans to pursue a doctoral degree. "
                      "My research revolves around all things Natural Language Processing, and"
                      " I believe that the secrets of the world are woven within our words and can best be understood through machine learning "
                      "and genuine curiosity."
                      " My languages of interest are Belarusian, Russian, Polish, and French (and will happily invite more). What are yours?"
                      " Contact me at nyafein@gmail.com or visit me on LinkedIn to let me know!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image("./headshot.png")
    with col3:
        st.write(' ')

with tabs[2]:
    st.header("Sources")
    argosurl = 'https://github.com/argosopentech/argos-translate'
    st.markdown(f'<a href={argosurl}><button style="background-color:Blue;">Argos Model</button></a>', unsafe_allow_html=True)
    blenderurl = 'https://huggingface.co/docs/transformers/model_doc/blenderbot#transformers.BlenderbotForCausalLM'
    st.markdown(f'<a href={blenderurl}><button style="background-color:Blue;">BlenderBot Model</button></a>', unsafe_allow_html=True)

import streamlit as st
from streamlit_mic_recorder import speech_to_text
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import argostranslate.package
import argostranslate.translate

# Cache the language model package for translation
@st.cache_resource
def load_language_package(from_code, to_code):
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages)
    )
    argostranslate.package.install_from_path(package_to_install.download())

# Translation function
def translate_text(from_language, to_language, text):
    load_language_package(from_language, to_language)
    translated_text = argostranslate.translate.translate(text, from_language, to_language)
    return translated_text

# Cache the Blenderbot model to avoid reloading
@st.cache_resource
def load_blenderbot_model():
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# Chat response generation
def generate_response(input_text, tokenizer, model):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    reply_ids = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=50)
    return tokenizer.decode(reply_ids[0], skip_special_tokens=True)

# UI - Three tabs
tabs = st.tabs(["PolyProse", "About Me", "Sources"])

# First Tab - PolyProse
with tabs[0]:
    try:
        st.title("PolyProse: Let's learn together!")
        st.markdown("<h1 style='font-size: 50px; text-align: center;'>&#127760;</h1>", unsafe_allow_html=True)

        option = st.selectbox(
            "Which language are we practicing today?",
            ("Russian", "French", "Polish", "Spanish", "Hindi"),
        )

        lang_mapping = {"Russian": "ru", "French": "fr", "Polish": "pl", "Spanish": "es", "Hindi": "hi"}
        lang = lang_mapping.get(option, "en")

        # Load Blenderbot model once
        tokenizer, model = load_blenderbot_model()

        # Speech-to-text (Optional for testing)
        c1, c2 = st.columns(2)
        with c1:
            st.write("Convert speech to text:")
        with c2:
            user_text = speech_to_text(language=lang, use_container_width=True, just_once=True, key='STT')

        if user_text:
            # Translate and display user message
            translated_user_text = translate_text(lang, "en", user_text)
            st.write(f"You: {user_text} (Translated: {translated_user_text})")

            # Generate AI response
            ai_response = generate_response(translated_user_text, tokenizer, model)

            # Translate AI response back to user’s language
            translated_ai_response = translate_text("en", lang, ai_response)
            st.write(f"PolyProse: {ai_response} (Translated: {translated_ai_response})")

        if st.button("Refresh"):
            st.experimental_rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Second Tab - About Me
with tabs[1]:
    st.header("About Me")
    bio = st.markdown("""
        Hi! My name is Nya Feinstein :blush: \n
        I am a senior at West Virginia University studying Data Science, Russian Studies, International Studies...
    """)
    st.image("./headshot.png")

# Third Tab - Sources
with tabs[2]:
    st.header("Sources")
    st.markdown("[Argos Translate](https://github.com/argosopentech/argos-translate)")
    st.markdown("[BlenderBot Model](https://huggingface.co/facebook/blenderbot-400M-distill)")

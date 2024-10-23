import streamlit as st
from streamlit_mic_recorder import speech_to_text
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import argostranslate.package
import argostranslate.translate


# Cache the language model package for translation (Caching is very useful)
@st.cache_resource
def load_language_package(from_code, to_code):
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages))
    argostranslate.package.install_from_path(package_to_install.download())


# Translation function!! This works just fine
def translate_text(from_language, to_language, text):
    load_language_package(from_language, to_language)
    translated_text = argostranslate.translate.translate(text, from_language, to_language)
    return translated_text


# Cache the Blenderbot model to avoid reloading (if not, it uses far too many resources)
@st.cache_resource
def load_blenderbot_model():
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# User message
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


# Function for response
def generate_response(input_text, tokenizer, model):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    reply_ids = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=50)
    return tokenizer.decode(reply_ids[0], skip_special_tokens=True)

def translate_title(target_language: str) -> str:
    """Translates the title from English to the target language."""
    translation_result = translate_text("en", target_language, "Let's learn together!")
    return translation_result


# UI - Three tabs
tabs = st.tabs(["PolyProse", "About Me", "Sources"])

# First Tab - PolyProse
with tabs[0]:
    try:
        # AESTHETICS - Put the title on!
        original_title = "PolyProse: Let's learn together!"
        st.title(original_title)
        st.markdown(f"<h1 style='font-size: 50px; text-align: center;'>&#127760;</h1>", unsafe_allow_html=True)

        option = st.selectbox(
            "Which language are we practicing today?",
            ("Russian", "French", "Polish", "Spanish", "Hindi"),
        )

        lang_mapping = {"Russian": "ru", "French": "fr", "Polish": "pl", "Spanish": "es", "Hindi": "hi"}
        lang = lang_mapping.get(option, "en")

        # Update the title to the translated text based on the selected language!!
        if option:
            translated_title = translate_title(lang)
            st.markdown(f"<h1 style='font-size: 24px; text-align: center;'>{translated_title}</h1>",
                        unsafe_allow_html=True)


        # Load Blenderbot model once (and ONLY once please...)
        tokenizer, model = load_blenderbot_model()

        # Speech-to-text
        c1, c2 = st.columns(2)
        with c1:
            st.write("Convert speech to text:")
        with c2:
            user_text = speech_to_text(language=lang, use_container_width=True, just_once=True, key='STT')

        if user_text:
            # Translate and display (this BETTER work)
            translated_user_text = translate_text(lang, "en", user_text)
            display_user_message(user_text, translated_user_text)
            #st.write(f"You: {user_text} (Translated: {translated_user_text})")

            with st.spinner('Pondering...'):

                # Response
                ai_response = generate_response(translated_user_text, tokenizer, model)

            # AI Back to language
                translated_ai_response = translate_text("en", lang, ai_response)
                display_ai_message(ai_response,  translated_ai_response)
            #st.write(f"PolyProse: {ai_response} (Translated: {translated_ai_response})")

            # Display user message and translation



        if st.button("Refresh"):
            st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Second Tab - About Me
with tabs[1]:
    st.header("About Me")
    bio = st.markdown("""
        Hi! My name is Nya Feinstein :blush: \n
        I am a senior at West Virginia University studying Data Science, Russian Studies, International Studies,
        with a minor in French and a certificate in Global Competency with plans to pursue a doctoral degree.
        My research revolves around all things Natural Language Processing, and
         I believe that the secrets of the world are woven within our words and can best be understood through machine learning
        and genuine curiosity.
        My languages of interest are Belarusian, Russian, Polish, and French (and will happily invite more). What are yours?
        Contact me at nyafein@gmail.com or visit me on LinkedIn to let me know!    """)
    st.image("./headshot.png")

# Third Tab - Sources
with tabs[2]:
    st.header("Sources")

    argosurl = 'https://github.com/argosopentech/argos-translate'

    blenderurl = 'https://huggingface.co/docs/transformers/model_doc/blenderbot#transformers.BlenderbotForCausalLM'

    st.markdown(
        f'<a href={argosurl}><button style="background-color:lightblue; color: black;">📖 Argos Model</button></a>',
        unsafe_allow_html=True)

    # For BlenderBot Model
    st.markdown(
        f'<a href={blenderurl}><button style="background-color:lightblue; color: black;">🤖 BlenderBot Model</button></a>',
        unsafe_allow_html=True)

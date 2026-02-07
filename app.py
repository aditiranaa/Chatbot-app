import streamlit as st
import json
import pickle
import random
import time
from preprocess import clean_text

# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open('intents.json') as f:
    intents = json.load(f)

# ---------- QUICK RULES ----------
def quick_rules(user_input):
    text = user_input.lower()

    if any(w in text for w in ["technical","coding","robot","ai"]):
        return "You can join coding, robotics, AI or cybersecurity societies."

    if any(w in text for w in ["sports","football","cricket","badminton"]):
        return "Sports clubs include football, cricket and badminton teams."

    if "cgpa formula" in text:
        return "CGPA = Total Grade Points ÷ Total Credits."

    if "sgpa formula" in text:
        return "SGPA = Sum of (Grade Point × Subject Credits) ÷ Total Semester Credits."

    return None

# ---------- RESPONSE ----------
def get_response(tag, user_text):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']
            if isinstance(responses, list):
                return random.choice(responses)
            if isinstance(responses, dict):
                return random.choice(responses.get("general", ["I can help with that!"]))
    return "Sorry, I didn’t understand."


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="College Help Chatbot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
# ---------- THEME OVERRIDE ----------
# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("Settings ⚙️")

    theme = st.radio("Theme", ["Light", "Dark"])

    if st.button("Clear Chat 🗑️"):
        st.session_state.messages = []

    st.markdown("---")
    st.write("**About**")
    st.write("College Help Assistant")
    st.write("Built with ML + Streamlit")
    st.write("Built by: Aditi Rana")


# ---------- THEME OVERRIDE ----------
if theme == "Light":
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }

        /* ALL TEXT */
        .stApp * {
            color: #000000 !important;
        }

        /* Chat bubbles */
        [data-testid="stChatMessageContent"] * {
            color: #000000 !important;
        }

        /* Input box */
        textarea, input {
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }

        /* Buttons */
        button {
            color: #000000 !important;
        }
        </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <style>
        .stApp {
            background-color: #000000 !important;
            color: #FFFFFF !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
        }

        .stApp * {
            color: #FFFFFF !important;
        }

        [data-testid="stChatMessageContent"] * {
            color: #FFFFFF !important;
        }

        textarea, input {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
        }

        button {
            color: #FFFFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)





# ---------- THEME STYLE ----------
if theme == "Dark":
    st.markdown("""
    <style>
    body {background-color: #0E1117; color: white;}
    </style>
    """, unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<p class="chat-title">🎓 College Help Assistant</p>', unsafe_allow_html=True)
st.write("Ask me about **CGPA, exams, clubs, internships, or study tips!**")

# ---------- LOGO (OPTIONAL) ----------
# Put logo.png in project folder
# st.image("logo.png", width=120)

# ---------- SESSION MEMORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY OLD MESSAGES ----------
for sender, msg in st.session_state.messages:
    if sender == "You":
        with st.chat_message("user", avatar="😃"):
            st.write(msg)
    else:
        with st.chat_message("assistant", avatar="🎓"):
            st.write(msg)

# ---------- CHAT INPUT ----------
user_input = st.chat_input("Type your question here...")

if user_input:

    with st.chat_message("user", avatar="😃"):
        st.write(user_input)

    # Typing indicator
    with st.chat_message("assistant", avatar="🎓"):
        typing_placeholder = st.empty()
        typing_placeholder.write("Bot is typing...")
        time.sleep(1)

        # QUICK RULE
        rule_answer = quick_rules(user_input)

        if rule_answer:
            bot_reply = rule_answer
        else:
            cleaned = " ".join(clean_text(user_input))
            X = vectorizer.transform([cleaned])

            probs = model.predict_proba(X)
            confidence = max(probs[0])

            if confidence < 0.20:
                bot_reply = "Sorry, I didn’t understand that."
            else:
                tag = model.predict(X)[0]
                bot_reply = get_response(tag, user_input)

        typing_placeholder.empty()
        st.write(bot_reply)

    # SAVE CHAT
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", bot_reply))

# ---------- FOOTER ----------
st.markdown('<p class="footer">Made for Academic Assistance • v1.0</p>', unsafe_allow_html=True)

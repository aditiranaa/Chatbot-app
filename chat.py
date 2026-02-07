import json
import pickle
import random
from preprocess import clean_text

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open('intents.json') as f:
    intents = json.load(f)

# ---------------- QUICK RULES ----------------
# Handles short / single-word inputs before ML
def quick_rules(user_input):
    text = user_input.lower()

    # SOCIETIES
    if any(w in text for w in ["technical", "coding", "robot", "ai"]):
        return "You can join coding, robotics, AI or cybersecurity societies."

    if any(w in text for w in ["sports", "football", "cricket", "badminton"]):
        return "Sports clubs include football, cricket and badminton teams."

    if any(w in text for w in ["cultural", "dance", "music", "drama"]):
        return "Cultural societies include dance, drama and music clubs."

    # CGPA / SGPA
    if "sgpa formula" in text:
        return "SGPA = Sum of (Grade Point × Subject Credits) ÷ Total Semester Credits."

    if "cgpa formula" in text:
        return "CGPA = Total Grade Points ÷ Total Credits."

    if "good cgpa" in text:
        return "CGPA above 7 is considered good, above 8 is very strong."

    return None


# ---------------- KEYWORD HELPER ----------------
def keyword_match(text, words):
    tokens = text.split()
    return any(word in tokens for word in words)


# ---------------- RESPONSE LOGIC ----------------
def get_response(tag, user_text):
    text = user_text.lower()

    for intent in intents['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']

            # ---------- SOCIETIES ----------
            if tag == "societies_clubs":
                if keyword_match(text, ["technical", "coding", "robot", "ai", "tech"]):
                    return random.choice(responses["technical"])
                elif keyword_match(text, ["cultural", "dance", "music", "drama"]):
                    return random.choice(responses["cultural"])
                elif keyword_match(text, ["sports", "football", "cricket", "badminton"]):
                    return random.choice(responses["sports"])
                else:
                    return random.choice(responses["general"])

            # ---------- INTERNSHIP ----------
            if tag == "internship_guidance":
                if "government" in text or "govt" in text:
                    return random.choice(responses["government"])
                elif "private" in text or "company" in text:
                    return random.choice(responses["private"])
                elif "resume" in text:
                    return random.choice(responses["resume"])
                elif "skill" in text or "learn" in text:
                    return random.choice(responses["skills"])
                else:
                    return random.choice(responses["general"])

            # ---------- EXAM ----------
            if tag == "exam_timetable":
                if "mid" in text:
                    return random.choice(responses["midsem"])
                elif "end" in text or "final" in text:
                    return random.choice(responses["endsem"])
                elif "holiday" in text:
                    return random.choice(responses["holiday"])
                elif "time" in text or "class" in text:
                    return random.choice(responses["timetable"])
                else:
                    return random.choice(responses["general"])

            # ---------- CGPA ----------
            if tag == "cgpa_details":
                if "cgpa" in text and "formula" in text:
                    return random.choice(responses["cgpa_formula"])
                elif "cgpa" in text:
                    return random.choice(responses["sgpa_formula"])
                elif "sgpa" in text and "formula" in text:
                    return random.choice(responses["sgpa_formula"])
                elif "sgpa" in text:
                    return random.choice(responses["sgpa_formula"])
                elif "good" in text:
                    return random.choice(responses["good"])
                elif "improve" in text:
                    return random.choice(responses["improve"])
                else:
                    return random.choice(responses["general"])

            # ---------- STUDY ----------
            if tag == "study_tips":
                if "focus" in text:
                    return random.choice(responses["focus"])
                elif "hour" in text:
                    return random.choice(responses["hours"])
                elif "revise" in text or "revision" in text:
                    return random.choice(responses["revision"])
                else:
                    return random.choice(responses["general"])

            # ---------- DEFAULT ----------
            if isinstance(responses, list):
                return random.choice(responses)

            if isinstance(responses, dict):
                return random.choice(responses.get("general", ["I can help with that!"]))

    return "Sorry, I didn’t understand."


# ---------------- CHAT LOOP ----------------
print("College Help Chatbot is running! Type 'exit' to quit.")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break

    # ---- QUICK RULE CHECK ----
    rule_answer = quick_rules(user_input)
    if rule_answer:
        print("Bot:", rule_answer)
        continue

    # ---- PREPROCESS ----
    cleaned = " ".join(clean_text(user_input))
    X = vectorizer.transform([cleaned])

    # ---- CONFIDENCE CHECK ----
    probs = model.predict_proba(X)
    confidence = max(probs[0])

    if confidence < 0.20:
        print("Bot: Sorry, I didn’t understand that.")
        continue

    # ---- PREDICT ----
    tag = model.predict(X)[0]
    response = get_response(tag, user_input)

    print("Bot:", response)

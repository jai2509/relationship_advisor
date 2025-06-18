# app.py - Full Version
import streamlit as st
import openai
import requests
import os
from dotenv import load_dotenv
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from reportlab.pdfgen import canvas
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

st.set_page_config(page_title="AI Relationship Companion", layout="wide", initial_sidebar_state="expanded")
st.title("💖 AI Relationship Companion")

# Load API Keys
groq_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")

# OAuth flow for Google Calendar
def authenticate_google_calendar():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "redirect_uris": ["http://localhost:8501/"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    flow.redirect_uri = "http://localhost:8501/"
    auth_url, _ = flow.authorization_url(prompt='consent')
    return auth_url

st.sidebar.title("🌗 App Navigation")
tab = st.sidebar.radio("Choose Feature", [
    "Relationship Advisor", "Legal Help", "Therapy Chat", "Mood Tracker", 
    "Task Recommender", "Calendar Scheduler", "FAISS Search", "PDF Export", 
    "Compatibility Test", "Daily Check-in", "Analytics", "Meditation", "Date Ideas",
    "CBT Thought Tracker", "EFT Tapping", "Behavior Activation", "Positive Reinforcement"
])

def query_groq(prompt, role="Expert AI"):
    headers = {"Authorization": f"Bearer {groq_key}"}
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "system", "content": role},
                     {"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    return res.json()["choices"][0]["message"]["content"]

# ------------------------ Tabs ------------------------

if tab == "Relationship Advisor":
    st.header("💬 Relationship Advice")
    user_input = st.text_area("Describe your situation:")
    if st.button("Get Advice"):
        reply = query_groq(user_input, role="Relationship Counselor for all types of couples")
        st.success(reply)

elif tab == "Legal Help":
    st.header("⚖️ Legal Assistance for Couples")
    legal_input = st.text_area("Explain your legal issue:")
    if st.button("Ask Legal Advisor"):
        legal_reply = query_groq(legal_input, role="Couples Legal Advisor")
        st.info(legal_reply)

elif tab == "Therapy Chat":
    st.header("🧠 Therapy Session")
    msg = st.text_input("Talk to your AI therapist:")
    if msg:
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": {"past_user_inputs": [], "text": msg}}
        res = requests.post("https://api-inference.huggingface.co/models/microsoft/DialoGPT-large", headers=headers, json=payload)
        reply = res.json().get("generated_text", "I'm here for you.")
        st.write(reply)

elif tab == "Mood Tracker":
    st.header("🧠 Mood Tracker")
    mood = st.text_input("What's your mood today?")
    if st.button("Analyze Mood"):
        res = requests.post(
            "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment",
            headers={"Authorization": f"Bearer {hf_token}"}, json={"inputs": mood}
        )
        st.write("Mood Sentiment:", res.json()[0][0]['label'])

elif tab == "Task Recommender":
    st.header("🎬 Task & Movie Recommender")
    current_mood = st.text_input("What's your current mood?")
    if st.button("Suggest Bonding Task"):
        task_suggestion = query_groq(f"Suggest bonding activities or movies for couples feeling '{current_mood}'", role="Couples Activity Recommender")
        st.success(task_suggestion)

elif tab == "Calendar Scheduler":
    st.header("📅 Schedule Bonding Task")
    st.markdown("[Click here to authorize Google Calendar]({})".format(authenticate_google_calendar()))

elif tab == "FAISS Search":
    st.header("🔁 Search Similar Relationship Case")
    past_cases = [
        "We keep arguing over small things.",
        "Lack of intimacy is hurting us.",
        "Our families don't support our relationship.",
        "Communication has dropped."
    ]
    query = st.text_input("Your case description:")
    if query and st.button("Find Similar Case"):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(past_cases + [query]).toarray()
        index = faiss.IndexFlatL2(len(vectors[0]))
        index.add(vectors[:-1])
        D, I = index.search([vectors[-1]], k=1)
        st.write("Most similar past case:", past_cases[I[0][0]])

elif tab == "PDF Export":
    st.header("📄 Export Advice to PDF")
    advice = st.text_area("Enter advice to export:")
    if st.button("Download PDF"):
        c = canvas.Canvas("advice.pdf")
        c.drawString(100, 750, advice)
        c.save()
        with open("advice.pdf", "rb") as f:
            st.download_button("Download PDF", f, "advice.pdf")

elif tab == "Compatibility Test":
    st.header("💑 Compatibility Quiz")
    q1 = st.radio("How often do you communicate?", ["Rarely", "Sometimes", "Daily"])
    q2 = st.radio("How do you resolve conflicts?", ["Ignore", "Talk", "Blame", "Calm discussion"])
    q3 = st.radio("Do you trust each other?", ["No", "Yes", "Sometimes"])
    score = sum([["Rarely", "Sometimes", "Daily"].index(q1),
                 ["Ignore", "Talk", "Blame", "Calm discussion"].index(q2),
                 ["No", "Yes", "Sometimes"].index(q3)])
    if st.button("Check Compatibility"):
        st.success(f"Compatibility Score: {score}/10")

elif tab == "Daily Check-in":
    st.header("✅ Daily Couple Check-in")
    if "checkins" not in st.session_state:
        st.session_state.checkins = []
    today_mood = st.text_input("How do you feel today (1-5)?")
    if st.button("Check-in"):
        st.session_state.checkins.append(today_mood)
        st.success("Check-in recorded!")

elif tab == "Analytics":
    st.header("📊 Relationship Analytics")
    st.metric("Total Check-ins", len(st.session_state.get("checkins", [])))
    st.metric("Compatibility Score", f"{score}/10" if 'score' in locals() else "N/A")

elif tab == "Meditation":
    st.header("🧘 AI-Guided Relationship Meditation")
    your_mood = st.text_input("Your mood before meditation:")
    if st.button("Start Meditation"):
        result = query_groq(f"My mood is {your_mood}. Please guide us in a calming 2-minute meditation.", role="Mindfulness Relationship Coach")
        st.info(result)

elif tab == "Date Ideas":
    st.header("🛍️ Date Night Idea Generator")
    vibe = st.selectbox("Choose your vibe", ["Romantic", "Fun", "Adventurous", "Budget-friendly"])
    season = st.selectbox("Season", ["Winter", "Summer", "Spring", "Autumn"])
    if st.button("Generate Date Night Ideas"):
        date_ideas = query_groq(f"Suggest 3 {vibe} date night ideas for a couple in {season}", role="Romantic Planner")
        st.success(date_ideas)

# ✅ New Psychological Tools
elif tab == "CBT Thought Tracker":
    st.header("🧠 CBT Thought Record")
    st.write("Use this to understand and reframe negative thoughts.")
    situation = st.text_area("1️⃣ Describe the situation:")
    thoughts = st.text_area("2️⃣ What automatic thoughts came to mind?")
    emotions = st.text_area("3️⃣ What emotions did you feel?")
    distortions = st.multiselect("4️⃣ Any cognitive distortions?", 
        ["All-or-Nothing Thinking", "Overgeneralization", "Catastrophizing", "Mind Reading", "Should Statements"])
    balanced_thought = st.text_area("5️⃣ Write a more balanced alternative thought:")
    if st.button("🧠 Save Thought Record"):
        st.success("✅ Thought recorded and reframed!")

elif tab == "EFT Tapping":
    st.header("👐 EFT (Tapping) for Emotional Relief")
    emotion = st.text_input("What emotion do you want to release (e.g. anxiety, guilt, stress)?")
    if st.button("🎯 Generate Tapping Script"):
        tapping_script = query_groq(
            f"Generate an EFT tapping script for someone feeling {emotion}. Include setup statement and tapping rounds.",
            role="Certified EFT Practitioner")
        st.info(tapping_script)

elif tab == "Behavior Activation":
    st.header("🧭 Behavior Activation Tasks for Mood Lifting")
    mood_input = st.text_input("Enter your current mood (e.g. sad, unmotivated, lonely):")
    if st.button("🎬 Suggest an Activation Task"):
        task = query_groq(
            f"Suggest 2 evidence-based behavior activation activities to help someone feeling '{mood_input}', ideally for couples.",
            role="Behavioral Psychologist")
        st.success(task)

elif tab == "Positive Reinforcement":
    st.header("🎁 Positive Reinforcement Tracker")
    habit = st.text_input("Enter a healthy habit to build (e.g. 'compliment partner daily'):")
    reward = st.text_input("Enter a reward (e.g. 'watch favorite show'):")
    if st.button("🚀 Log Habit & Reward"):
        st.success(f"Awesome! Doing '{habit}' earns you '{reward}'. Repeat for 7 days to form a habit!")

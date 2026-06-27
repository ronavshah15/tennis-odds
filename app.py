import streamlit as st
import pickle
import pandas as pd
from groq import Groq
import json
import re

# Load model and features
with open('tennis_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('features.pkl', 'rb') as f:
    feature_list = pickle.load(f)

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

st.title("🎾 ATP Tennis Odds Predictor")
st.write("Ask me about any ATP match and I'll give you odds and analysis.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("e.g. Sinner vs Alcaraz at Wimbledon, who wins?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            try:
                # Step 1: Extract match details
                extract_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": f"""Extract tennis match details from this text and return ONLY a JSON object with:
                    p1_name, p2_name, 
                    Rank_1 (number), Rank_2 (number),
                    Pts_1 (number), Pts_2 (number),
                    Surface (Hard/Clay/Grass),
                    round_num (1-7 where 7=final),
                    level_num (1=ATP250, 2=ATP500, 3=Masters, 4=Grand Slam),
                    rank_diff (Rank_1 minus Rank_2),
                    pts_diff (Pts_1 minus Pts_2),
                    bookie_p1 (your estimated win probability for p1 as decimal 0-1),
                    bookie_p2 (your estimated win probability for p2 as decimal 0-1),
                    h2h_diff (how many more times p1 has beaten p2, use 0 if unknown),
                    p1_form (recent form score, positive=good, use 5 if unknown),
                    p2_form (recent form score, positive=good, use 5 if unknown)
                    
                    Defaults: Rank=50, Pts=1000, Surface=Hard, round_num=3, level_num=1
                    All values must be numbers not strings.
                    
                    Text: {user_input}"""}]
                )

                raw = extract_response.choices[0].message.content
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                data = json.loads(json_match.group())

                surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
                data['surface_num'] = surface_map.get(data['Surface'], 0)

                features = pd.DataFrame([[
                    float(data['Rank_1']), float(data['Rank_2']),
                    float(data['Pts_1']), float(data['Pts_2']),
                    float(data['surface_num']), float(data['round_num']),
                    float(data['level_num']), float(data['rank_diff']),
                    float(data['pts_diff']), float(data['bookie_p1']),
                    float(data['bookie_p2']), float(data['h2h_diff']),
                    float(data['p1_form']), float(data['p2_form'])]],
                    columns=feature_list)

                prob = model.predict_proba(features)[0]
                p1_prob = round(float(prob[1]) * 100, 1)
                p2_prob = round(float(prob[0]) * 100, 1)
                p1_odds = round(float(1 / prob[1]), 2)
                p2_odds = round(float(1 / prob[0]), 2)

                # Step 2: Generate detailed analysis
                analysis_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                   messages=[{"role": "user", "content": f"""You are a professional tennis analyst.
                    The match is {data['p1_name']} vs {data['p2_name']} on {data['Surface']}.
                    Our model gives {data['p1_name']} a {p1_prob}% chance (odds {p1_odds}) and {data['p2_name']} a {p2_prob}% chance (odds {p2_odds}).
                    
                    STRICT RULES:
                    - Only use facts explicitly provided to you above
                    - Do NOT invent court numbers, venues, crowd details, or any specific facts not given
                    - If you don't know something, do not mention it at all
                    - Only reference the surface, player names, odds, and probabilities given
                    
                    Write a short 2-3 sentence analysis covering:
                    - Who is the favourite and why based on the odds
                    - How the {data['Surface']} surface affects this matchup
                    - A predicted scoreline
                    - One key factor that could swing the match
                    
                    Be confident and specific. No bullet points, just natural flowing text."""}]
                )

                analysis = analysis_response.choices[0].message.content

                result = f"""
**{data['p1_name']}** — {p1_prob}% chance → odds **{p1_odds}**
**{data['p2_name']}** — {p2_prob}% chance → odds **{p2_odds}**

{analysis}
"""
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as e:
                msg = "Couldn't process that match. Try adding more detail like surface, tournament, or rankings."
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

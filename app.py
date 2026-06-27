import streamlit as st
import pickle
import pandas as pd
from groq import Groq
import json
import re

with open('tennis_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('features.pkl', 'rb') as f:
    feature_list = pickle.load(f)

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

st.title("🎾 ATP Tennis Odds Predictor")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.info("💡 For best results include: surface, tournament, and rankings. Example: 'Sinner rank 1 vs Alcaraz rank 3 at Wimbledon on grass'")

user_input = st.chat_input("Ask me about any ATP match...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Step 1: Check if it's a tennis question
                check_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": f"""Is this message asking about a specific tennis match or player matchup? 
                    Reply with only YES or NO.
                    Message: {user_input}"""}]
                )
                
                is_tennis = "YES" in check_response.choices[0].message.content.upper()
                
                if not is_tennis:
                    # Handle non-tennis questions
                    general_response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are a tennis expert chatbot. Answer tennis-related questions helpfully. If asked something completely unrelated to tennis, politely redirect the conversation back to tennis."},
                            {"role": "user", "content": user_input}
                        ]
                    )
                    result = general_response.choices[0].message.content
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                
                else:
                    # Step 2: Extract match details
                    extract_response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": f"""Extract tennis match details from this text and return ONLY a valid JSON object.

IMPORTANT: p1_name and p2_name must be the actual player names from the text, never numbers or empty strings.

Fields required:
- p1_name (string, first player's name)
- p2_name (string, second player's name)
- Rank_1 (number, default 100 if unknown)
- Rank_2 (number, default 100 if unknown)
- Pts_1 (number, default 500 if unknown)
- Pts_2 (number, default 500 if unknown)
- Surface (Hard/Clay/Grass, default Hard)
- round_num (1-7 where 7=final, default 3)
- level_num (1=ATP250, 2=ATP500, 3=Masters, 4=Grand Slam, default 2)
- rank_diff (Rank_1 minus Rank_2)
- pts_diff (Pts_1 minus Pts_2)
- bookie_p1 (estimated win probability for p1 as decimal 0-1)
- bookie_p2 (estimated win probability for p2 as decimal 0-1)
- h2h_diff (how many more times p1 has beaten p2, default 0)
- p1_form (recent form, positive=good, default 5)
- p2_form (recent form, positive=good, default 5)

Text: {user_input}"""}]
                    )

                    raw = extract_response.choices[0].message.content
                    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                    data = json.loads(json_match.group())

                    # Safety check for player names
                    if not data.get('p1_name') or str(data.get('p1_name')) == '0':
                        data['p1_name'] = "Player 1"
                    if not data.get('p2_name') or str(data.get('p2_name')) == '0':
                        data['p2_name'] = "Player 2"

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

                    # Step 3: Generate analysis
                    analysis_response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": f"""You are a professional tennis analyst.
                        The match is {data['p1_name']} vs {data['p2_name']} on {data['Surface']}.
                        Our model gives {data['p1_name']} a {p1_prob}% chance (odds {p1_odds}) and {data['p2_name']} a {p2_prob}% chance (odds {p2_odds}).
                        Player 1 rank: {data['Rank_1']}, Player 2 rank: {data['Rank_2']}.
                        
                        STRICT RULES:
                        - Only use facts explicitly provided above
                        - Do NOT invent court numbers, venues, crowd details, or any specific facts not given
                        - If you don't know something, do not mention it
                        - Use actual player names, never "Player 1" or "Player 2"
                        
                        Write a short 3-4 sentence analysis covering:
                        - Who is the favourite and why based on the odds and rankings
                        - How the {data['Surface']} surface affects this matchup
                        - A predicted scoreline
                        - One key factor that could swing the match
                        
                        No bullet points, just natural flowing text."""}]
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
                msg = "I couldn't process that. Try asking about a specific match like: 'Sinner rank 1 vs Alcaraz rank 3 at Wimbledon on grass'"
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

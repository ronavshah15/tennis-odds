import streamlit as st
import pickle
import pandas as pd
from groq import Groq
import json
import re

with open('tennis_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎾 ATP Tennis Odds Predictor")
st.write("Type a match below and get betting odds instantly.")

api_key = st.text_input("Enter your Groq API key", type="password")
match_input = st.text_area("Describe the match", placeholder="e.g. Djokovic rank 1 vs Alcaraz rank 3, Roland Garros final on clay, Alcaraz leads h2h 3-2")

if st.button("Get Odds") and api_key and match_input:
    with st.spinner("Calculating..."):
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"""Extract tennis match details from this text and return ONLY a JSON object with these fields:
            - p1_name, p1_rank, p1_points, p1_surface_wr
            - p2_name, p2_rank, p2_points, p2_surface_wr
            - surface, h2h_diff, p1_form, p2_form, round_num, tourney_level
            
            Defaults if unknown: rank=50, points=1000, surface_wr=0.5, h2h_diff=0, form=5, round_num=3, tourney_level=A, surface=Hard
            All numeric fields must be numbers not strings.
            
            Text: {match_input}"""}]
        )
        
        raw = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(json_match.group())
        
        surface_num = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}.get(data['surface'], 0)
        level_num = {'D': 0, 'A': 1, 'M': 2, 'G': 3, 'F': 4}.get(data['tourney_level'], 1)
        
        features = pd.DataFrame([[
            float(data['p1_rank']), float(data['p2_rank']),
            float(data['p1_points']), float(data['p2_points']),
            float(surface_num), float(data['h2h_diff']),
            float(data['p1_form']), float(data['p2_form']),
            float(data['round_num']), float(data['p1_surface_wr']),
            float(data['p2_surface_wr']), float(level_num)]],
            columns=['p1_rank', 'p2_rank', 'p1_points', 'p2_points',
                     'surface_num', 'h2h_diff', 'p1_form', 'p2_form', 'round_num',
                     'p1_surface_wr', 'p2_surface_wr', 'tourney_level_num'])
        
        prob = model.predict_proba(features)[0]
        p1_prob = round(float(prob[1]) * 100, 1)
        p2_prob = round(float(prob[0]) * 100, 1)
        p1_odds = round(float(1 / prob[1]), 2)
        p2_odds = round(float(1 / prob[0]), 2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=data['p1_name'], value=f"odds {p1_odds}", delta=f"{p1_prob}% chance")
        with col2:
            st.metric(label=data['p2_name'],

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and stats
with open("score_predictor_model.pkl", "rb") as f:
    model = pickle.load(f)

best_batsman = pd.read_csv("stats_best_batsman.csv")
best_bowler = pd.read_csv("stats_best_bowler.csv")
matches = pd.read_csv("matches.csv")

st.set_page_config(layout="wide")
st.title("🏏 IPL Score Predictor & Dashboard")

tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Predict Score"])

# ---------- Tab 2: Score Prediction ----------
with tab2:
    st.header("🔮 Predict First Innings Score (Runs, 4s & 6s)")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team A", sorted(matches['team1'].dropna().unique()), key="team1")
        team2 = st.selectbox("Select Team B", sorted(matches['team2'].dropna().unique()), key="team2")

    with col2:
        toss_winner = st.selectbox("Toss Winner", [team1, team2], key="toss_winner")
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"], key="toss_decision")

    venue = st.selectbox("Venue", sorted(matches["venue"].dropna().unique()), key="venue")

    # Determine batting and bowling teams
    if toss_decision == "bat":
        batting_team = toss_winner
        bowling_team = team2 if toss_winner == team1 else team1
        role = "bat"
    else:
        bowling_team = toss_winner
        batting_team = team2 if toss_winner == team1 else team1
        role = "bowl"

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "venue": venue,
            "our_team": batting_team,
            "opponent_team": bowling_team,
            "role": role
        }])
        prediction = model.predict(input_df)[0]
        st.subheader(f"Predicted Score for {batting_team} 🏏")
        st.write(f"Estimated Total Runs: **{round(prediction)}**")
        st.write("Note: This prediction assumes typical match conditions for a first innings.")

# ---------- Tab 1: Dashboard ----------
with tab1:
    st.header("📊 IPL Dashboard: Team & Match Insights")

    col1, col2, col3 = st.columns(3)
    with col1:
        venue_stats = matches.groupby("venue")["target_runs"].mean().sort_values(ascending=False).head(10)
        st.subheader("🏟️ Avg Runs at Top Venues")
        st.bar_chart(venue_stats)

    with col2:
        team_wins = matches["winner"].value_counts().head(10)
        st.subheader("🏆 Most Wins by Teams")
        st.bar_chart(team_wins)

    with col3:
        toss_wins = matches["toss_winner"].value_counts().head(10)
        st.subheader("🎲 Most Toss Wins")
        st.bar_chart(toss_wins)

    st.subheader("🧢 Toss Decisions")
    toss_decision_count = matches["toss_decision"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(toss_decision_count, labels=toss_decision_count.index, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("🔥 Top 10 Batsmen")
        st.dataframe(best_batsman)

    with col5:
        st.subheader("💥 Top 10 Bowlers")
        st.dataframe(best_bowler)

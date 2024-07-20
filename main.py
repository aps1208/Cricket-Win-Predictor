import base64
import streamlit as st
import pickle
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
@st.cache_data
def get_img_as_base64(file):
    with open(file,"rb") as f:
        data=f.read()
    return base64.b64encode(data).decode()
img = get_img_as_base64(r"C:\Users\aman2\Desktop\DA PROJECT\background.jpg")
page_bg_img=f"""
    <style>
    [data-testid="stAppViewContainer"] >.main{{
    background-image: url("data:image/jpg;base64,{img}");
    width: 100%;
    height: 100%;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-size: cover;
    }}
    [data-testid="stSidebar"] > div:first-child{{
    background-image: url("data:image/jpg;base64,{img}");
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    [data-testid="stToolbar"]{{
    right: 2rem;
    }}
    </style>
    """
teams=["--select--","Mumbai Indians","Chennai Super Kings","Royal Challengers Banglore",
       "Delhi Capitals","Kolkata Knight Riders","Sunrisers Hyderabad","Punjab Kings","Rajasthan Royals"]

venues=['--select--','Bangalore', 'Hyderabad', 'Kolkata', 'Mumbai', 'Visakhapatnam',
       'Indore', 'Durban', 'Chandigarh', 'Delhi', 'Dharamsala',
       'Ahmedabad', 'Chennai', 'Ranchi', 'Nagpur', 'Mohali', 'Pune',
       'Bengaluru', 'Jaipur', 'Port Elizabeth', 'Centurion', 'Raipur',
       'Sharjah', 'Cuttack', 'Johannesburg', 'Cape Town', 'East London',
       'Abu Dhabi', 'Kimberley', 'Bloemfontein']

pipe=pickle.load(open('pipe.pkl','rb'))

st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(""" # **CRICKET WIN PREDICTOR** """)

col1,col2=st.columns(2)
with col1:
    batting_team=st.selectbox("Select Batting Team",teams)
    
with col2:
    if(batting_team=="--select--"):
        bowling_team=st.selectbox("Select Bowling Team",teams)
    else:
        filtered_teams=[team for team in teams if team!=batting_team]
        bowling_team=st.selectbox("Select Bowling Team",filtered_teams)
        
selected_venues=st.selectbox("Select venue",venues)
target=st.number_input("Target")

col1,col2,col3=st.columns(3)

with col1:
    score=st.number_input("Score",min_value=0)
    
with col2:
    overs=st.number_input("Overs Completed",max_value=20)
    
with col3:
    wickets=st.number_input("Wickets Down",max_value=10)
    
if st.button("PREDICT WINNING PROBABILITY"):
    col1,col2=st.columns(2)
    loss=0
    win=0
    with col1:
        runs_left=target-score
        balls_left=120-(overs*6)
        wickets=10-wickets
        CRR=score/overs
        RRR=runs_left/(balls_left/6)
    
        input_data=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],
                             'venue':[selected_venues],'runs_left':[runs_left],'balls_left':[balls_left],
                             'wickets_remaining':[wickets],'total_runs_x':[target],'CRR':[CRR],'RRR':[RRR]})
    
        result=pipe.predict_proba(input_data)
    
        loss = result[0][0]
        win = result[0][1]
        st.header(batting_team + ": "+str(round(win*100))+"%")
        st.header(bowling_team + ": "+str(round(loss*100))+"%")
    with col2:
        # Sample data (replace this with your actual data)
        labels = [batting_team,bowling_team]
        sizes = [win,loss] # percentages
        team_colors={'Mumbai Indians':'blue','Chennai Super Kings':'yellow','Royal Challengers Banglore':'red',
       'Delhi Capitals':'cyan','Kolkata Knight Riders':'purple','Sunrisers Hyderabad':'orange',
       'Punjab Kings':'gold','Rajasthan Royals':'pink'}
        # Create a pie chart
        fig, ax = plt.subplots()
        ax.set_facecolor("none")
        fig.patch.set_facecolor('none')
        ax.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, colors=[team_colors[lable]for lable in labels])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # Display the chart using Streamlit
        st.pyplot(fig)
import pandas as pd
first_dataset = pd.read_csv("C:\\Users\\aman2\\Desktop\\DA PROJECT\\FIRST DATASET.csv")
second_dataset = pd.read_csv("C:\\Users\\aman2\\Desktop\\DA PROJECT\\SECOND DATASET.csv")

# calculating the total score of first innings of each match
total_score=second_dataset.groupby(['match_id','inning']).sum()['total_runs'].add(1).reset_index()
total_score=total_score[total_score['inning']==1]

#creating a new dataframe
match_df=first_dataset.merge(total_score[['match_id','total_runs']],left_on='id',right_on='match_id')

#changing names of teams in the dataset 
match_df['team1']=match_df['team1'].replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].replace('Delhi Daredevils','Delhi Capitals')

match_df['team1']=match_df['team1'].replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].replace('Deccan Chargers','Sunrisers Hyderabad')

match_df['team1']=match_df['team1'].replace('Kings XI Punjab','Punjab Kings')
match_df['team2']=match_df['team2'].replace('Kings XI Punjab','Punjab Kings')

teams = ['Mumbai Indians', 'Chennai Super Kings','Punjab Kings', 'Rajasthan Royals','Kolkata Knight Riders','Delhi Capitals','Royal Challengers Bangalore','Sunrisers Hyderabad']

match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]



match_df=match_df[['id','venue','winner','total_runs']]
match_df=match_df.rename(columns={'id':'match_id'})

#creating a dataframe
second_innings = match_df.merge(second_dataset,on='match_id')

second_innings=second_innings[second_innings['inning']==2]

#Changing names of teams
second_innings['batting_team'] = second_innings['batting_team'].replace("Delhi Daredevils","Delhi Capitals")
second_innings['bowling_team'] = second_innings['bowling_team'].replace("Delhi Daredevils","Delhi Capitals")

second_innings['batting_team'] = second_innings['batting_team'].replace("Deccan Chargers","Sunrisers Hyderabad")
second_innings['bowling_team'] = second_innings['bowling_team'].replace("Deccan Chargers","Sunrisers Hyderabad")

second_innings['batting_team'] = second_innings['batting_team'].replace("Kings XI Punjab","Punjab Kings")
second_innings['bowling_team'] = second_innings['bowling_team'].replace("Kings XI Punjab","Punjab Kings")

#Removing unwanted teams
teams = ['Mumbai Indians', 'Chennai Super Kings','Punjab Kings', 'Rajasthan Royals','Kolkata Knight Riders','Delhi Capitals','Royal Challengers Bangalore','Sunrisers Hyderabad']
second_innings=second_innings[second_innings['batting_team'].isin(teams)]
second_innings=second_innings[second_innings['bowling_team'].isin(teams)]


#Calculating current score and runs left after each ball
second_innings['current_score'] = second_innings.groupby('match_id')['total_runs_y'].cumsum()
second_innings['runs_left'] = second_innings['total_runs_x'] - second_innings['current_score']


#Calculating balls left in the innings
second_innings['balls_left']=126-(second_innings['over']*6 + second_innings['ball'])

#CALCULATION OF WICKETS

second_innings['player_dismissed']=second_innings['player_dismissed'].fillna(0)
second_innings['player_dismissed']=second_innings['player_dismissed'].apply(lambda x:x if x==0 else 1)
second_innings['player_dismissed']=second_innings['player_dismissed'].astype('int')
wickets=second_innings.groupby('match_id')['player_dismissed'].cumsum()
second_innings['wickets_remaining']=10-wickets

#CALCULATING CURRENT RUN RATE
second_innings['CRR']=round((second_innings['current_score']*6)/(120-second_innings['balls_left']),2)

#CALCULATING REQUIRED RUN RATE
second_innings['RRR']=round((second_innings['current_score']*6)/(second_innings['balls_left']),2)

# FINDING A WINNER
def winner(team):
    return 1 if team['batting_team'] == team['winner'] else 0

second_innings['winner']=second_innings.apply(winner,axis=1)

# CREATING THE FINAL DATASET

final=second_innings[['batting_team','bowling_team','venue','runs_left','balls_left','wickets_remaining','total_runs_x','CRR','RRR','winner']]

final_dataset=final.sample(final.shape[0])

#CLEANING THE FINAL DATASET BY DROPPING NULL VALUES AND OTHER NONUSEFUL VALUES
final_dataset.dropna(inplace=True)
final_dataset=final_dataset[final_dataset['balls_left']!=0]

#STORING FINAL DATASET
final_dataset.to_csv("FINAL DATASET.csv",index=False)

'''
#BUILDING MODEL using random forest classifier
from sklearn.model_selection import train_test_split as tts
x = final_dataset.iloc[:, :-1]
y = final_dataset.iloc[:,-1]
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.25,random_state=100)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import pickle
trf = ColumnTransformer([('trf',OneHotEncoder(sparse=False,drop='first', handle_unknown='ignore'),['batting_team','bowling_team','venue'])],remainder='passthrough')

pipe = Pipeline([('step1',trf),('step2',RandomForestClassifier())])
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
pickle.dump(pipe,open('pipe.pkl','wb'))

'''

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import pickle

# Building the model using decision tree
x = final_dataset.iloc[:, :-1]
y = final_dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)

trf = ColumnTransformer([('trf', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'),
                          ['batting_team', 'bowling_team', 'venue'])], remainder='passthrough')

# Create a Decision Tree model pipeline
decision_tree_model = Pipeline([('step1', trf), ('step2', DecisionTreeClassifier())])

# Fit the model
decision_tree_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = decision_tree_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Model Accuracy: {accuracy}")

# Save the model
joblib.dump(decision_tree_model, 'decision_tree_model.pkl')
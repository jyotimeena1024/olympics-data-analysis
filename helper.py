import numpy as np


def fetch_medal_tally(df, year, country):
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    if year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    if year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['Year'] == year) & (medal_df['region'] == country)]

    if flag == 1:
        x = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
    else:
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold',
                                                                                      ascending=False).reset_index()

    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']

    x['Gold'] = x['Gold'].astype('int')
    x['Silver'] = x['Silver'].astype('int')
    x['Bronze'] = x['Bronze'].astype('int')
    x['total'] = x['total'].astype('int')

    return x


def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years,country

def data_over_time(df, col):
    
    nations_over_time = (
        df.drop_duplicates(['Year', col])
          .groupby('Year')
          .size()
          .reset_index(name=col)
    )

    nations_over_time.rename(columns={'Year': 'Edition'}, inplace=True)
    return nations_over_time


def most_successful(df, sport):
    temp_df = df.dropna(subset=['Medal'])

    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    x = (
        temp_df['Name']
        .value_counts()
        .rename_axis('Name')
        .reset_index(name='Medals')
    )

    x = x.head(15).merge(df, on='Name', how='left')[
        ['Name', 'Medals', 'Sport', 'region']
    ].drop_duplicates('Name')

    return x

def yearwise_medal_tally(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()

    return final_df

def country_event_heatmap(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]

    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return pt


def most_successful_countrywise(df, country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df = temp_df[temp_df['region'] == country]

    x = (
        temp_df['Name']
        .value_counts()
        .rename_axis('Name')
        .reset_index(name='Medals')
    )

    x = x.head(10).merge(df, on='Name', how='left')[
        ['Name', 'Medals', 'Sport']
    ].drop_duplicates('Name')

    return x

def weight_v_height(df,sport):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    athlete_df['Medal'].fillna('No Medal', inplace=True)
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df

def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)

    final.fillna(0, inplace=True)

    return final


# -------------------------------------------------------------------------
# MACHINE LEARNING MODEL (LOGISTIC REGRESSION)
# -------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

def train_model(df):
    """
    Trains a simple Logistic Regression model.
    Features: Age, Height, Weight, Sex
    Target: Medal (1 if won, 0 if not)
    """
    # Create a copy for ML to avoid changing the original dataframe
    data = df[['Age', 'Height', 'Weight', 'Sex', 'Medal']].copy()
    
    # Drop rows where Age, Height, or Weight is missing
    data = data.dropna(subset=['Age', 'Height', 'Weight', 'Sex'])
    
    # Convert Sex to numbers: Male = 1, Female = 0
    data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
    
    # Convert Medal to numbers: 1 if won any medal, 0 if no medal
    data['Medal'] = data['Medal'].notna().astype(int)
    
    # Features (X) and Target (y)
    X = data[['Age', 'Height', 'Weight', 'Sex']]
    y = data['Medal']
    
    # Train Logistic Regression
    model = LogisticRegression()
    model.fit(X, y)
    
    return model

def predict_medal(model, age, height, weight, sex):
    """
    Predicts the probability of winning a medal.
    """
    # Convert inputs to the format model expects
    sex_encoded = 1 if sex == 'Male' else 0
    
    # Create input array
    input_data = [[age, height, weight, sex_encoded]]
    
    # Predict probability (returns [prob_no_medal, prob_yes_medal])
    prob = model.predict_proba(input_data)[0][1] 
    
    return prob
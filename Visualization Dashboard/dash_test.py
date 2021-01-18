import os
import base64

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
from urllib.request import urlopen
from sklearn.cluster import KMeans
from dash.dependencies import Input, Output, State
import plotly.express as px
import json
import dash
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import requests
import io
from sklearn.preprocessing import MinMaxScaler

from dash_extensions import Download
from dash_extensions.snippets import send_data_frame

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

covid_livedat = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv"
s = requests.get(covid_livedat).content

covid_dat = pd.read_csv(io.StringIO(s.decode('utf-8')), converters={'fips': lambda x: str(x)})

covid_dat.drop(covid_dat.columns[[6, 7, 8, 9]], axis=1, inplace=True)

covid_dat.county = covid_dat.county + " County"

indexer = covid_dat[covid_dat.county == 'Oglala Lakota County'].index
covid_dat.loc[indexer, 'fips'] = 46113

us_state_abbrev = {'01': 'AL',
                   '02': 'AK',
                   '04': 'AZ',
                   '05': 'AR',
                   '06': 'CA',
                   '08': 'CO',
                   '09': 'CT',
                   '10': 'DE',
                   '12': 'FL',
                   '13': 'GA',
                   '15': 'HI',
                   '16': 'ID',
                   '17': 'IL',
                   '18': 'IN',
                   '19': 'IA',
                   '20': 'KS',
                   '21': 'KY',
                   '22': 'LA',
                   '23': 'ME',
                   '24': 'MD',
                   '25': 'MA',
                   '26': 'MI',
                   '27': 'MN',
                   '28': 'MS',
                   '29': 'MO',
                   '30': 'MT',
                   '31': 'NE',
                   '32': 'NV',
                   '33': 'NH',
                   '34': 'NJ',
                   '35': 'NM',
                   '36': 'NY',
                   '37': 'NC',
                   '38': 'ND',
                   '39': 'OH',
                   '40': 'OK',
                   '41': 'OR',
                   '42': 'PA',
                   '44': 'RI',
                   '45': 'SC',
                   '46': 'SD',
                   '47': 'TN',
                   '48': 'TX',
                   '49': 'UT',
                   '50': 'VT',
                   '51': 'VA',
                   '53': 'WA',
                   '54': 'WV',
                   '55': 'WI',
                   '56': 'WY'}

subset = pd.DataFrame()

fig1 = px.choropleth_mapbox(covid_dat, geojson=counties, locations='fips', color='cases',
                            hover_name='county',
                            range_color=(0, 33000),
                            hover_data={'fips': False, 'cases': True},
                            color_continuous_scale="Inferno_r",
                            mapbox_style="carto-positron",
                            zoom=3.2, center={"lat": 38.0902, "lon": -95.7129},
                            opacity=0.8,
                            labels={'cases': 'Number of Cases'}
                            )

fig1.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

fig1.update_layout(coloraxis_colorbar=dict(
    tickvals=[0, 5000, 10000, 15000, 20000, 25000, 30000],
    ticktext=[0, "5k", "10k", "15k", "20k", "25k", "30k+"]
))

with open("test.json", "r") as f:
    fig2 = go.Figure(json.load(f))

df_2020 = pd.read_csv('rates_6.csv')

true_roc = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv',
                       dtype={'fips': str})
true_roc['date'] = pd.to_datetime(true_roc['date'])

roc_2021 = true_roc[true_roc['date'].dt.year == 2021]
roc_2021['Month'] = roc_2021['date'].dt.month

def get_roc(FIPS):
    state_info = roc_2021[roc_2021['fips'] == FIPS]
    change_in_time = list(state_info.groupby(['fips', 'Month'])['date'].count())

    first = list(state_info.groupby(['fips', 'Month'])['cases'].first())
    last = list(state_info.groupby(['fips', 'Month'])['cases'].last())
    change_in_cases = [a_i - b_i for a_i, b_i in zip(last, first)]
    roc = [a_i / b_i for a_i, b_i in zip(change_in_cases, change_in_time)]

    return roc


unique_states = roc_2021.fips.unique()
df = pd.DataFrame(unique_states, columns=['FIPS'])
df2 = pd.DataFrame(df['FIPS'].apply(get_roc).values.tolist(), df['FIPS'], )
df2.reset_index(level=0, inplace=True)
df2.rename(
    columns={0: "Jan 2021", 1: "Feb 2021", 2: "Mar 2021", 3: "Apr 2021", 4: "May 2021", 5: "Jun", 6: "Jul", 7: "Aug",
             8: "Sept", 9: "Oct", 10: "Nov", 11: 'Dec'}, inplace=True)
df_2021 = df2.set_index('FIPS').stack().reset_index(name='ROC').rename(columns={'level_1': 'Month'})

df_2021['State'] = df_2021['FIPS'].map(us_state_abbrev)
df_2021.drop('FIPS', axis=1, inplace=True)
df_2021.dropna(inplace=True)

frames = [df_2020, df_2021]
result = pd.concat(frames)

fig3 = px.choropleth(result,
                     locations='State',
                     color="ROC",
                     animation_frame="Month",
                     locationmode='USA-states',
                     scope="usa",
                     height=670,
                     color_continuous_scale="Inferno_r",
                     range_color=(0, 10500),
                     labels={'ROC': 'Avg Rate of Change'}
                     )

fig3.update_layout(coloraxis_colorbar=dict(
    tickvals=[0, 2000, 4000, 6000, 8000, 10000],
    ticktext=[0, "2k", "4k", "6k", "8k", "10k+"]
))


metrics = pd.read_csv('merged_data_final.csv', converters={'fips': lambda x: str(x)})

merged_data = pd.merge(metrics, covid_dat, on='fips')
#
merged_data['COVID Cases per Capita'] = merged_data['cases'] / merged_data['County Population']

merged_data['COVID Deaths per Capita'] = merged_data['deaths'] / merged_data['County Population']

merged_data.rename(columns={'cases': 'COVID Cases', 'deaths': 'COVID Deaths', 'Total Elderly Count': 'Elderly Count'},
                   inplace=True)

merged_data.set_index(['county', 'fips', 'state', 'date'], inplace=True)

# Let's Normalize all our features
norm = MinMaxScaler()
scaled = norm.fit_transform(merged_data)
scaled_df = pd.DataFrame(scaled, columns=merged_data.columns, index=merged_data.index)

scaled_df.reset_index(level=0, inplace=True)
scaled_df.reset_index(level=0, inplace=True)
scaled_df.reset_index(level=0, inplace=True)
scaled_df.reset_index(level=0, inplace=True)

scaled_df.drop(columns=['date'], inplace=True)
scaled_df.set_index(['county', 'fips', 'state'], inplace=True)

n = 19
missing_dict = {
    'county': ["Slope County", "Billings County", "Oglala Lakota County", "Arthur County", 'McPherson County',
               "Doña Ana County", "Hartley County", "Loving County", "Borden County", "McMullen County",
               "Kenedy County", "King County", "La Salle Parish County", "Suffolk County", "Chesapeake County",
               "Virginia Beach County", "Newport News County", "Hampton County", "Quitman County"],
    'fips': [38087, 38007, 46113, 31005, 31117, 35013, 48205, 48301, 48033, 48311, 48261, 48269, 22059, 51800,
             51550, 51810, 51700, 51650, 13239],
    'state': ['North Dakota', 'North Dakota', 'South Dakota', 'Nebraska', 'Nebraska', 'New Mexico', 'Texas', 'Texas',
              'Texas', 'Texas', 'Texas', 'Texas', 'Louisiana', 'Virginia', 'Virginia', 'Virginia', 'Virginia',
              'Virginia', 'Georgia'],
    'County Population': [0] * n,
    'Elderly Count': [0] * n,
    'Elderly per Capita': [0] * n,
    'Maskless per Capita': [0] * n,
    'ICU Beds': [0] * n,
    'Rate of Change': [0] * n,
    'COVID Cases': [0] * n,
    'COVID Deaths': [0] * n,
    'COVID Cases per Capita': [0] * n,
    'COVID Deaths per Capita': [0] * n,
    'Density per square mile': [0] * n}

missing_counties = pd.DataFrame(missing_dict)
missing_counties.set_index(['county', 'fips', 'state'], inplace=True)
scaled_df = pd.concat([scaled_df, missing_counties])

columns = list(scaled_df.columns)
my_dict = {k: v for v, k in enumerate(columns)}


def rank(subset, rank_by):
    vals = subset.groupby('Cluster').mean()
    sorted_vals = vals.sort_values(by=rank_by)
    subset['Cluster'].replace(list(sorted_vals.index), list(vals.index), inplace=True)
    return subset


image_filename = 'optimal_k.png'  # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
server = app.server

app.title = 'Real Time COVID-19 Dashboard'

app.layout = html.Div(children=[
    html.H1('Real Time COVID-19 Dashboard'),
    html.H4('Created by Andrew Liu, Adhvaith Vijay, and Shail Mirpuri'),
    # All elements from the top of the page
    html.Div([
        html.Br(),
        html.H2(children='Cumulative COVID Cases By County'),

        html.H4(children='Last Updated: ' + covid_dat.date.max()),
        dcc.Markdown('''
        * Hover over map for county-specific information
        '''),
        dcc.Graph(
            figure=fig1
        )
    ]),
    html.Br(),
    html.Div([
        html.H2(children='Elderly Population By County'),
        dcc.Markdown('''
                * Hover over map for county-specific information
                '''),
        dcc.Graph(
            figure=fig2
        ),
    ]),
    html.Br(),
    html.Div([
        html.H2(children='Average Rate Of Change of COVID-19 Cases'),
        dcc.Markdown('''
            * Click play and run the animation for state-specific information. Data is pulled from the 
              New York Times' GitHub repository, so this map updates in real-time.
            
            * The map is interactive and will automatically be updated and display Avg. ROC information for 
              later months in 2021.
        ''', style={"white-space": "pre"}),
        dcc.Graph(
            figure=fig3
        ),
    ]),
    html.Br(),
    html.Div([
        html.H2(children='Clustering Counties By Various Metrics'),
        dcc.Markdown('''
            * Using a clustering technique known as K-Means clustering I aimed to group United States counties 
              based on similar COVID-19 metrics. 
              
            * K-Means clustering works on the principle of having 'x' clusters - or groups - to organize data into. 
              By using the 'kneed' python library and the Elbow Method, this data is most optimally grouped into
              3 distinct clusters based on any given subset of variables. 
        ''', style={"white-space": "pre"}),
        html.Br(),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
        dcc.Markdown('''
            * Having discovered the 'optimal k' (i.e. optimal number of clusters to use), as a user you can select
              variables to cluster by in the 1st dropdown below.
              
            * Similarly you can 'rank' these clusters by some user-defined 'severity metric' in the 2nd dropdown
              below. Doing so will allow you to rank clusters based on your severity index.
              
            * The **figure initially displayed below is the default-loaded map** of counties clustered by the 
              Maskless per Capita metric and ranked by COVID Cases per Capita. Darker regions indicate greater
              severity.
        
            * Please play around with different COVID-19 related metrics and change values as you see fit.
        ''', style={"white-space": "pre"}),

    ]),
    html.Br(),
    html.Div(children=[
        html.Br(),
        html.Label(["Select One or More Variables to Cluster By",
                    dcc.Dropdown(id="user_input",
                                 options=[{'label': k, 'value': k} for k in columns],
                                 value=['Maskless per Capita'],
                                 clearable=False,
                                 multi=True)]),
        html.Br(),
        html.Label(["Select Metric to Rank Clusters by (Severity Index)",
                    dcc.Dropdown(id="rank_by",
                                 options=[{'label': k, 'value': k}
                                          for k in columns],
                                 value='COVID Cases per Capita',
                                 clearable=False,
                                 multi=False)]),
        dcc.Markdown('''
        * Click Submit to create map of clustered counties 
          (takes 10-15 seconds to generate)
            
        * Download the results of clustering **only** after generating 
          the map (wait a few seconds after generating map)
        ''', style={"white-space": "pre"}),
        html.Br(),
        dbc.Button(id='my_button', n_clicks=0, children="Submit", color="primary"),
        html.Br(),
        html.Br(),
        dcc.Loading(id="loading-1",
                    children=[dcc.Graph(id='graph-output', figure={})],
                    type="cube"),
    ], style={'width': '97%'}),
    html.Br(),
    html.Div([dbc.Button("Download Results of Clustering", id="btn", color="success"), Download(id="download")]),
    html.Br()
])


# ------------------------------------------------------------------------------
@app.callback(
    Output(component_id='graph-output', component_property='figure'),
    Input(component_id='my_button', component_property='n_clicks'),
    [State(component_id='user_input', component_property='value'),
     State(component_id='rank_by', component_property='value')],
    prevent_initial_call=False
)
def get_clusters(n, user_input, rank_by):
    a1 = list(user_input)
    a2 = list([rank_by])
    temp = list(set(a1 + a2))
    user_input = [my_dict[x] for x in temp]

    values = scaled_df.iloc[:, user_input]

    kmeans = KMeans(n_clusters=3).fit(values)
    values['Cluster'] = kmeans.labels_
    values['Cluster'] = values['Cluster'].astype(int) + 1

    global subset
    subset = rank(values, rank_by)

    subset.reset_index(level=0, inplace=True)
    subset.reset_index(level=0, inplace=True)
    subset.reset_index(level=0, inplace=True)

    fig = px.choropleth_mapbox(subset, geojson=counties, locations='fips', color='Cluster',
                               color_continuous_scale="Plasma_r",
                               hover_name='county',
                               hover_data={'fips': False, 'Cluster': True},
                               mapbox_style="carto-positron",
                               zoom=3.2, center={"lat": 38.0902, "lon": -95.7129},
                               opacity=0.8,
                               labels={'state': 'State'}
                               )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig.update_layout(coloraxis_colorbar=dict(
        tickvals=[1, 1.5, 2, 2.5, 3],
        ticktext=[1, "", 2, "", 3]
    ))

    return fig


@app.callback(Output("download", "data"), [Input("btn", "n_clicks")], prevent_initial_call=True)
def generate_csv(n_nlicks):
    sorted_df = subset.sort_values(by='Cluster', ascending=False).reset_index(drop=True)
    columns_of_interest = sorted_df[['county', 'state', 'Cluster']]
    return send_data_frame(columns_of_interest.to_csv, filename="cluster_results.csv")


if __name__ == '__main__':
    app.run_server(debug=True)

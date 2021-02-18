import os

import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(
    children=[
        html.H1(
            children="Illinois Institute of Technology",
            style={"textAlign": "center", "color": "#fa1b02"},
        ),
        html.Div(
            children="""
        Designed to find the best policy for reopening(SafeGraph dataset).
    """,
            style={"textAlign": "center", "color": "#ff7d6f"},
        ),
         html.Div([
    
        html.Br(),
        html.Br(),

        dcc.Dropdown(id='County',
            options=[
                     {'label': 'IL-Cook', 'value': 'IL-Cook'},
                     {'label':'TX-Dallas','value':'TX-Dallas'},
                     {'label':'NY-New York','value':'NY-New York'},
                     {'label':'CA-Los Angeles','value':'CA-Los Angeles'},
                     {'label':'GA-Fulton','value':'GA-Fulton'},
                     {'label':'MN-Hennepin','value':'MN-Hennepin'},
                     {'label':'NV-Clark','value':'NV-Clark'},
                     {'label':'NJ-Hudson','value':'NJ-Hudson'},
                     {'label':'NJ-Bergen','value':'NJ-Bergen'},
                     {'label':'LA-Jefferson','value':'LA-Jefferson'},
                     {'label':'OH-Franklin','value':'OH-Franklin'},
                     {'label':'PA-Philadelphia','value':'PA-Philadelphia'},
                     {'label':'NC-Mecklenburg','value':'NC-Mecklenburg'},
                     {'label':'TN-Shelby','value':'TN-Shelby'},
                     {'label':'WI-Milwaukee','value':'WI-Milwaukee'},
                     {'label':'VA-Fairfax','value':'VA-Fairfax'},
                     {'label':'AZ-Maricopa','value':'AZ-Maricopa'},
                     {'label':'CA-Riverside','value':'CA-Riverside'},
                     {'label':'FL-Broward','value':'FL-Broward'},
                     {'label':'FL-Miami-Dade','value':'FL-Miami-Dade'},
                     {'label':'MA-Middlesex','value':'MA-Middlesex'},
                     {'label':'TX-Harris','value':'TX-Harris'},
                     {'label':'UT-Salt Lake','value':'UT-Salt Lake'},
                     
                     
            ],
            optionHeight=35,                   
            value='IL-Cook',                    
            disabled=False,                     
            multi=False,                        
            searchable=True,                    
            search_value='',                    
            placeholder='Choose County...',     
            clearable=True,                     
            style={'width':"50%"}          
            ),
        html.Label('Fixed School beta: ',style={'display':'inline-block','margin-right': '15px'}),
        dcc.RadioItems(id="fsb",options=[
        {'label': 'Yes', 'value': 'yes'},
        {'label': 'No', 'value': 'no'}
    ],value='yes',style={})                                  
    ]),html.Br(),
    html.Div([html.Label('D/N :',style = {'display':'inline-block','margin-right':'120px'}),
              html.Label('City Testing Rate :',style = {'display':'inline-block','margin-right':'45px','margin-left':'7px'}),
              html.Label('Vaccination Rate :',style = {'display':'inline-block','margin-right':'45px'}),
              html.Label('Testing Cost :',style = {'display':'inline-block','margin-right':'50px'})]),
    html.Div([ dcc.Input(id="D/N",type='number',placeholder="D/N...",
            min=0, max=1, step=0.1,style = {'width':'10%','display':'inline-block'},className='six columns',value = 1 / 2
        ),     #dcc.Input(id="sbm",type='number',placeholder="school beta multiplier...",
            #min=10, max=100, step=3,style = {'width':'10%','display':'inline-block','margin-left':'15px'},className='six columns'), 
            #dcc.Input(id="cbm",type='number',placeholder="city beta multiplier...",
           # min=10, max=100, step=3,style = {'width':'10%','display':'inline-block','margin-left':'15px'}),
            dcc.Input(id="tr",type='number',placeholder="testing rate...",
            min=0, max=1, step=0.00000001,style = {'width':'10%','display':'inline-block','margin-left':'15px'},value=0.02),
            dcc.Input(id="vr",type='number',placeholder="vaccination rate...",
            min=0, max=1, step=0.00000001,style = {'width':'10%','display':'inline-block','margin-left':'15px'},value = 0.003),
            dcc.Input(id="tc",type='number',placeholder="testing cost...",
            min=10, max=100, step=1,style = {'width':'10%','display':'inline-block','margin-left':'15px'},value = 25)
            ]),html.Br(),
    html.Div([
        html.Label('Hospitalization Cost :',style = {'display':'inline-block','margin-right':'10px'}),
        dcc.Input(id="HC1",type='number',placeholder="Hospitalization cost...",
            min=1, max=1000000, step=1,style = {'width':'10%','display':'inline-block','margin-left':'15px'},value = 5500),
        dcc.Input(id="HC2",type='number',placeholder="Hospitalization cost...",
            min=1, max=1000000, step=1,style = {'width':'10%','display':'inline-block','margin-left':'15px'},value = 8500),
        dcc.Input(id="HC3",type='number',placeholder="Hospitalization cost...",
            min=1, max=1000000, step=1,style = {'width':'10%','display':'inline-block','margin-left':'15px'},value = 11600)
        ]),html.Br(),
    #html.Div(html.H4(id='output-container',style={'display':'inline-block','margin-right':'20%'})),
        
         html.Div([dcc.Checklist(id='checkbox1',options = [{'label':'Active Infection','value':'I'},
                                                     {'label':'Active hospitilization','value':'IH'},
                                                     {'label':'Cummulative hospitilization','value':'GH'}],value = ['IH']), 
           # dcc.Input(id="t_rate",type='number',placeholder="testing rate...",
            #min=0, max=1, step=0.1,style = {'width':'10%','display':'inline-block','margin-left':'15px'},value = 0),
            #dcc.Graph(id='fig2',figure = compare('IH','IL-Cook',0.2,0.0015,1/2),),
            #dcc.Graph(id="fig",figure = school_testing_cost1(0.0015,0.02,25,1 / 2,'IL-Cook',5500,8500,11600),
             #                                    style={'width':'20%','display':'inline-block'},),
            
                  ]),
         
            
             
         

    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import contextily as cx
import plotly.graph_objects as go
import geopandas as gpd

import os
import argparse
import dash

from dash import dcc, html, Input, Output, State


def visualize_map(nodes_input, color='smoothed', size=None, size_max = None, animation_frame=None, hover_data=[None],
                  colormap = px.colors.diverging.oxy, zoom=15, range_color = None, opacity=1, return_fig=False,
                  title=None, transparent=False, discrete_colormap = px.colors.qualitative.Light24,
                  figsize=(1200,800), renderer=None, mapbox_style='carto-positron', savehtml=None):
    
    columns = [col for col in ['pid', 'easting','northing',color, animation_frame]+hover_data if col is not None]
    columns = list(set(columns))

    if isinstance(size,str):
        columns.append(size)

    nodes = nodes_input[columns].copy()

    nodes_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.easting,nodes.northing), crs='3035')
    nodes_gdf = nodes_gdf.to_crs('4326')

    if animation_frame is not None:
        nodes_gdf[animation_frame] = nodes_gdf[animation_frame].astype(str)
        nodes_gdf = nodes_gdf.sort_values([animation_frame, 'pid'])

    if range_color is None:
        range_color = (nodes_gdf[color].min(), nodes_gdf[color].max())
        
    fig = px.scatter_mapbox(nodes_gdf, lat=nodes_gdf.geometry.y, lon=nodes_gdf.geometry.x,
                            hover_name = 'pid', hover_data = hover_data, opacity=opacity,
                            color=color, size=size, size_max = size_max,
                            mapbox_style=mapbox_style, animation_frame=animation_frame,
                            width=figsize[0], height=figsize[1], zoom=zoom, color_discrete_map=discrete_colormap,
                            color_continuous_scale=colormap, range_color=range_color,
                            )
    

    # cbar_y = 0.775 if animation_frame is not None else 0.9
    fig.update_layout(coloraxis={'colorbar': {'title': {'text': ''},
                                            'len':0.5,
                                            # 'y':cbar_y,
                                            'thickness':5
                                            }})
    fig.update_layout(title=title)
    if transparent:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    return fig, nodes_gdf


# INPUT FILE

# Set up argument parser
parser = argparse.ArgumentParser(description='Run Dash app with input data.')
parser.add_argument('input_file', type=str, help='Path to the input Parquet file')
args = parser.parse_args()

# INPUT FILE
df_orig = pd.read_parquet(args.input_file)


# df_orig = pd.read_parquet('/home/vitorro/Repositories/stae/data/interim/df_Oslo.parq')
# df_orig = pd.read_parquet('/home/vitorro/Repositories/stae/data/interim/df_Malmo.parq')
df_orig['label'] = df_orig['ptp'].apply(lambda x: 0 if x < 25 else 1)
df_orig['size'] = np.ones(df_orig.shape[0])
df_orig['size'] = df_orig['size'] + df_orig['label']*5
df_orig['label'] = df_orig['label'].astype(str)


# OUTPUT FILE
output_dir = '/home/vitorro/Repositories/stae/data/interim/'
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

discrete_colormap = px.colors.qualitative.Pastel1
discrete_colormap = {'1':'red','0':'lightgreen'}
size_max=8

fig, df_fig = visualize_map(df_orig, color='label', colormap='Safe', hover_data=['ptp','evo','grad'], transparent=True,
                    size='size', size_max=size_max, return_fig=True, zoom=13, discrete_colormap=discrete_colormap)

col1 = 'ptp'
col2 = 'grad'

# Dash app layout
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='map', figure=fig, style={'border':'10px solid black', 'margin-bottom':'20px'}),
    
    html.Div([
        html.Div([
            html.Label("Select Label Column", style={'font-size': '18px', 'margin-right': '10px'}),
            dcc.Dropdown(
                id='label-column',
                options=[
                    {'label': 'ptp', 'value': 'ptp'},
                    {'label': 'evo', 'value': 'evo'},
                    {'label': 'grad', 'value': 'grad'}
                ],
                value='ptp',
                style={
                    'width': '100px',  # Set the width of the input box
                    'height': '20px',  # Set the height of the input box
                }
            ),
            html.Label("Threshold", style={'font-size': '18px', 'margin-left': '20px', 'margin-right': '10px'}),
            dcc.Input(id='threshold-slider', type="number", value=25, debounce=True,
                    style={
                        'width': '100px',  # Set the width of the input box
                        'height': '30px',  # Set the height of the input box
                        'font-family': 'Palatino, URW Palladio L, serif', 'font-size':'18px'
                    }
            ),
        ], style={'display': 'flex', 'align-items': 'top'}),


        html.Br(),
        html.Br(),

        html.Div(id='col1-range-text', style={'font-size': '20px', 'margin-bottom':'10px'}),
        dcc.RangeSlider(
            id='col1-range',
            min=df_orig[col1].min(),
            max=df_orig[col1].max(),
            step=0.1,
            marks={i: str(np.round(i,2)) for i in np.linspace(df_orig[col1].min(), df_orig[col1].max(), 10)  },
            value=[df_orig[col1].min(), df_orig[col1].max()]
        ),

        html.Div(id='col2-range-text', style={'font-size': '20px', 'margin-bottom':'10px'}),
        dcc.RangeSlider(
            id='col2-range',
            min=df_orig[col2].min(),
            max=df_orig[col2].max(),
            step=0.1,
            # marks={i: str(i) for i in range(int(df_orig[col2].min()), int(df_orig[col2].max()) + 1, 5)},
            marks={i: str(np.round(i,2)) for i in np.linspace(df_orig[col2].min(), df_orig[col2].max(), 10 ) },
            value=[df_orig[col2].min(), df_orig[col2].max()],
        ),
        
        ],
        style={'width': '40%'}  # Adjust the width of the parent div
    ),


   
    html.P("Info:", style={'font-size': '32px', 'color': 'darkgray', 'text-align': 'left'}),
    html.Div(id='info', style={'font-size': '12px', 'color': 'black', 'text-align': 'left'}),
    html.Div(id='range', style={'font-size': '12px', 'color': 'black', 'text-align': 'left'}),
 
    html.Div(
        id='output',
        style={
            'height': '400px',  # Fixed height (adjust as needed)
            'overflowY': 'scroll',  # Make it scrollable if content exceeds height
            'border': '1px solid black',  # Optional: Add border for visibility
            'padding': '10px'  # Optional: Add padding for better visual presentation
        }
    ),
    html.Br(),

    # Text input for filename
    html.Label("Save the selected dataframe", style={'font-size':'32px'}),
    html.Br(),
    dcc.Input(id='filename', type='text', value='Enter dataframe_name.parq', debounce=True,
                style={
                'width': '300px',  # Set the width of the input box
                'height': '20px',  # Set the height of the input box
                'font-size': '20px',  # Increase font size
                'padding': '10px',  # Add padding inside the input box
                'border': '2px solid black'  # Add a border for better visibility
    }),
    # Save button
    html.Button("Save", id='save-button', n_clicks=0,
                 style={'color':'blue', 'background-color':'cyan', 'font-size':'20px', 'font-weight':'normal',
                        'height':'44px', 'padding':'8px', 'border': '2px solid black'}),

    # Output messages
    html.Div(id='output-message'),

    # Hidden Div to store selected data for saving
    dcc.Store(id='selected-data-store')
], style={'font-family': 'Palatino, URW Palladio L, serif'})


@app.callback(
    [Output('map', 'figure'),
     Output('col1-range-text', 'children'),
     Output('col2-range-text', 'children')],
    [Input('col1-range', 'value'),
     Input('col2-range', 'value'),
     Input('label-column', 'value'),
     Input('threshold-slider', 'value')]
)
def filter_map_data(col1_range, col2_range, label_column, threshold):
    # Update the label column based on the selected threshold
    df_fig['label'] = df_fig[label_column].apply(lambda x: 0 if x < float(threshold) else 1)
    df_fig['size'] = np.ones(df_fig.shape[0])
    df_fig['size'] = df_fig['size'] + df_fig['label']*5
    df_fig['label'] = df_fig['label'].astype(str)
    
    # Filter df_orig based on slider selections
    filtered_df = df_fig[
        (df_fig[col1] >= col1_range[0]) & (df_fig[col1] <= col1_range[1]) &
        (df_fig[col2] >= col2_range[0]) & (df_fig[col2] <= col2_range[1])
    ]

    # Create a new figure with the filtered data
    new_fig = px.scatter_mapbox(filtered_df, lat=filtered_df.geometry.y, lon=filtered_df.geometry.x,
                                hover_name='pid', hover_data=['ptp', 'evo', 'grad'], opacity=1,
                                color='label', size='size', size_max=size_max,
                                mapbox_style='carto-positron', zoom=13,
                                color_continuous_scale='Safe', color_discrete_map=discrete_colormap)

    new_fig.update_layout(coloraxis={'colorbar': {'title': {'text': ''},
                                                  'len': 0.5,
                                                  'thickness': 5}})
    new_fig.update_layout(title=fig.layout.title)
    # new_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    col1text = f"{col1}: {np.round(col1_range[0], 1)} - {np.round(col1_range[1], 1)}"
    col2text = f"{col2}: {np.round(col2_range[0], 1)} - {np.round(col2_range[1], 1)}"

    return new_fig, col1text, col2text


@app.callback(
    [Output('output', 'children'),
     Output('selected-data-store', 'data'),
     Output('info', 'children'),
     Output('range', 'children')
     ],
    [Input('map', 'selectedData')]
)
def display_selected_data(selectedData):
    if selectedData is None:
        return ("Select desired points in the map.", None, None, None)

    # # Extract selected point indices
    # selected_points = [point['pointIndex'] for point in selectedData['points']]
    
    # # Get the corresponding rows from the DataFrame
    # selected_df = df_fig.iloc[selected_points].drop(columns=['geometry'])
    # Extract selected point pids
    selected_pids = [point['hovertext'] for point in selectedData['points']]
    
    # Get the corresponding rows from the DataFrame based on pid
    selected_df = df_fig[df_fig['pid'].isin(selected_pids)].drop(columns=['geometry', 'size'])

    # Reorder columns
    columns_order = (['pid', 'easting', 'northing']
                      + [col for col in selected_df.columns if col not in ['pid', 'easting', 'northing', 'label']]
                      + ['label']
    )
    selected_df = selected_df[columns_order]

    info_text = f"{selected_df.pid.nunique()} points"
    range_text = f"{np.ptp(selected_df.easting.values)}m width x {np.ptp(selected_df.northing.values)}m height"

    return html.Pre(selected_df.to_string(index=False)), selected_df.to_dict('records'), info_text, range_text


# Callback to save the selected data when button is clicked
@app.callback(
    Output('output-message', 'children'),
    Input('save-button', 'n_clicks'),
    State('filename', 'value'),
    State('selected-data-store', 'data')
)
def save_selected_data(n_clicks, filename, stored_data):
    if n_clicks > 0:
        if not stored_data:
            return "No data selected to save."
        
        # Convert stored data back to DataFrame
        selected_df = pd.DataFrame(stored_data)

        # Ensure the filename ends with '.parq'
        if not filename.endswith('.parq'):
            filename += '.parq'
        
        # Define the file path
        filepath = os.path.join(output_dir, filename)

        # Save DataFrame to Parquet format
        try:
            selected_df.to_parquet(filepath, index=False)
            return f"Data saved successfully as {filename} in {output_dir}"
        except Exception as e:
            return f"Error saving file: {str(e)}"

    return ""

if __name__ == '__main__':
    app.run(debug=True)

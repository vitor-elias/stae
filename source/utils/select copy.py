import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Sample DataFrame
nodes_gdf = pd.DataFrame({
    'pid': [1, 2, 3, 4, 5],
    'lat': [37.77, 37.78, 37.76, 37.75, 37.74],
    'lon': [-122.42, -122.41, -122.43, -122.44, -122.45],
})

# Create figure with scatter_mapbox
fig = px.scatter_mapbox(nodes_gdf, lat='lat', lon='lon',
                        hover_name='pid', opacity=0.8,
                        color='pid', size_max=15,
                        mapbox_style="carto-positron", zoom=10)

# Dash app layout
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='map', figure=fig),
    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    Input('map', 'selectedData')
)
def display_selected_data(selectedData):
    if selectedData is None:
        return "No points selected."

    # Extract selected point indices
    selected_points = [point['pointIndex'] for point in selectedData['points']]
    
    # Get the corresponding rows from the DataFrame
    selected_df = nodes_gdf.iloc[selected_points]
    return html.Pre(selected_df.to_string(index=False))

if __name__ == '__main__':
    app.run(debug=True)

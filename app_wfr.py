import pandas as pd
import dash
from dash import dcc, html, dash_table
import plotly.express as px
from dash.dependencies import Output, Input
import numpy as np

# Read data
delay_df = pd.read_csv('data-wfr-lvl-delay-time.csv')
goodbad_df = pd.read_csv('data-wfr-lvl-good-bad.txt')

# Merge GVB from goodbad_df into delay_df using WAFER_ID
merged_gvb = goodbad_df[['WAFER_ID', 'GVB']]
delay_df = delay_df.merge(merged_gvb, on='WAFER_ID', how='left')

# Calculate DELAY_TIME for each WAFER_ID
def calculate_delay_time(df):
		results = []
		for wafer_id in df['WAFER_ID'].unique():
			wafer_data = df[df['WAFER_ID'] == wafer_id]
			# Find END_DATE for OPN=194997 and CHAMBER startswith 'CGCH'
			end_rows = wafer_data[(wafer_data['OPN'] == 194997) & (wafer_data['CHAMBER'].str.startswith('CGCH'))]
			start_rows = wafer_data[(wafer_data['OPN'] == 197573) & (wafer_data['CHAMBER'].str.startswith('PC'))]
			if not end_rows.empty and not start_rows.empty:
				# Use the most recent END_DATE and START_DATE
				end_date_idx = pd.to_datetime(end_rows['END_DATE']).idxmax()
				start_date_idx = pd.to_datetime(start_rows['START_DATE']).idxmax()
				end_date = pd.to_datetime(end_rows.loc[end_date_idx]['END_DATE'])
				start_date = pd.to_datetime(start_rows.loc[start_date_idx]['START_DATE'])
				delay_time = (start_date - end_date).total_seconds() / 3600.0
				gvb = wafer_data.iloc[0]['GVB']
				# Get CHAMBER values for both OPNs
				opn_194997_lot = end_rows.loc[end_date_idx]['LOT'] if 'LOT' in end_rows.columns else None
				opn_194997_entity = end_rows.loc[end_date_idx]['ENTITY'] if 'ENTITY' in end_rows.columns else None
				opn_194997_chamber = end_rows.loc[end_date_idx]['CHAMBER'] if 'CHAMBER' in end_rows.columns else None
				opn_197573_entity = start_rows.loc[start_date_idx]['ENTITY'] if 'ENTITY' in start_rows.columns else None
				opn_197573_lot = start_rows.loc[start_date_idx]['LOT'] if 'LOT' in start_rows.columns else None
				opn_197573_chamber = start_rows.loc[start_date_idx]['CHAMBER'] if 'CHAMBER' in start_rows.columns else None
				opn_194997_end_date = end_rows.loc[end_date_idx]['END_DATE'] if 'END_DATE' in end_rows.columns else None
				opn_197573_start_date = start_rows.loc[start_date_idx]['START_DATE'] if 'START_DATE' in start_rows.columns else None
				results.append({
					'WAFER_ID': wafer_id,
					'DELAY_TIME': delay_time,
					'GVB': gvb,
					'OPN_194997_LOT': opn_194997_lot,
					'OPN_194997_ENTITY': opn_194997_entity,
					'OPN_194997_CHAMBER': opn_194997_chamber,
					'OPN_194997_END_DATE': opn_194997_end_date,
					'OPN_197573_ENTITY': opn_197573_entity,
					'OPN_197573_LOT': opn_197573_lot,
					'OPN_197573_CHAMBER': opn_197573_chamber,
					'OPN_197573_START_DATE': opn_197573_start_date
				})
		return pd.DataFrame(results)


plot_df = calculate_delay_time(delay_df)
# Assign 'NA' to WAFER_IDs without GVB match
if 'GVB' in plot_df.columns:
	plot_df['GVB'] = plot_df['GVB'].fillna('NA')

# Jitter the y-axis (OPN_194997_CHAMBER) but keep chamber names on axis
if 'OPN_194997_CHAMBER' in plot_df.columns:
	chamber_map = {chamber: i for i, chamber in enumerate(sorted(plot_df['OPN_194997_CHAMBER'].unique()))}
	plot_df['CHAMBER_NUM'] = plot_df['OPN_194997_CHAMBER'].map(chamber_map)
	np.random.seed(42)
	plot_df['CHAMBER_NUM_JITTER'] = plot_df['CHAMBER_NUM'] + np.random.uniform(-0.05, 0.05, size=len(plot_df))

# Dash app
app = dash.Dash(__name__)
fig = px.scatter(
	plot_df,
	x='DELAY_TIME',
	y='CHAMBER_NUM_JITTER',
	color='GVB',
	hover_data=['WAFER_ID', 'OPN_194997_CHAMBER'],
	title='DELAY_TIME (hrs) vs OPN 194997 CHAMBER (Jittered)'
)
fig.update_yaxes(
	tickvals=list(range(len(chamber_map))),
	ticktext=list(chamber_map.keys())
)

app.layout = html.Div([
	html.H2('Wafer Delay Time vs GVB'),
	html.Label('Include NA (no GVB match) wafers:'),
	dcc.RadioItems(
		id='include-na-radio',
		options=[
			{'label': 'Include NA', 'value': 'include'},
			{'label': 'Exclude NA', 'value': 'exclude'}
		],
		value='exclude',
		inline=True
	),
	dcc.Graph(
		id='delay-gvb-scatter',
		figure=fig
	),
	html.Hr(),
	html.H3('Wafer Delay Table'),
	dash_table.DataTable(
		id='wafer-delay-table',
		columns=[
			{'name': 'WAFER_ID', 'id': 'WAFER_ID'},
			{'name': 'GVB', 'id': 'GVB'},
			{'name': 'DELAY_TIME', 'id': 'DELAY_TIME'},
			{'name': 'OPN 194997 LOT', 'id': 'OPN_194997_LOT'},
			{'name': 'OPN 194997 ENTITY', 'id': 'OPN_194997_ENTITY'},
			{'name': 'OPN 194997 CHAMBER', 'id': 'OPN_194997_CHAMBER'},
			{'name': 'OPN 194997 END_DATE', 'id': 'OPN_194997_END_DATE'},
			{'name': 'OPN 197573 ENTITY', 'id': 'OPN_197573_ENTITY'},
			{'name': 'OPN 197573 LOT', 'id': 'OPN_197573_LOT'},
			{'name': 'OPN 197573 CHAMBER', 'id': 'OPN_197573_CHAMBER'},
			{'name': 'OPN 197573 START_DATE', 'id': 'OPN_197573_START_DATE'}
		],
		data=plot_df[['WAFER_ID', 'GVB', 'DELAY_TIME', 'OPN_194997_LOT', 'OPN_194997_ENTITY', 'OPN_194997_CHAMBER', 'OPN_194997_END_DATE', 'OPN_197573_ENTITY', 'OPN_197573_LOT', 'OPN_197573_CHAMBER', 'OPN_197573_START_DATE']].to_dict('records'),
		page_size=10,
		style_table={'height': '300px', 'overflowY': 'auto'},
		style_cell={'textAlign': 'left'},
		style_header={'fontWeight': 'bold'},
		sort_action='native',
	),
	html.Button('Export to Excel', id='export-excel-btn'),
	dcc.Download(id='download-wafer-table'),
	html.Hr(),
	html.H3('Good vs Bad Wafer Count'),
	dash_table.DataTable(
		id='wafer-summary-table',
		columns=[
			{'name': 'GVB', 'id': 'GVB'},
			{'name': 'Wafer Count', 'id': 'Wafer_Count'}
		],
		data=plot_df.groupby('GVB')['WAFER_ID'].nunique().reset_index().rename(columns={'WAFER_ID': 'Wafer_Count'}).to_dict('records'),
		style_cell={'textAlign': 'left'},
		style_header={'fontWeight': 'bold'},
	)
])


# Callback to export table to Excel
@app.callback(
    Output('download-wafer-table', 'data'),
    Input('export-excel-btn', 'n_clicks'),
    prevent_initial_call=True
)
def export_table_to_excel(n_clicks):
    if n_clicks:
        export_df = plot_df[['WAFER_ID', 'GVB', 'DELAY_TIME', 'OPN_194997_LOT', 'OPN_194997_ENTITY', 'OPN_194997_CHAMBER', 'OPN_194997_END_DATE', 'OPN_197573_ENTITY', 'OPN_197573_LOT', 'OPN_197573_CHAMBER', 'OPN_197573_START_DATE']]
        csv_data = export_df.to_csv(index=False)
        return dict(content=csv_data, filename='wafer_delay_table.csv')
    return None

# Add this callback to handle the radio button
@app.callback(
    Output('delay-gvb-scatter', 'figure'),
    Input('include-na-radio', 'value')
)
def update_scatter_plot(include_na):
    filtered_df = plot_df.copy()
    
    if include_na == 'exclude':
        filtered_df = filtered_df[filtered_df['GVB'] != 'NA']
    
    fig = px.scatter(
        filtered_df,
        x='DELAY_TIME',
        y='CHAMBER_NUM_JITTER',
        color='GVB',
        hover_data=['WAFER_ID', 'OPN_194997_CHAMBER'],
        title='DELAY_TIME (hrs) vs OPN 194997 CHAMBER (Jittered)'
    )
    
    # Only update y-axis if chamber_map exists and has data
    if 'chamber_map' in globals() and chamber_map:
        fig.update_yaxes(
            tickvals=list(range(len(chamber_map))),
            ticktext=list(chamber_map.keys())
        )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)


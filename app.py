import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Read CSV
df = pd.read_csv('data.csv')

# --- Merge Lot-good-bad.csv columns ---
lot_good_bad = pd.read_csv('Lot-good-bad.csv')
df = df.merge(lot_good_bad, on='LOT', how='left')
df['Fab Defect Scans'] = df['Fab Defect Scans'].fillna('NA')  # <-- mark missing as 'NA'
df['Nrows'] = df['Nrows'].fillna('NA')
df['Nlot'] = df['Nlot'].fillna('NA')
df['DiePrep CIM'] = df['DiePrep CIM'].fillna('NA')

# Filter for OPN 194997 and 197573
df_194997 = df[df['OPN'] == 194997].copy()
df_197573 = df[df['OPN'] == 197573].copy()

# Prepare lookup for matching LOTs at OPN=197573
lot_197573 = df_197573.set_index('LOT')

# Calculate DELAY_TIME and add ENTITY for matching LOTs
delay_times = []
entities = []
multi_counts = []
split_values = []
for idx, row in df_194997.iterrows():
    lot = row['LOT']
    lot_rows = df_197573[df_197573['LOT'] == lot]
    multi_count = len(lot_rows)
    if multi_count > 0:
        # Use the most recent LAST_WAFER_END_DATE
        lot_rows = lot_rows.copy()
        lot_rows['LAST_WAFER_END_DATE'] = pd.to_datetime(lot_rows['LAST_WAFER_END_DATE'])
        most_recent_row = lot_rows.loc[lot_rows['LAST_WAFER_END_DATE'].idxmax()]
        end_194997 = pd.to_datetime(row['LAST_WAFER_END_DATE'])
        end_197573 = most_recent_row['LAST_WAFER_END_DATE']
        delay = round((end_197573 - end_194997).total_seconds() / 3600, 1)
        entity = most_recent_row['ENTITY']
        qty_194997 = row['QTY']
        qty_197573 = most_recent_row['QTY']
        split = 'YES' if qty_194997 != qty_197573 else 'NO'
    else:
        delay = None
        entity = None
        split = 'NO'
    delay_times.append(delay)
    entities.append(entity)
    multi_counts.append(multi_count)
    split_values.append(split)

df_194997['DELAY_TIME'] = delay_times
df_194997['ENTITY_197573'] = entities
df_194997['MULTI'] = multi_counts
df_194997['SPLIT'] = split_values

# Chart: DELAY_TIME distribution
df_194997_unique = df_194997.sort_values('LAST_WAFER_END_DATE').dropna(subset=['DELAY_TIME']).drop_duplicates(subset=['LOT'], keep='last')
# Get min and max date for calendar
min_date = pd.to_datetime(df_194997_unique['LAST_WAFER_END_DATE']).min().date()
max_date = pd.to_datetime(df_194997_unique['LAST_WAFER_END_DATE']).max().date()
fig = px.scatter(
    df_194997_unique,
    x='DELAY_TIME',
    y='ENTITY',
    color='Fab Defect Scans',
    color_discrete_map={'Clean': 'green', 'Defects': 'red'},  # custom color mapping
    title='DELAY_TIME vs ENTITY Scatter',
    hover_data=['LOT', 'MULTI', 'SPLIT', 'LOT_ABORT_FLAG', 'QTY', 'DOTPROCESS', 'PRODUCT'] + [col for col in lot_good_bad.columns if col != 'LOT']
)
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('DELAY_TIME Analysis'),
    dcc.DatePickerRange(
        id='date-picker-range',
        min_date_allowed=min_date,
        max_date_allowed=max_date,
        start_date=min_date,
        end_date=max_date
    ),
    html.Button('Reset Dates', id='reset-dates-btn', n_clicks=0),
    dcc.Dropdown(
        id='lot-dropdown',
        options=[{'label': str(lot), 'value': lot} for lot in df_194997_unique['LOT'].unique()],
        placeholder='Select a LOT',
        multi=False
    ),
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Button('Export to Excel', id='export-excel-btn', n_clicks=0),
    dcc.Download(id='download-dataframe-xlsx'),
    html.H2('Summary Statistics by ENTITY'),
    html.Div(id='summary-table'),
    html.H2('LOT Count by ENTITY and Month'),
    dcc.Graph(id='lot-entity-month-plot')
])

from dash.dependencies import Output, Input, State

@app.callback(
    [Output('date-picker-range', 'start_date'), Output('date-picker-range', 'end_date')],
    [Input('reset-dates-btn', 'n_clicks')],
    [State('date-picker-range', 'start_date'), State('date-picker-range', 'end_date')]
)
def reset_dates(n_clicks, start_date, end_date):
    if n_clicks:
        return min_date, max_date
    return start_date, end_date

@app.callback(
    Output('download-dataframe-xlsx', 'data'),
    Input('export-excel-btn', 'n_clicks'),
    State('lot-dropdown', 'value'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date'),
    prevent_initial_call=True
)
def export_to_excel(n_clicks, selected_lot, start_date, end_date):
    if n_clicks:
        filtered_df = df_194997_unique.copy()
        if start_date and end_date:
            filtered_df = filtered_df[(pd.to_datetime(filtered_df['LAST_WAFER_END_DATE']).dt.date >= pd.to_datetime(start_date).date()) &
                                      (pd.to_datetime(filtered_df['LAST_WAFER_END_DATE']).dt.date <= pd.to_datetime(end_date).date())]
        if selected_lot:
            filtered_df = filtered_df[filtered_df['LOT'] == selected_lot]
        return dcc.send_data_frame(filtered_df.to_excel, 'filtered_data.xlsx', index=False)
    return None

@app.callback(
    [Output('scatter-plot', 'figure'), Output('summary-table', 'children'), Output('lot-entity-month-plot', 'figure')],
    [Input('lot-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_figure(selected_lot, start_date, end_date):
    import numpy as np
    filtered_df = df_194997_unique.copy()
    # Prepare Year-Month column
    filtered_df['YEAR_MONTH'] = pd.to_datetime(filtered_df['LAST_WAFER_END_DATE']).dt.to_period('M').astype(str)
    lot_month_counts = filtered_df.groupby(['YEAR_MONTH', 'ENTITY'])['LOT'].count().reset_index()
    month_plot = px.bar(
        lot_month_counts,
        x='YEAR_MONTH',
        y='LOT',
        color='ENTITY',
        barmode='group',
        title='LOT Count by ENTITY and Month',
        labels={'LOT': 'LOT Count', 'YEAR_MONTH': 'Year-Month'}
    )
    month_plot.update_xaxes(tickangle=90)
    # Filter by date range
    if start_date and end_date:
        filtered_df = filtered_df[(pd.to_datetime(filtered_df['LAST_WAFER_END_DATE']).dt.date >= pd.to_datetime(start_date).date()) &
                                  (pd.to_datetime(filtered_df['LAST_WAFER_END_DATE']).dt.date <= pd.to_datetime(end_date).date())]
    # Filter by LOT
    if selected_lot:
        filtered_df = filtered_df[filtered_df['LOT'] == selected_lot]
    jitter_strength = 0.05
    if not filtered_df.empty:
        filtered_df['ENTITY_JITTER'] = filtered_df['ENTITY']
        if np.issubdtype(filtered_df['ENTITY'].dtype, np.number):
            filtered_df['ENTITY_JITTER'] += np.random.uniform(-jitter_strength, jitter_strength, size=len(filtered_df))
        else:
            entity_map = {v: i for i, v in enumerate(sorted(filtered_df['ENTITY'].unique()))}
            filtered_df['ENTITY_NUM'] = filtered_df['ENTITY'].map(entity_map)
            filtered_df['ENTITY_JITTER'] = filtered_df['ENTITY_NUM'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(filtered_df))
    fig = px.scatter(
        filtered_df,
        x='DELAY_TIME',
        y='ENTITY_JITTER',
        color='Fab Defect Scans',
        color_discrete_map={'Clean': 'green', 'Defects': 'red'},  # custom color mapping
        title='DELAY_TIME vs ENTITY Scatter (Y Jittered)',
        hover_data=['LOT', 'MULTI', 'SPLIT', 'LOT_ABORT_FLAG', 'QTY', 'DOTPROCESS', 'ENTITY'] + [col for col in lot_good_bad.columns if col != 'LOT']
    )
    fig.update_xaxes(title_text='DELAY_TIME (hours)')
    if not filtered_df.empty and not np.issubdtype(filtered_df['ENTITY'].dtype, np.number):
        entity_map = {v: i for i, v in enumerate(sorted(filtered_df['ENTITY'].unique()))}
        fig.update_yaxes(
            tickvals=list(entity_map.values()),
            ticktext=list(entity_map.keys()),
            title_text='ENTITY'
        )
    else:
        fig.update_yaxes(title_text='ENTITY')

    if not filtered_df.empty:
        summary = filtered_df.groupby('ENTITY').agg(
            count=('DELAY_TIME', 'count'),
            mean_delay=('DELAY_TIME', 'mean'),
            min_delay=('DELAY_TIME', 'min'),
            max_delay=('DELAY_TIME', 'max'),
            split_yes=('SPLIT', lambda x: (x == 'YES').sum())
        ).reset_index()
        table_header = [html.Tr([html.Th(col, style={'border': '1px solid black', 'padding': '4px'}) for col in summary.columns])]
        table_rows = []
        for _, row in summary.iterrows():
            cells = []
            for col in summary.columns:
                val = row[col]
                if col == 'mean_delay' and pd.notnull(val):
                    val = f"{val:.1f}"
                cells.append(html.Td(val, style={'border': '1px solid black', 'padding': '4px'}))
            table_rows.append(html.Tr(cells))
        summary_table = html.Table(table_header + table_rows, style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid black'})
    else:
        summary_table = html.Div('No data for selected filters.')
    return (fig, summary_table, month_plot)

if __name__ == '__main__':
    app.run(debug=True)
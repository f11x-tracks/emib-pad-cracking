import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Output, Input, State

# Read CSV
df = pd.read_csv('data.csv')
# Load wafer bow data
df_bow = pd.read_csv('data-wafer-bow.csv')
## PRODUCT_DESCRIPTION filter removed
df['LOT7'] = df['LOT'].astype(str).str[:7]

# --- Merge Lot-good-bad.csv columns ---
lot_good_bad = pd.read_csv('Lot-good-bad.csv')
df = df.merge(lot_good_bad, on='LOT', how='left')
df['Fab Defect Scans'] = df['Fab Defect Scans'].fillna('NA')  # <-- mark missing as 'NA'
df['Nrows'] = df['Nrows'].fillna('NA')
df['Nlot'] = df['Nlot'].fillna('NA')
df['DiePrep CIM'] = df['DiePrep CIM'].fillna('NA')

# Filter for OPN 194997 and 197573
def format_opn(opn):
    try:
        return str(int(float(opn)))
    except:
        return str(opn)
opn_desc_map = {}
for opn_val in df['OPN'].unique():
    if pd.notnull(opn_val):
        opn_str = format_opn(opn_val)
        desc = df[df['OPN'].apply(format_opn) == opn_str]['OPER_SHORT_DESC'].iloc[0] if 'OPER_SHORT_DESC' in df.columns else ''
        opn_desc_map[opn_str] = f"{opn_str} - {desc}" if desc else opn_str
unique_opns = sorted(opn_desc_map.keys())
opn_dfs = {}
for opn in unique_opns:
    df_opn = df[df['OPN'].apply(format_opn) == opn].copy()
    # For each LOT, keep only the row with the most recent LAST_WAFER_END_DATE
    if 'LAST_WAFER_END_DATE' in df_opn.columns:
        df_opn['LAST_WAFER_END_DATE'] = pd.to_datetime(df_opn['LAST_WAFER_END_DATE'])
        df_opn = df_opn.sort_values('LAST_WAFER_END_DATE').drop_duplicates(subset=['LOT'], keep='last')
    opn_dfs[opn] = df_opn

# Calculate DELAY_TIME and add ENTITY for matching LOTs
def calculate_delay_time(start_opn, end_opn, lot_col='LOT'):
    df_start = opn_dfs[start_opn]
    df_end = opn_dfs[end_opn]
    delay_times = []
    entities = []
    multi_counts = []
    split_values = []
    for idx, row in df_start.iterrows():
        lot_val = row[lot_col]
        lot_rows = df_end[df_end[lot_col] == lot_val]
        multi_count = len(lot_rows)
        if multi_count > 0:
            lot_rows = lot_rows.copy()
            lot_rows['LAST_WAFER_END_DATE'] = pd.to_datetime(lot_rows['LAST_WAFER_END_DATE'])
            valid_dates = lot_rows['LAST_WAFER_END_DATE'].notna()
            if not lot_rows.empty and valid_dates.any():
                most_recent_row = lot_rows.loc[lot_rows.loc[valid_dates, 'LAST_WAFER_END_DATE'].idxmax()]
                end_start = pd.to_datetime(row['LAST_WAFER_END_DATE'])
                end_end = most_recent_row['LAST_WAFER_END_DATE']
                delay = round((end_end - end_start).total_seconds() / 3600, 1)
                entity = most_recent_row['ENTITY']
                qty_start = row['QTY']
                qty_end = most_recent_row['QTY']
                split = 'YES' if qty_start != qty_end else 'NO'
            else:
                delay = None
                entity = None
                split = 'NO'
        else:
            delay = None
            entity = None
            split = 'NO'
        delay_times.append(delay)
        entities.append(entity)
        multi_counts.append(multi_count)
        split_values.append(split)
    df_result = df_start.copy()
    df_result['DELAY_TIME'] = delay_times
    df_result['ENTITY_END_OPN'] = entities
    df_result['MULTI'] = multi_counts
    df_result['SPLIT'] = split_values
    return df_result

# Chart: DELAY_TIME distribution
default_start_opn = unique_opns[0]
default_end_opn = unique_opns[1] if len(unique_opns) > 1 else unique_opns[0]
df_delay = calculate_delay_time(default_start_opn, default_end_opn)
df_delay_unique = df_delay.sort_values('LAST_WAFER_END_DATE').dropna(subset=['DELAY_TIME']).drop_duplicates(subset=['LOT'], keep='last')
min_date = pd.to_datetime(df_delay_unique['LAST_WAFER_END_DATE']).min().date()
max_date = pd.to_datetime(df_delay_unique['LAST_WAFER_END_DATE']).max().date()
fig = px.scatter(
    df_delay_unique,
    x='DELAY_TIME',
    y='ENTITY',
    color='Fab Defect Scans',
    color_discrete_map={'Clean': 'green', 'Defects': 'red'},  # custom color mapping
    title='DELAY_TIME vs ENTITY Scatter',
    hover_data=['LOT', 'MULTI', 'SPLIT', 'LOT_ABORT_FLAG', 'QTY', 'DOTPROCESS', 'PRODUCT', 'LAST_WAFER_END_DATE'] + [col for col in lot_good_bad.columns if col != 'LOT']
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
    html.Div([
        html.Label('LOT7?'),
        dcc.RadioItems(
            id='lot7-radio',
            options=[
                {'label': 'Use LOT', 'value': 'LOT'},
                {'label': 'Use LOT7', 'value': 'LOT7'}
            ],
            value='LOT',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        ),
        html.Label('LOT_TYPE:'),
        dcc.Dropdown(
            id='lot-type-dropdown',
            options=[{'label': 'All', 'value': 'All'}] + [{'label': str(lot_type), 'value': lot_type} for lot_type in sorted(df['LOT_TYPE'].dropna().unique())],
            value='All',
            clearable=False,
            style={'width': '200px'},
            placeholder='Select LOT_TYPE'
        ),
        html.Label('Start OPN:'),
        dcc.Dropdown(
            id='start-opn-dropdown',
            options=[{'label': opn_desc_map[opn], 'value': opn} for opn in unique_opns],
            value=default_start_opn,
            clearable=False,
            style={'width': '340px'}
        ),
        html.Label('End OPN:'),
        dcc.Dropdown(
            id='end-opn-dropdown',
            options=[{'label': opn_desc_map[opn], 'value': opn} for opn in unique_opns],
            value=default_end_opn,
            clearable=False,
            style={'width': '340px'}
        ),
        html.Label('ENTITY OPN:'),
        dcc.Dropdown(
            id='entity-opn-dropdown',
            options=[{'label': opn_desc_map[opn], 'value': opn} for opn in unique_opns],
            value=default_end_opn,
            clearable=False,
            style={'width': '340px'}
        ),
        html.Label('Include NA (no Fab Defect Scans match):'),
        dcc.RadioItems(
            id='include-na-radio',
            options=[
                {'label': 'Include NA', 'value': 'include'},
                {'label': 'Exclude NA', 'value': 'exclude'}
            ],
            value='exclude',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        ),
    ], style={'display': 'flex', 'gap': '20px', 'alignItems': 'center'}),
    dcc.Dropdown(
        id='lot-dropdown',
        options=[{'label': str(lot), 'value': lot} for lot in df_delay_unique['LOT'].unique()],
        placeholder='Select a LOT',
        multi=False
    ),
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Button('Export to Excel', id='export-excel-btn', n_clicks=0),
    dcc.Download(id='download-dataframe-xlsx'),
    html.H2('Summary Statistics by ENTITY'),
    html.Div(id='summary-table'),
    html.H2('LOT Count by ENTITY and Month'),
    dcc.Graph(id='lot-entity-month-plot'),
    html.H2('Wafer Bow Analysis'),
    html.Div(id='bow-charts-container')
])

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
    State('start-opn-dropdown', 'value'),
    State('end-opn-dropdown', 'value'),
    State('entity-opn-dropdown', 'value'),
    State('lot7-radio', 'value'),
    State('lot-type-dropdown', 'value'),
    State('lot-dropdown', 'value'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date'),
    State('include-na-radio', 'value'),
    prevent_initial_call=True
)
def export_to_excel(n_clicks, start_opn, end_opn, entity_opn, lot_col, selected_lot_type, selected_lot, start_date, end_date, include_na):
    if n_clicks:
        # Recalculate with current filter settings
        df_delay = calculate_delay_time(start_opn, end_opn, lot_col=lot_col)
        filtered_df = df_delay.sort_values('LAST_WAFER_END_DATE').dropna(subset=['DELAY_TIME']).drop_duplicates(subset=[lot_col], keep='last')
        
        # Assign NA to missing Fab Defect Scans
        if 'Fab Defect Scans' in filtered_df.columns:
            filtered_df['Fab Defect Scans'] = filtered_df['Fab Defect Scans'].fillna('NA')
        
        # Filter NA if requested
        if include_na == 'exclude' and 'Fab Defect Scans' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Fab Defect Scans'] != 'NA']
        
        # Filter by LOT_TYPE if not 'All'
        if selected_lot_type and selected_lot_type != 'All':
            filtered_df = filtered_df[filtered_df['LOT_TYPE'] == selected_lot_type]
        
        # Use ENTITY values from selected ENTITY_OPN
        entity_df = opn_dfs[entity_opn]
        if 'LAST_WAFER_END_DATE' in entity_df.columns:
            entity_df['LAST_WAFER_END_DATE'] = pd.to_datetime(entity_df['LAST_WAFER_END_DATE'])
            entity_df = entity_df.sort_values('LAST_WAFER_END_DATE').drop_duplicates(subset=[lot_col], keep='last')
        entity_map = dict(zip(entity_df[lot_col], entity_df['ENTITY']))
        filtered_df['ENTITY_FOR_PLOT'] = filtered_df[lot_col].map(entity_map)
        
        # Filter by date range
        if start_date and end_date:
            filtered_df = filtered_df[(pd.to_datetime(filtered_df['LAST_WAFER_END_DATE']).dt.date >= pd.to_datetime(start_date).date()) &
                                      (pd.to_datetime(filtered_df['LAST_WAFER_END_DATE']).dt.date <= pd.to_datetime(end_date).date())]
        
        # Filter by LOT/LOT7
        if selected_lot:
            filtered_df = filtered_df[filtered_df[lot_col] == selected_lot]
        # Wafer bow data for matching LOTs only
        lots_bow = filtered_df['LOT'].unique()
        bow_df_export = df_bow[df_bow['LOT'].isin(lots_bow)].copy()
        bow_df_export = pd.merge(bow_df_export, filtered_df[['LOT', 'DELAY_TIME']], on='LOT', how='left')
        # Export both sheets
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            filtered_df.to_excel(writer, sheet_name='Main', index=False)
            bow_df_export.to_excel(writer, sheet_name='WaferBow', index=False)
        buffer.seek(0)
        return dcc.send_bytes(buffer.getvalue(), 'filtered_data_with_bow.xlsx')
    return None
    # ...removed old export_df merge and return...
    return None

@app.callback(
    [Output('scatter-plot', 'figure'), Output('summary-table', 'children'), Output('lot-entity-month-plot', 'figure'), Output('lot-dropdown', 'options'),
     Output('bow-charts-container', 'children')],
    [Input('start-opn-dropdown', 'value'),
     Input('end-opn-dropdown', 'value'),
     Input('entity-opn-dropdown', 'value'),
     Input('lot7-radio', 'value'),
     Input('lot-type-dropdown', 'value'),
     Input('lot-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('include-na-radio', 'value')]
)
def update_figure(start_opn, end_opn, entity_opn, lot_col, selected_lot_type, selected_lot, start_date, end_date, include_na):
    import numpy as np
    df_delay = calculate_delay_time(start_opn, end_opn, lot_col=lot_col)
    df_delay_unique = df_delay.sort_values('LAST_WAFER_END_DATE').dropna(subset=['DELAY_TIME']).drop_duplicates(subset=[lot_col], keep='last')
    # Assign NA to missing Fab Defect Scans
    if 'Fab Defect Scans' in df_delay_unique.columns:
        df_delay_unique['Fab Defect Scans'] = df_delay_unique['Fab Defect Scans'].fillna('NA')
    filtered_df = df_delay_unique.copy()
    # Filter NA if requested
    if include_na == 'exclude' and 'Fab Defect Scans' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Fab Defect Scans'] != 'NA']
    # Merge wafer bow data
    # Ensure OPN is string for comparison
    df_bow_merged = pd.merge(filtered_df, df_bow, on='LOT', how='left')
    # Use correct ENTITY column for filtering (ENTITY_END_OPN from delay calculation)
    # Wafer bow data: match to filtered_df LOTs only
    lots_bow = filtered_df['LOT'].unique()
    bow_df = df_bow[df_bow['LOT'].isin(lots_bow)].copy()
    # Add DELAY_TIME to bow_df by merging with filtered_df
    # Add DELAY_TIME and Fab Defect Scans to bow_df by merging with filtered_df
    bow_df = pd.merge(bow_df, filtered_df[['LOT', 'DELAY_TIME', 'Fab Defect Scans']], on='LOT', how='left')
    # Create separate chart for each CHART_TEST_NAME and CHART_TYPE
    bow_charts = []
    color_map_bow = {'Clean': 'green', 'Defects': 'red'}
    # Map ENTITY from selected ENTITY OPN to bow_df
    entity_df = opn_dfs[entity_opn]
    entity_df['LAST_WAFER_END_DATE'] = pd.to_datetime(entity_df['LAST_WAFER_END_DATE'])
    entity_df = entity_df.sort_values('LAST_WAFER_END_DATE').drop_duplicates(subset=[lot_col], keep='last')
    entity_map = dict(zip(entity_df[lot_col], entity_df['ENTITY']))
    bow_df['ENTITY_FOR_BOW'] = bow_df['LOT'].map(entity_map)
    for (test_name, chart_type), group in bow_df.groupby(['CHART_TEST_NAME', 'CHART_TYPE']):
        if group.empty:
            continue
        fig_bow = px.scatter(
            group,
            x='DELAY_TIME',
            y='CHART_VALUE',
            color='Fab Defect Scans',
            symbol='Fab Defect Scans',
            color_discrete_map={'Clean': 'green', 'Defects': 'red'},
            title=f'Wafer Bow: {test_name} / {chart_type} (Grouped by ENTITY)',
            hover_data=['LOT', 'CHART_TEST_NAME', 'CHART_TYPE', 'CHART_VALUE', 'DELAY_TIME', 'ENTITY_FOR_BOW', 'Fab Defect Scans']
        )
        bow_charts.append(html.Div([
            html.H4(f'{test_name} / {chart_type}'),
            dcc.Graph(figure=fig_bow)
        ], style={'marginBottom': '40px'}))
    # Filter by LOT_TYPE if not 'All'
    if selected_lot_type and selected_lot_type != 'All':
        filtered_df = filtered_df[filtered_df['LOT_TYPE'] == selected_lot_type]
    # PRODUCT_DESCRIPTION filter removed
    # Use ENTITY values from selected ENTITY_OPN
    entity_df = opn_dfs[entity_opn]
    # For each LOT, use the most recent LAST_WAFER_END_DATE
    if 'LAST_WAFER_END_DATE' in entity_df.columns:
        entity_df['LAST_WAFER_END_DATE'] = pd.to_datetime(entity_df['LAST_WAFER_END_DATE'])
        entity_df = entity_df.sort_values('LAST_WAFER_END_DATE').drop_duplicates(subset=[lot_col], keep='last')
    entity_map = dict(zip(entity_df[lot_col], entity_df['ENTITY']))
    filtered_df['ENTITY_FOR_PLOT'] = filtered_df[lot_col].map(entity_map)
    # Prepare Year-Month column
    filtered_df['YEAR_MONTH'] = pd.to_datetime(filtered_df['LAST_WAFER_END_DATE']).dt.to_period('M').astype(str)
    lot_month_counts = filtered_df.groupby(['YEAR_MONTH', 'ENTITY_FOR_PLOT'])[lot_col].count().reset_index().rename(columns={'ENTITY_FOR_PLOT': 'ENTITY', lot_col: 'LOT_COUNT'})
    month_plot = px.bar(
        lot_month_counts,
        x='YEAR_MONTH',
        y='LOT_COUNT',
        color='ENTITY',
        barmode='group',
        title=f'{lot_col} Count by ENTITY and Month',
        labels={'LOT_COUNT': f'{lot_col} Count', 'YEAR_MONTH': 'Year-Month'}
    )
    month_plot.update_xaxes(tickangle=90)
    # Filter by date range
    if start_date and end_date:
        filtered_df = filtered_df[(pd.to_datetime(filtered_df['LAST_WAFER_END_DATE']).dt.date >= pd.to_datetime(start_date).date()) &
                                  (pd.to_datetime(filtered_df['LAST_WAFER_END_DATE']).dt.date <= pd.to_datetime(end_date).date())]
    # Filter by LOT/LOT7
    if selected_lot:
        filtered_df = filtered_df[filtered_df[lot_col] == selected_lot]
    lot_options = [{'label': str(lot), 'value': lot} for lot in filtered_df[lot_col].unique()]
    jitter_strength = 0.05
    y_col = 'ENTITY_FOR_PLOT'
    if not filtered_df.empty:
        try:
            if np.issubdtype(filtered_df['ENTITY_FOR_PLOT'].dtype, np.number):
                filtered_df['ENTITY_JITTER'] = filtered_df['ENTITY_FOR_PLOT'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(filtered_df))
            else:
                entity_unique = [str(v) for v in filtered_df['ENTITY_FOR_PLOT'].unique()]
                entity_plot_map = {v: i for i, v in enumerate(sorted(entity_unique))}
                filtered_df['ENTITY_NUM'] = filtered_df['ENTITY_FOR_PLOT'].astype(str).map(entity_plot_map)
                filtered_df['ENTITY_JITTER'] = filtered_df['ENTITY_NUM'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(filtered_df))
            y_col = 'ENTITY_JITTER'
        except Exception:
            y_col = 'ENTITY_FOR_PLOT'
    fig = px.scatter(
        filtered_df,
        x='DELAY_TIME',
        y=y_col,
        color='Fab Defect Scans',
        color_discrete_map={'Clean': 'green', 'Defects': 'red'},  # custom color mapping
        title=f'DELAY_TIME vs ENTITY Scatter (Y Jittered) [{lot_col}]',
        hover_data=[lot_col, 'MULTI', 'SPLIT', 'LOT_ABORT_FLAG', 'QTY', 'DOTPROCESS', 'PRODUCT', 'LAST_WAFER_END_DATE', 'ENTITY_FOR_PLOT'] + [col for col in lot_good_bad.columns if col != 'LOT']
    )
    fig.update_xaxes(title_text='DELAY_TIME (hours)')
    if not filtered_df.empty and not np.issubdtype(filtered_df['ENTITY_FOR_PLOT'].dtype, np.number):
        entity_unique = [str(v) for v in filtered_df['ENTITY_FOR_PLOT'].unique()]
        entity_plot_map = {v: i for i, v in enumerate(sorted(entity_unique))}
        fig.update_yaxes(
            tickvals=list(entity_plot_map.values()),
            ticktext=list(entity_plot_map.keys()),
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
    return (fig, summary_table, month_plot, lot_options, bow_charts)

if __name__ == '__main__':
    app.run(debug=True)
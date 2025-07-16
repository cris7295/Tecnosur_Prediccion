import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import src.fuzzy_logic as fuzzy
from src.data_processor import data_processor
from src.alert_system import alert_system
from src.report_generator import report_generator
import os
import json
from datetime import datetime
import base64
import io

# Cargar el dataset mejorado
try:
    df = data_processor.df
    if df is None or df.empty:
        raise ValueError("No se pudieron cargar los datos")
    print(f"Dataset cargado exitosamente: {len(df)} estudiantes")
except Exception as e:
    print(f"Error cargando datos: {str(e)} - Generando datos de ejemplo")
    data_processor.generate_sample_data()
    df = data_processor.df

# Cargar los modelos entrenados con rutas absolutas
base_dir = os.path.dirname(os.path.abspath(__file__))

try:
    model_dt = joblib.load(os.path.join(base_dir, 'models', 'prediction_model.joblib'))
except FileNotFoundError:
    model_dt = None

try:
    model_keras = load_model(os.path.join(base_dir, 'models', 'keras_model.keras'))
except Exception:
    model_keras = None

try:
    model_rf = joblib.load(os.path.join(base_dir, 'models', 'random_forest_model.joblib'))
except FileNotFoundError:
    model_rf = None

# Inicializar la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ======================
# 🚀 API Simple (sin modificar el código existente)
# ======================
@server.route('/api/predict', methods=['POST'])
def api_predict():
    """Endpoint para predicciones via API"""
    try:
        # 1. Recibir datos JSON
        data = request.json
        
        # 2. Validar campos requeridos
        required = ['calificaciones', 'asistencia', 'participacion', 'horas_estudio', 'nivel_socioeconomico']
        if not all(field in data for field in required):
            return jsonify({"error": "Faltan campos requeridos"}), 400

        # 3. Usar la misma lógica de predicción del callback
        mapping_nivel = {'Bajo': 3, 'Medio': 6, 'Alto': 9}
        nivel_val = mapping_nivel.get(data['nivel_socioeconomico'], 5)
        
        # Lógica difusa
        riesgo_fuzzy = fuzzy.evaluar_riesgo(
            nivel_val,
            data['participacion'] * 2,
            data['asistencia'],
            data['calificaciones']
        )

        # Predicción con Random Forest (default)
        input_df = pd.DataFrame([[
            data['calificaciones'],
            data['asistencia'],
            data['participacion'],
            data['horas_estudio'],
            data['nivel_socioeconomico']
        ]], columns=['calificaciones_anteriores', 'asistencia_porcentaje', 
                    'participacion_clase', 'horas_estudio_semanal', 
                    'nivel_socioeconomico'])

        if model_rf:
            pred = model_rf.predict(input_df)[0]
            proba = model_rf.predict_proba(input_df)[0]
            return jsonify({
                "prediccion": int(pred),
                "confianza_riesgo": float(proba[1]),
                "logica_difusa": float(riesgo_fuzzy) if riesgo_fuzzy else None
            })
        else:
            return jsonify({"error": "Modelo no cargado"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@server.route('/api/data_sample', methods=['GET'])
def api_data_sample():
    """Endpoint para obtener muestra de datos"""
    try:
        sample = df.sample(min(5, len(df))).to_dict('records')
        return jsonify({"data": sample, "total_records": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Figuras para el Dashboard ---
# Gráfico de pie para la distribución de riesgo
fig_pie_riesgo = px.pie(
    df, names='rendimiento_riesgo', title='Proporción de Estudiantes en Riesgo vs. No en Riesgo', hole=.3
).update_traces(textinfo='percent+label')

# Histograma de calificaciones por nivel de riesgo
fig_hist_calificaciones = px.histogram(
    df, x="calificaciones_anteriores", color="rendimiento_riesgo",
    marginal="box", barmode="overlay",
    title="Distribución de Calificaciones por Nivel de Riesgo"
)

# Obtener estadísticas para el dashboard
stats = data_processor.get_statistics()

# Definir el layout de la aplicación
app.layout = html.Div(className='container', children=[
    html.Div(className='header', children=[
        html.H1('🎓 Sistema de Predicción de Notas y Alertas Académicas', 
                style={'color': '#34495e', 'fontWeight': 'bold', 'textAlign': 'center'})
    ]),

    dcc.Tabs(id="tabs-main", value='tab-dashboard', children=[
        # Pestaña 1: Dashboard General Mejorado
        dcc.Tab(label='📊 Dashboard General', value='tab-dashboard', children=[
            # KPIs principales
            html.Div(className='kpi-container', style={'display': 'flex', 'justifyContent': 'space-around', 'marginTop': '20px'}, children=[
                html.Div(className='kpi-card', style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#3498db', 'color': 'white', 'borderRadius': '10px', 'minWidth': '150px'}, children=[
                    html.H2(str(stats['total_estudiantes']), style={'margin': '0', 'fontSize': '2.5rem'}),
                    html.P('Total Estudiantes', style={'margin': '5px 0', 'fontSize': '1rem'})
                ]),
                html.Div(className='kpi-card', style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#e74c3c', 'color': 'white', 'borderRadius': '10px', 'minWidth': '150px'}, children=[
                    html.H2(str(stats['estudiantes_riesgo']), style={'margin': '0', 'fontSize': '2.5rem'}),
                    html.P('En Riesgo Alto', style={'margin': '5px 0', 'fontSize': '1rem'})
                ]),
                html.Div(className='kpi-card', style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#2ecc71', 'color': 'white', 'borderRadius': '10px', 'minWidth': '150px'}, children=[
                    html.H2(f"{stats['porcentaje_sin_riesgo']}%", style={'margin': '0', 'fontSize': '2.5rem'}),
                    html.P('Sin Riesgo', style={'margin': '5px 0', 'fontSize': '1rem'})
                ]),
                html.Div(className='kpi-card', style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f39c12', 'color': 'white', 'borderRadius': '10px', 'minWidth': '150px'}, children=[
                    html.H2(str(stats['promedio_general']), style={'margin': '0', 'fontSize': '2.5rem'}),
                    html.P('Promedio General', style={'margin': '5px 0', 'fontSize': '1rem'})
                ])
            ]),
            
            # Gráficos principales
            html.Div(style={'display': 'flex', 'gap': '20px', 'marginTop': '30px'}, children=[
                html.Div(className='card', style={'flex': '1'}, children=[
                    dcc.Graph(id='grafico-riesgo-mejorado')
                ]),
                html.Div(className='card', style={'flex': '1'}, children=[
                    dcc.Graph(id='grafico-carreras')
                ])
            ]),
            
            html.Div(style={'display': 'flex', 'gap': '20px', 'marginTop': '20px'}, children=[
                html.Div(className='card', style={'flex': '1'}, children=[
                    dcc.Graph(id='grafico-asistencia-vs-notas')
                ]),
                html.Div(className='card', style={'flex': '1'}, children=[
                    dcc.Graph(id='grafico-semestres')
                ])
            ]),
            
            # Panel de alertas recientes
            html.Div(className='card', style={'marginTop': '20px'}, children=[
                html.H3('🚨 Alertas Recientes', style={'color': '#e74c3c', 'marginBottom': '15px'}),
                html.Div(id='alertas-recientes')
            ])
        ]),

        # Pestaña 2: Lista de Estudiantes
        dcc.Tab(label='👥 Lista de Estudiantes', value='tab-students', children=[
            html.Div(className='card', style={'marginTop': '20px'}, children=[
                html.H2('Lista de Estudiantes con Filtros', style={'color': '#2c3e50', 'marginBottom': '20px'}),
                
                # Filtros
                html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '20px', 'flexWrap': 'wrap'}, children=[
                    html.Div([
                        html.Label('Buscar:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                        dcc.Input(
                            id='filtro-busqueda',
                            type='text',
                            placeholder='Nombre, apellido o email...',
                            style={'width': '200px', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #ddd'}
                        )
                    ]),
                    html.Div([
                        html.Label('Carrera:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='filtro-carrera',
                            options=[{'label': 'Todas', 'value': ''}] + [{'label': c, 'value': c} for c in stats['carreras']],
                            value='',
                            style={'width': '150px'}
                        )
                    ]),
                    html.Div([
                        html.Label('Semestre:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='filtro-semestre',
                            options=[{'label': 'Todos', 'value': ''}] + [{'label': f'Semestre {s}', 'value': s} for s in stats['semestres']],
                            value='',
                            style={'width': '120px'}
                        )
                    ]),
                    html.Div([
                        html.Label('Riesgo:', style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='filtro-riesgo',
                            options=[
                                {'label': 'Todos', 'value': ''},
                                {'label': 'Sin Riesgo', 'value': 0},
                                {'label': 'En Riesgo', 'value': 1}
                            ],
                            value='',
                            style={'width': '120px'}
                        )
                    ]),
                    html.Div([
                        html.Label(' ', style={'marginBottom': '5px', 'display': 'block'}),
                        html.Button('Aplicar Filtros', id='btn-aplicar-filtros', n_clicks=0,
                                  style={'padding': '8px 15px', 'backgroundColor': '#3498db', 'color': 'white', 
                                        'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
                    ])
                ]),
                
                # Tabla de estudiantes
                html.Div(id='tabla-estudiantes-container')
            ])
        ]),

        # Pestaña 3: Detalle del Estudiante
        dcc.Tab(label='👤 Detalle Estudiante', value='tab-student-detail', children=[
            html.Div(className='card', style={'marginTop': '20px'}, children=[
                html.H2('Seleccionar Estudiante', style={'color': '#2c3e50', 'marginBottom': '20px'}),
                dcc.Dropdown(
                    id='selector-estudiante',
                    options=[{'label': f"{row['nombre']} {row['apellido']} - {row['carrera']}", 'value': row['id_estudiante']} 
                            for _, row in df.iterrows()],
                    placeholder='Seleccione un estudiante...',
                    style={'marginBottom': '20px'}
                ),
                html.Div(id='detalle-estudiante-container')
            ])
        ]),

        # Pestaña 4: Predicción Individual (Mejorada)
        dcc.Tab(label='🔮 Predicción Individual', value='tab-prediction', children=[
            html.Div(className='card', style={'marginTop': '20px'}, children=[
                html.H2('Predicción en Tiempo Real', style={'color': '#2c3e50', 'fontWeight': '600'}),
                html.P('Ingrese los datos del estudiante para predecir el riesgo académico usando múltiples modelos de ML.', 
                      style={'fontSize': '16px', 'color': '#7f8c8d', 'marginBottom': '25px'}),

                html.Div([
                    dcc.Input(id='calificaciones_anteriores', type='number', placeholder='Calificaciones (4.0-10.0)', 
                             min=4, max=10, step=0.1, style={'marginRight':'10px', 'width': '150px'}),
                    dcc.Input(id='asistencia_porcentaje', type='number', placeholder='Asistencia % (50-100)', 
                             min=50, max=100, step=1, style={'marginRight':'10px', 'width': '150px'}),
                    dcc.Input(id='participacion_clase', type='number', placeholder='Participación (1-5)', 
                             min=1, max=5, step=1, style={'marginRight':'10px', 'width': '150px'}),
                ], style={'marginBottom': '15px'}),

                html.Div([
                    dcc.Input(id='horas_estudio_semanal', type='number', placeholder='Horas de Estudio (1-25)', 
                             min=1, max=25, step=1, style={'marginRight':'10px', 'width': '150px'}),
                    dcc.Dropdown(
                        id='nivel_socioeconomico',
                        options=[{'label': i, 'value': i} for i in df['nivel_socioeconomico'].unique()],
                        placeholder='Nivel Socioeconómico',
                        style={'width': '220px', 'marginRight':'10px'}
                    ),
                ], style={'marginBottom': '25px', 'display': 'flex', 'alignItems': 'center'}),

                dcc.Dropdown(
                    id='modelo-seleccionado',
                    options=[
                        {'label': '🌳 Árbol de Decisión', 'value': 'dt'},
                        {'label': '🧠 Red Neuronal', 'value': 'keras'},
                        {'label': '🌲 Random Forest', 'value': 'rf'}
                    ],
                    value='rf',
                    clearable=False,
                    style={'width': '220px', 'marginBottom': '20px'}
                ),

                html.Button('🔍 Predecir Riesgo', id='boton-predecir', n_clicks=0, 
                           style={'backgroundColor': '#2980b9', 'color': 'white', 'padding': '12px 25px', 
                                 'borderRadius': '6px', 'fontSize': '1.1rem', 'cursor': 'pointer', 'border': 'none'}),
                
                html.Div(id='resultado-prediccion', style={'marginTop': '20px', 'fontSize': '20px', 'fontWeight': 'bold', 'color': '#34495e'}),
                html.H3('🔬 Evaluación de Riesgo con Lógica Difusa', style={'marginTop': '30px', 'color': '#2c3e50'}),
                html.Div(id='resultado-fuzzy', style={'marginTop': '10px', 'fontSize': '18px', 'fontWeight': 'normal', 'color': '#7f8c8d'}),
                
                # Sección de gráficos de análisis visual
                html.Div(id='seccion-graficos', style={'marginTop': '30px'}, children=[
                    html.H3('📊 Análisis Visual del Estudiante', style={'color': '#2c3e50', 'marginBottom': '20px'}),
                    
                    # Primera fila de gráficos
                    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}, children=[
                        html.Div(className='card', style={'flex': '1'}, children=[
                            dcc.Graph(id='grafico-radar-estudiante')
                        ]),
                        html.Div(className='card', style={'flex': '1'}, children=[
                            dcc.Graph(id='grafico-barras-estudiante')
                        ])
                    ]),
                    
                    # Segunda fila de gráficos
                    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}, children=[
                        html.Div(className='card', style={'flex': '1'}, children=[
                            dcc.Graph(id='grafico-gauge-riesgo')
                        ]),
                        html.Div(className='card', style={'flex': '1'}, children=[
                            dcc.Graph(id='grafico-comparacion-promedio')
                        ])
                    ])
                ]),
                
                html.Div([
                    html.H4('📋 Historial de Predicciones', style={'marginTop': '30px', 'color': '#2c3e50'}),
                    html.Ul(id='historial-predicciones', style={'fontSize': '16px', 'color': '#7f8c8d', 'maxHeight': '150px', 'overflowY': 'auto', 'paddingLeft': '20px'})
                ])
            ])
        ]),

        # Pestaña 5: Subida de Datos
        dcc.Tab(label='📤 Subir Datos', value='tab-upload', children=[
            html.Div(className='card', style={'marginTop': '20px'}, children=[
                html.H2('Subir Nuevos Datos', style={'color': '#2c3e50', 'marginBottom': '20px'}),
                html.P('Suba archivos CSV o Excel con datos de estudiantes. El sistema validará automáticamente los datos.', 
                      style={'color': '#7f8c8d', 'marginBottom': '20px'}),
                
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        '📁 Arrastre y suelte archivos aquí o ',
                        html.A('seleccione archivos', style={'color': '#3498db', 'textDecoration': 'underline'})
                    ]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                        'textAlign': 'center', 'margin': '10px 0', 'borderColor': '#3498db',
                        'backgroundColor': '#f8f9fa'
                    },
                    multiple=False
                ),
                
                html.Div(id='upload-status', style={'marginTop': '20px'}),
                html.Div(id='validation-results', style={'marginTop': '20px'})
            ])
        ]),

        # Pestaña 6: Alertas y Reportes
        dcc.Tab(label='🚨 Alertas y Reportes', value='tab-alerts', children=[
            html.Div(className='card', style={'marginTop': '20px'}, children=[
                html.H2('Sistema de Alertas y Reportes', style={'color': '#2c3e50', 'marginBottom': '20px'}),
                
                # Estadísticas de alertas
                html.Div(id='estadisticas-alertas', style={'marginBottom': '30px'}),
                
                # Filtros de alertas
                html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '20px'}, children=[
                    dcc.Dropdown(
                        id='filtro-tipo-alerta',
                        options=[
                            {'label': 'Todas las Alertas', 'value': ''},
                            {'label': 'Bajo Rendimiento', 'value': 'BAJO_RENDIMIENTO'},
                            {'label': 'Baja Asistencia', 'value': 'BAJA_ASISTENCIA'},
                            {'label': 'Baja Participación', 'value': 'BAJA_PARTICIPACION'},
                            {'label': 'Riesgo Alto', 'value': 'RIESGO_ALTO'}
                        ],
                        value='',
                        placeholder='Filtrar por tipo...',
                        style={'width': '200px'}
                    ),
                    dcc.Dropdown(
                        id='filtro-prioridad-alerta',
                        options=[
                            {'label': 'Todas las Prioridades', 'value': ''},
                            {'label': 'Crítica', 'value': 'CRITICAL'},
                            {'label': 'Alta', 'value': 'HIGH'},
                            {'label': 'Media', 'value': 'MEDIUM'},
                            {'label': 'Baja', 'value': 'LOW'}
                        ],
                        value='',
                        placeholder='Filtrar por prioridad...',
                        style={'width': '200px'}
                    ),
                    html.Button('🔄 Actualizar Alertas', id='btn-actualizar-alertas', n_clicks=0,
                              style={'padding': '8px 15px', 'backgroundColor': '#e74c3c', 'color': 'white', 
                                    'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
                ]),
                
                # Lista de alertas
                html.Div(id='lista-alertas'),
                
                # Botones de reportes
                html.Div(style={'marginTop': '30px', 'borderTop': '1px solid #ddd', 'paddingTop': '20px'}, children=[
                    html.H3('📊 Generar Reportes', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div(style={'display': 'flex', 'gap': '15px'}, children=[
                        html.Button('📄 Reporte General PDF', id='btn-reporte-general', n_clicks=0,
                                  style={'padding': '10px 20px', 'backgroundColor': '#27ae60', 'color': 'white', 
                                        'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                        html.Button('📧 Simular Envío Email', id='btn-simular-email', n_clicks=0,
                                  style={'padding': '10px 20px', 'backgroundColor': '#f39c12', 'color': 'white', 
                                        'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
                    ])
                ]),
                
                html.Div(id='resultado-reportes', style={'marginTop': '20px'})
            ])
        ]),

        # Pestaña 7: Visualizaciones Avanzadas
        dcc.Tab(label='📈 Visualizaciones Avanzadas', value='tab-advanced', children=[
            html.Div(className='card', style={'marginTop': '20px'}, children=[
                html.H2('Análisis Avanzado y Visualizaciones', style={'color': '#2c3e50', 'marginBottom': '20px'}),
                
                html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}, children=[
                    html.Div(className='card', style={'flex': '1'}, children=[
                        html.H3('🔥 Mapa de Calor - Correlaciones', style={'textAlign': 'center'}),
                        dcc.Graph(id='heatmap-correlaciones')
                    ])
                ]),
                
                html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}, children=[
                    html.Div(className='card', style={'flex': '1'}, children=[
                        html.H3('🎯 Clustering de Estudiantes', style={'textAlign': 'center'}),
                        dcc.Graph(id='clustering-estudiantes')
                    ]),
                    html.Div(className='card', style={'flex': '1'}, children=[
                        html.H3('📊 Distribución por Género', style={'textAlign': 'center'}),
                        dcc.Graph(id='distribucion-genero')
                    ])
                ]),
                
                html.Div(className='card', children=[
                    html.H3('📈 Series Temporales - Progreso Académico', style={'textAlign': 'center'}),
                    dcc.Graph(id='series-temporales')
                ])
            ])
        ]),

        # Pestaña 8: Dataset Completo
        dcc.Tab(label='📋 Dataset Completo', value='tab-data', children=[
            html.Div(className='card', style={'marginTop': '20px'}, children=[
                html.H2('Visualización del Dataset Completo', style={'color': '#2c3e50', 'fontWeight': '600'}),
                html.P(f'Total de registros: {len(df)} estudiantes', style={'color': '#7f8c8d', 'marginBottom': '20px'}),
                dash_table.DataTable(
                    id='tabla-dataset',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Arial', 'fontSize': '12px'},
                    style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{rendimiento_riesgo} = 1'},
                            'backgroundColor': '#ffebee',
                            'color': 'black',
                        }
                    ],
                    page_size=20,
                    sort_action="native",
                    filter_action="native"
                )
            ])
        ])
    ])
])

# ==========================================
# CALLBACKS PARA TODAS LAS FUNCIONALIDADES
# ==========================================

# Callback para gráficos del dashboard principal
@app.callback(
    [Output('grafico-riesgo-mejorado', 'figure'),
     Output('grafico-carreras', 'figure'),
     Output('grafico-asistencia-vs-notas', 'figure'),
     Output('grafico-semestres', 'figure')],
    [Input('tabs-main', 'value')]
)
def update_dashboard_graphs(active_tab):
    if active_tab != 'tab-dashboard':
        return {}, {}, {}, {}
    
    # Gráfico de riesgo mejorado
    risk_data = data_processor.get_risk_distribution()
    fig_riesgo = px.pie(
        values=list(risk_data.values()),
        names=list(risk_data.keys()),
        title='📊 Distribución de Riesgo Académico',
        color_discrete_map={'Sin Riesgo': '#2ecc71', 'En Riesgo': '#e74c3c'},
        hole=0.4
    )
    fig_riesgo.update_traces(textinfo='percent+label', textfont_size=14)
    fig_riesgo.update_layout(font=dict(size=12))
    
    # Gráfico por carreras
    carrera_risk = df.groupby('carrera')['rendimiento_riesgo'].agg(['count', 'sum']).reset_index()
    carrera_risk['porcentaje_riesgo'] = (carrera_risk['sum'] / carrera_risk['count'] * 100).round(1)
    
    fig_carreras = px.bar(
        carrera_risk, 
        x='carrera', 
        y='porcentaje_riesgo',
        title='📚 Porcentaje de Riesgo por Carrera',
        color='porcentaje_riesgo',
        color_continuous_scale='RdYlGn_r'
    )
    fig_carreras.update_layout(xaxis_tickangle=-45)
    
    # Gráfico asistencia vs notas
    fig_scatter = px.scatter(
        df, 
        x='asistencia_porcentaje', 
        y='calificaciones_anteriores',
        color='rendimiento_riesgo',
        title='📈 Relación Asistencia vs Calificaciones',
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        labels={'rendimiento_riesgo': 'Riesgo'}
    )
    
    # Gráfico por semestres
    semestre_stats = df.groupby('semestre').agg({
        'rendimiento_riesgo': ['count', 'sum'],
        'calificaciones_anteriores': 'mean'
    }).round(2)
    semestre_stats.columns = ['total', 'en_riesgo', 'promedio']
    semestre_stats = semestre_stats.reset_index()
    
    fig_semestres = px.bar(
        semestre_stats,
        x='semestre',
        y='en_riesgo',
        title='📅 Estudiantes en Riesgo por Semestre',
        color='promedio',
        color_continuous_scale='RdYlGn'
    )
    
    return fig_riesgo, fig_carreras, fig_scatter, fig_semestres

# Callback para alertas recientes
@app.callback(
    Output('alertas-recientes', 'children'),
    [Input('tabs-main', 'value')]
)
def update_recent_alerts(active_tab):
    if active_tab != 'tab-dashboard':
        return []
    
    # Generar algunas alertas de ejemplo para estudiantes en riesgo
    students_at_risk = df[df['rendimiento_riesgo'] == 1].head(5)
    alerts = []
    
    for _, student in students_at_risk.iterrows():
        alert_system.check_student_alerts(student.to_dict())
    
    recent_alerts = alert_system.get_recent_alerts(days=7, limit=5)
    
    if not recent_alerts:
        return html.P("No hay alertas recientes.", style={'color': '#7f8c8d', 'fontStyle': 'italic'})
    
    alert_items = []
    for alert in recent_alerts:
        priority_color = {
            'CRITICAL': '#e74c3c',
            'HIGH': '#f39c12', 
            'MEDIUM': '#3498db',
            'LOW': '#95a5a6'
        }.get(alert['priority'], '#95a5a6')
        
        alert_items.append(
            html.Div([
                html.Div([
                    html.Strong(alert['type_name'], style={'color': priority_color}),
                    html.Span(f" - {alert['student_name']}", style={'marginLeft': '10px'}),
                    html.Br(),
                    html.Small(alert['message'], style={'color': '#7f8c8d'})
                ], style={'padding': '10px', 'border': f'1px solid {priority_color}', 
                         'borderRadius': '5px', 'marginBottom': '10px', 'backgroundColor': '#f8f9fa'})
            ])
        )
    
    return alert_items

# Callback para filtros de estudiantes
@app.callback(
    Output('tabla-estudiantes-container', 'children'),
    [Input('btn-aplicar-filtros', 'n_clicks')],
    [State('filtro-busqueda', 'value'),
     State('filtro-carrera', 'value'),
     State('filtro-semestre', 'value'),
     State('filtro-riesgo', 'value')]
)
def update_students_table(n_clicks, busqueda, carrera, semestre, riesgo):
    filters = {
        'busqueda': busqueda,
        'carrera': carrera if carrera else None,
        'semestre': int(semestre) if semestre else None,
        'riesgo': int(riesgo) if riesgo != '' else None
    }
    
    filtered_df = data_processor.filter_students(filters)
    
    if filtered_df.empty:
        return html.P("No se encontraron estudiantes con los filtros aplicados.", 
                     style={'color': '#7f8c8d', 'fontStyle': 'italic'})
    
    # Preparar datos para la tabla
    display_columns = ['nombre', 'apellido', 'carrera', 'semestre', 'promedio_general', 
                      'asistencia_porcentaje', 'estado_academico', 'rendimiento_riesgo']
    
    table_data = filtered_df[display_columns].copy()
    table_data['riesgo_color'] = table_data['rendimiento_riesgo'].map({0: '🟢', 1: '🔴'})
    
    return dash_table.DataTable(
        data=table_data.to_dict('records'),
        columns=[
            {"name": "Nombre", "id": "nombre"},
            {"name": "Apellido", "id": "apellido"},
            {"name": "Carrera", "id": "carrera"},
            {"name": "Semestre", "id": "semestre"},
            {"name": "Promedio", "id": "promedio_general", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "Asistencia %", "id": "asistencia_porcentaje", "type": "numeric"},
            {"name": "Estado", "id": "estado_academico"},
            {"name": "Riesgo", "id": "riesgo_color"}
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Arial'},
        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{rendimiento_riesgo} = 1'},
                'backgroundColor': '#ffebee',
                'color': 'black',
            }
        ],
        page_size=15,
        sort_action="native",
        filter_action="native"
    )

# Callback para detalle del estudiante
@app.callback(
    Output('detalle-estudiante-container', 'children'),
    [Input('selector-estudiante', 'value')]
)
def update_student_detail(student_id):
    if not student_id:
        return html.P("Seleccione un estudiante para ver los detalles.", 
                     style={'color': '#7f8c8d', 'fontStyle': 'italic'})
    
    student_data = data_processor.get_student_detail(student_id)
    if not student_data:
        return html.P("Estudiante no encontrado.", style={'color': '#e74c3c'})
    
    # Generar reporte del estudiante
    report = report_generator.generate_student_report(student_data)
    
    # Crear layout del detalle
    detail_layout = html.Div([
        # Información básica
        html.Div(className='card', style={'marginBottom': '20px'}, children=[
            html.H3(f"👤 {report['student_info']['nombre_completo']}", 
                    style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div(style={'display': 'flex', 'gap': '30px', 'flexWrap': 'wrap'}, children=[
                html.Div([
                    html.Strong("ID: "), student_data['id_estudiante'], html.Br(),
                    html.Strong("Carrera: "), student_data['carrera'], html.Br(),
                    html.Strong("Semestre: "), student_data['semestre'], html.Br(),
                    html.Strong("Email: "), student_data['email']
                ]),
                html.Div([
                    html.Strong("Edad: "), student_data['edad'], html.Br(),
                    html.Strong("Género: "), student_data['genero'], html.Br(),
                    html.Strong("Estado: "), student_data['estado_academico'], html.Br(),
                    html.Strong("Teléfono: "), student_data['telefono']
                ])
            ])
        ]),
        
        # Métricas académicas
        html.Div(className='card', style={'marginBottom': '20px'}, children=[
            html.H4("📊 Resumen Académico", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div(style={'display': 'flex', 'gap': '20px', 'justifyContent': 'space-around'}, children=[
                html.Div(style={'textAlign': 'center'}, children=[
                    html.H3(str(student_data['promedio_general']), style={'color': '#3498db', 'margin': '0'}),
                    html.P("Promedio General", style={'margin': '5px 0'})
                ]),
                html.Div(style={'textAlign': 'center'}, children=[
                    html.H3(f"{student_data['asistencia_porcentaje']}%", style={'color': '#2ecc71', 'margin': '0'}),
                    html.P("Asistencia", style={'margin': '5px 0'})
                ]),
                html.Div(style={'textAlign': 'center'}, children=[
                    html.H3(f"{student_data['participacion_clase']}/5", style={'color': '#f39c12', 'margin': '0'}),
                    html.P("Participación", style={'margin': '5px 0'})
                ]),
                html.Div(style={'textAlign': 'center'}, children=[
                    html.H3(f"{student_data['horas_estudio_semanal']}h", style={'color': '#9b59b6', 'margin': '0'}),
                    html.P("Estudio Semanal", style={'margin': '5px 0'})
                ])
            ])
        ]),
        
        # Análisis de riesgo
        html.Div(className='card', style={'marginBottom': '20px'}, children=[
            html.H4("⚠️ Análisis de Riesgo", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div(style={
                'padding': '15px', 
                'backgroundColor': '#ffebee' if student_data['rendimiento_riesgo'] == 1 else '#e8f5e8',
                'borderRadius': '5px',
                'border': f"2px solid {'#e74c3c' if student_data['rendimiento_riesgo'] == 1 else '#2ecc71'}"
            }, children=[
                html.H5(
                    f"🔴 ESTUDIANTE EN RIESGO ALTO" if student_data['rendimiento_riesgo'] == 1 else "🟢 ESTUDIANTE SIN RIESGO",
                    style={'color': '#e74c3c' if student_data['rendimiento_riesgo'] == 1 else '#2ecc71', 'margin': '0 0 10px 0'}
                ),
                html.P(student_data.get('motivo_riesgo', 'Sin motivo específico'), style={'margin': '0'})
            ])
        ]),
        
        # Recomendaciones
        html.Div(className='card', children=[
            html.H4("💡 Recomendaciones", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.H6(rec['titulo'], style={'color': '#3498db', 'margin': '0 0 5px 0'}),
                    html.P(rec['descripcion'], style={'margin': '0 0 10px 0', 'color': '#7f8c8d'})
                ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'marginBottom': '10px'})
                for rec in report['recommendations']
            ])
        ])
    ])
    
    return detail_layout

# Callback para visualizaciones avanzadas
@app.callback(
    [Output('heatmap-correlaciones', 'figure'),
     Output('clustering-estudiantes', 'figure'),
     Output('distribucion-genero', 'figure'),
     Output('series-temporales', 'figure')],
    [Input('tabs-main', 'value')]
)
def update_advanced_visualizations(active_tab):
    if active_tab != 'tab-advanced':
        return {}, {}, {}, {}
    
    # Heatmap de correlaciones
    numeric_cols = ['calificaciones_anteriores', 'asistencia_porcentaje', 'participacion_clase', 
                   'horas_estudio_semanal', 'rendimiento_riesgo', 'edad', 'semestre']
    corr_matrix = df[numeric_cols].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        title="Matriz de Correlaciones",
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    
    # Clustering (simulado con scatter plot)
    fig_clustering = px.scatter(
        df,
        x='calificaciones_anteriores',
        y='asistencia_porcentaje',
        color='carrera',
        size='horas_estudio_semanal',
        title="Clustering de Estudiantes por Rendimiento"
    )
    
    # Distribución por género
    gender_risk = df.groupby(['genero', 'rendimiento_riesgo']).size().reset_index(name='count')
    fig_gender = px.bar(
        gender_risk,
        x='genero',
        y='count',
        color='rendimiento_riesgo',
        title="Distribución de Riesgo por Género",
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
    )
    
    # Series temporales (simuladas por semestre)
    temporal_data = df.groupby('semestre')['calificaciones_anteriores'].mean().reset_index()
    fig_temporal = px.line(
        temporal_data,
        x='semestre',
        y='calificaciones_anteriores',
        title="Evolución del Promedio por Semestre",
        markers=True
    )
    
    return fig_heatmap, fig_clustering, fig_gender, fig_temporal

# Callback para la predicción y lógica difusa con gráficos
@app.callback(
    [Output('resultado-prediccion', 'children'),
     Output('resultado-fuzzy', 'children'),
     Output('grafico-radar-estudiante', 'figure'),
     Output('grafico-barras-estudiante', 'figure'),
     Output('grafico-gauge-riesgo', 'figure'),
     Output('grafico-comparacion-promedio', 'figure')],
    Input('boton-predecir', 'n_clicks'),
    State('calificaciones_anteriores', 'value'),
    State('asistencia_porcentaje', 'value'),
    State('participacion_clase', 'value'),
    State('horas_estudio_semanal', 'value'),
    State('nivel_socioeconomico', 'value'),
    State('modelo-seleccionado', 'value')
)
def update_prediction(n_clicks, calif, asist, part, horas, socio, modelo):
    if n_clicks == 0:
        return "", "", {}, {}, {}, {}
    
    if any(v is None for v in [calif, asist, part, horas, socio]):
        return "❌ Error: Por favor, complete todos los campos.", "", {}, {}, {}, {}

    # Evaluar lógica difusa
    mapping_nivel = {'Bajo': 3, 'Medio': 6, 'Alto': 9}
    nivel_val = mapping_nivel.get(socio, 5)
    participacion_val = part * 2
    asistencia_val = asist
    calif_val = calif

    riesgo_fuzzy = fuzzy.evaluar_riesgo(nivel_val, participacion_val, asistencia_val, calif_val)
    if riesgo_fuzzy is not None:
        riesgo_fuzzy_str = f"🔬 Nivel de riesgo (lógica difusa): {riesgo_fuzzy:.2f} / 10"
    else:
        riesgo_fuzzy_str = "🔬 Nivel de riesgo (lógica difusa): Error en cálculo"

    # Preparar datos para predicción
    input_df = pd.DataFrame(
        [[calif, asist, part, horas, socio]],
        columns=['calificaciones_anteriores', 'asistencia_porcentaje', 'participacion_clase', 'horas_estudio_semanal', 'nivel_socioeconomico']
    )

    # Lógica de predicción con los modelos ML (manteniendo la original)
    pred_str = ""
    proba = [0.5, 0.5]  # Default values
    
    if modelo == 'dt':
        if model_dt is None:
            pred_str = "❌ Error: Modelo de Árbol de Decisión no cargado."
        else:
            try:
                pred = model_dt.predict(input_df)[0]
                proba = model_dt.predict_proba(input_df)[0]
            except Exception as e:
                pred_str = f"❌ Error en predicción Árbol de Decisión: {e}"
    elif modelo == 'keras':
        if model_keras is None:
            pred_str = "❌ Error: Modelo de Red Neuronal no cargado."
        else:
            try:
                ohe_vals = [0, 0, 0]
                if socio == 'Bajo':
                    ohe_vals[0] = 1
                elif socio == 'Medio':
                    ohe_vals[1] = 1
                elif socio == 'Alto':
                    ohe_vals[2] = 1
                
                calif_scaled = (calif - 7) / 2
                asist_scaled = (asist - 75) / 15
                part_scaled = (part - 3) / 1.5
                horas_scaled = (horas - 13) / 6

                input_keras = np.array([ohe_vals + [calif_scaled, asist_scaled, part_scaled, horas_scaled]])
                input_keras = input_keras.astype(np.float32)
                pred_proba = model_keras.predict(input_keras)
                pred = np.argmax(pred_proba, axis=1)[0]
                proba = pred_proba[0]
            except Exception as e:
                pred_str = f"❌ Error en predicción Red Neuronal: {e}"
    elif modelo == 'rf':
        if model_rf is None:
            pred_str = "❌ Error: Modelo Random Forest no cargado."
        else:
            try:
                pred = model_rf.predict(input_df)[0]
                proba = model_rf.predict_proba(input_df)[0]
            except Exception as e:
                pred_str = f"❌ Error en predicción Random Forest: {e}"
    else:
        pred_str = "❌ Modelo seleccionado no válido."

    # Si no hay error, generar el resultado
    if not pred_str.startswith("❌"):
        if pred == 1:
            prob_riesgo = proba[1] * 100
            pred_str = f"🔴 Resultado: Estudiante EN RIESGO (Confianza: {prob_riesgo:.2f}%)"
        else:
            prob_no_riesgo = proba[0] * 100
            pred_str = f"🟢 Resultado: Estudiante NO en riesgo (Confianza: {prob_no_riesgo:.2f}%)"

    # Crear gráficos de visualización
    # 1. Gráfico de radar del perfil del estudiante
    categories = ['Calificaciones', 'Asistencia', 'Participación', 'Horas Estudio', 'Nivel Socioeconómico']
    values = [
        calif,
        asist / 10,  # Normalizar a escala 0-10
        part * 2,    # Escalar de 1-5 a 2-10
        min(horas / 2.5, 10),  # Normalizar a escala 0-10
        nivel_val
    ]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Perfil del Estudiante',
        line_color='#3498db'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="📊 Perfil Académico del Estudiante"
    )

    # 2. Gráfico de barras comparativo
    metrics = ['Calificaciones', 'Asistencia %', 'Participación', 'Horas Estudio']
    student_values = [calif, asist, part, horas]
    avg_values = [
        df['calificaciones_anteriores'].mean(),
        df['asistencia_porcentaje'].mean(),
        df['participacion_clase'].mean(),
        df['horas_estudio_semanal'].mean()
    ]
    
    fig_bars = go.Figure(data=[
        go.Bar(name='Estudiante', x=metrics, y=student_values, marker_color='#3498db'),
        go.Bar(name='Promedio General', x=metrics, y=avg_values, marker_color='#95a5a6')
    ])
    fig_bars.update_layout(
        barmode='group',
        title="📈 Comparación con Promedio General",
        yaxis_title="Valores"
    )

    # 3. Gauge de nivel de riesgo
    riesgo_valor = riesgo_fuzzy if riesgo_fuzzy is not None else 5
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = riesgo_valor,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "🎯 Nivel de Riesgo (Lógica Difusa)"},
        delta = {'reference': 5},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "#e74c3c" if riesgo_valor > 6 else "#f39c12" if riesgo_valor > 3 else "#2ecc71"},
            'steps': [
                {'range': [0, 3], 'color': "#d5f4e6"},
                {'range': [3, 6], 'color': "#fef9e7"},
                {'range': [6, 10], 'color': "#fadbd8"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 7
            }
        }
    ))

    # 4. Gráfico de comparación con estudiantes similares
    # Filtrar estudiantes con características similares
    similar_students = df[
        (abs(df['calificaciones_anteriores'] - calif) <= 1) |
        (abs(df['asistencia_porcentaje'] - asist) <= 10) |
        (df['nivel_socioeconomico'] == socio)
    ]
    
    if len(similar_students) > 0:
        risk_comparison = similar_students['rendimiento_riesgo'].value_counts()
        labels = ['Sin Riesgo', 'En Riesgo']
        values = [risk_comparison.get(0, 0), risk_comparison.get(1, 0)]
        
        fig_comparison = px.pie(
            values=values,
            names=labels,
            title="👥 Estudiantes con Perfil Similar",
            color_discrete_map={'Sin Riesgo': '#2ecc71', 'En Riesgo': '#e74c3c'}
        )
    else:
        fig_comparison = go.Figure()
        fig_comparison.add_annotation(
            text="No hay estudiantes con perfil similar",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig_comparison.update_layout(title="👥 Estudiantes con Perfil Similar")

    return pred_str, riesgo_fuzzy_str, fig_radar, fig_bars, fig_gauge, fig_comparison

# Callback para manejo de subida de archivos
@app.callback(
    [Output('upload-status', 'children'),
     Output('validation-results', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def handle_file_upload(contents, filename):
    if contents is None:
        return "", ""
    
    try:
        # Decodificar el archivo
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Leer el archivo según su tipo
        if filename.endswith('.csv'):
            df_new = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith(('.xlsx', '.xls')):
            df_new = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div("❌ Formato de archivo no soportado. Use CSV o Excel.", 
                           style={'color': '#e74c3c'}), ""
        
        # Validar datos
        validation_errors = []
        required_cols = ['nombre', 'apellido', 'calificaciones_anteriores', 'asistencia_porcentaje']
        
        for col in required_cols:
            if col not in df_new.columns:
                validation_errors.append(f"Columna faltante: {col}")
        
        if validation_errors:
            error_list = html.Ul([html.Li(error) for error in validation_errors])
            return (
                html.Div("❌ Archivo subido con errores", style={'color': '#e74c3c'}),
                html.Div([
                    html.H4("Errores de validación:", style={'color': '#e74c3c'}),
                    error_list
                ])
            )
        
        # Crear alerta de archivo subido
        alert_system.create_file_upload_alert(filename, len(df_new))
        
        return (
            html.Div(f"✅ Archivo '{filename}' subido exitosamente ({len(df_new)} registros)", 
                    style={'color': '#2ecc71'}),
            html.Div([
                html.H4("Vista previa de los datos:", style={'color': '#2c3e50'}),
                dash_table.DataTable(
                    data=df_new.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df_new.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '8px'},
                    style_header={'backgroundColor': '#3498db', 'color': 'white'}
                )
            ])
        )
        
    except Exception as e:
        return html.Div(f"❌ Error procesando archivo: {str(e)}", 
                       style={'color': '#e74c3c'}), ""

# Callback para estadísticas de alertas
@app.callback(
    Output('estadisticas-alertas', 'children'),
    [Input('tabs-main', 'value')]
)
def update_alert_statistics(active_tab):
    if active_tab != 'tab-alerts':
        return []
    
    # Generar alertas para estudiantes en riesgo
    students_at_risk = df[df['rendimiento_riesgo'] == 1].head(10)
    for _, student in students_at_risk.iterrows():
        alert_system.check_student_alerts(student.to_dict())
    
    # Obtener estadísticas
    total_alerts = len(alert_system.alerts)
    critical_alerts = len([a for a in alert_system.alerts if a['priority'] == 'CRITICAL'])
    high_alerts = len([a for a in alert_system.alerts if a['priority'] == 'HIGH'])
    
    return html.Div(style={'display': 'flex', 'gap': '20px', 'justifyContent': 'space-around'}, children=[
        html.Div(style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#3498db', 'color': 'white', 'borderRadius': '8px'}, children=[
            html.H3(str(total_alerts), style={'margin': '0', 'fontSize': '2rem'}),
            html.P('Total Alertas', style={'margin': '5px 0'})
        ]),
        html.Div(style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#e74c3c', 'color': 'white', 'borderRadius': '8px'}, children=[
            html.H3(str(critical_alerts), style={'margin': '0', 'fontSize': '2rem'}),
            html.P('Críticas', style={'margin': '5px 0'})
        ]),
        html.Div(style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#f39c12', 'color': 'white', 'borderRadius': '8px'}, children=[
            html.H3(str(high_alerts), style={'margin': '0', 'fontSize': '2rem'}),
            html.P('Alta Prioridad', style={'margin': '5px 0'})
        ])
    ])

# Callback para lista de alertas
@app.callback(
    Output('lista-alertas', 'children'),
    [Input('btn-actualizar-alertas', 'n_clicks'),
     Input('tabs-main', 'value')],
    [State('filtro-tipo-alerta', 'value'),
     State('filtro-prioridad-alerta', 'value')]
)
def update_alerts_list(n_clicks, active_tab, tipo_filtro, prioridad_filtro):
    if active_tab != 'tab-alerts':
        return []
    
    # Obtener alertas filtradas
    alerts = alert_system.get_filtered_alerts(
        alert_type=tipo_filtro if tipo_filtro else None,
        priority=prioridad_filtro if prioridad_filtro else None,
        limit=20
    )
    
    if not alerts:
        return html.P("No hay alertas que coincidan con los filtros.", 
                     style={'color': '#7f8c8d', 'fontStyle': 'italic'})
    
    alert_items = []
    for alert in alerts:
        priority_colors = {
            'CRITICAL': '#e74c3c',
            'HIGH': '#f39c12',
            'MEDIUM': '#3498db',
            'LOW': '#95a5a6'
        }
        color = priority_colors.get(alert['priority'], '#95a5a6')
        
        alert_items.append(
            html.Div([
                html.Div([
                    html.Div([
                        html.Strong(alert['type_name'], style={'color': color, 'fontSize': '16px'}),
                        html.Span(f" - {alert['student_name']}", style={'marginLeft': '10px', 'fontSize': '14px'}),
                        html.Span(f" ({alert['priority']})", style={'marginLeft': '10px', 'fontSize': '12px', 'color': color})
                    ], style={'marginBottom': '5px'}),
                    html.P(alert['message'], style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '13px'}),
                    html.Small(f"Creada: {alert['created_at']}", style={'color': '#95a5a6'})
                ], style={
                    'padding': '15px', 
                    'border': f'1px solid {color}', 
                    'borderRadius': '8px', 
                    'marginBottom': '10px', 
                    'backgroundColor': '#f8f9fa',
                    'borderLeft': f'4px solid {color}'
                })
            ])
        )
    
    return alert_items

# Callback para generar reportes
@app.callback(
    Output('resultado-reportes', 'children'),
    [Input('btn-reporte-general', 'n_clicks'),
     Input('btn-simular-email', 'n_clicks')]
)
def handle_reports(n_clicks_pdf, n_clicks_email):
    ctx = callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'btn-reporte-general':
        try:
            # Generar reporte general
            report_data = {
                'total_students': len(df),
                'at_risk_students': len(df[df['rendimiento_riesgo'] == 1]),
                'average_grade': df['calificaciones_anteriores'].mean(),
                'average_attendance': df['asistencia_porcentaje'].mean(),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Simular generación de PDF
            pdf_path = report_generator.generate_general_report(report_data)
            
            return html.Div([
                html.H4("✅ Reporte PDF Generado", style={'color': '#27ae60'}),
                html.P(f"Archivo: {pdf_path}"),
                html.P(f"Total estudiantes: {report_data['total_students']}"),
                html.P(f"Estudiantes en riesgo: {report_data['at_risk_students']}"),
                html.P(f"Promedio general: {report_data['average_grade']:.2f}"),
                html.Small(f"Generado: {report_data['generated_at']}")
            ], style={'padding': '15px', 'backgroundColor': '#d5f4e6', 'borderRadius': '8px'})
            
        except Exception as e:
            return html.Div(f"❌ Error generando reporte: {str(e)}", 
                           style={'color': '#e74c3c', 'padding': '15px', 'backgroundColor': '#fadbd8', 'borderRadius': '8px'})
    
    elif button_id == 'btn-simular-email':
        try:
            # Simular envío de email
            students_at_risk = df[df['rendimiento_riesgo'] == 1].head(5)
            email_result = alert_system.simulate_email_alerts(students_at_risk.to_dict('records'))
            
            return html.Div([
                html.H4("📧 Simulación de de Envío de Emails", style={'color': '#f39c12'}),
                html.P(f"Emails enviados: {email_result['emails_sent']}"),
                html.P(f"Estudiantes notificados: {email_result['students_notified']}"),
                html.Ul([
                    html.Li(f"{student['nombre']} {student['apellido']} - {student['email']}")
                    for student in email_result['recipients']
                ])
            ], style={'padding': '15px', 'backgroundColor': '#fef9e7', 'borderRadius': '8px'})
            
        except Exception as e:
            return html.Div(f"❌ Error simulando emails: {str(e)}", 
                           style={'color': '#e74c3c', 'padding': '15px', 'backgroundColor': '#fadbd8', 'borderRadius': '8px'})
    
    return ""

# Punto de entrada para ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)

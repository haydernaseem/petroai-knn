import os
import base64
import io
import json
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import math
from werkzeug.utils import secure_filename

# استخدام backend غير تفاعلي لـ matplotlib
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# إعدادات CORS للسماح لـ Firebase Hosting
CORS(app, origins=[
    "https://petroai-web.web.app",
    "https://petroai-web.firebaseapp.com",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "https://petroai-iq.web.app",
    "https://petroai-iq.web.app/KNN.html"
    "https://petroai-iq.web.app/ploty.html"
])

# إعدادات التطبيق
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls', 'csv'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# =============================================================================
# وظائف مساعدة مشتركة - SHARED UTILITIES
# =============================================================================


def allowed_file(filename):
    """التحقق من نوع الملف"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


def clean_dataframe(df):
    """تنظيف DataFrame وتحويل قيم NaN"""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].apply(
                lambda x: None if pd.isna(x) else x)
        else:
            df_clean[col] = df_clean[col].apply(
                lambda x: None if pd.isna(x) else str(x))
    return df_clean


def convert_to_serializable(obj):
    """تحويل الكائنات غير القابلة للتسلسل"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if pd.isna(obj) or np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (bool, str)):
        return obj
    else:
        return str(obj)


def read_data_file(file):
    """قراءة ملف البيانات بناءً على نوعه"""
    filename = secure_filename(file.filename)
    if filename.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# =============================================================================
# KNN INTERPOLATION TOOL - مستقلة تماماً
# =============================================================================


class KNNInterpolationTool:
    """أداة KNN للاستيفاء المكاني"""

    def __init__(self):
        self.name = "knn_interpolation"
        self.version = "1.0"

    class ContourMap:
        def __init__(self, X, Y, Z, well_names=None):
            self.X = X
            self.Y = Y
            self.Z = Z
            self.well_names = well_names

        def knn_interpolation(self, target_x, target_y, n_neighbors=5):
            """KNN interpolation for a single point"""
            try:
                distances = []
                for i in range(len(self.X)):
                    dist = math.sqrt(
                        (self.X[i] - target_x)**2 + (self.Y[i] - target_y)**2)
                    distances.append((dist, self.Z[i]))

                distances.sort(key=lambda x: x[0])
                neighbors = distances[:n_neighbors]

                if neighbors[0][0] == 0:
                    return neighbors[0][1]

                weights = [1 / (dist + 1e-8) for dist, _ in neighbors]
                weighted_sum = sum(weight * z for (_, z),
                                   weight in zip(neighbors, weights))
                total_weight = sum(weights)

                return weighted_sum / total_weight
            except Exception as e:
                print(f"KNN interpolation error: {e}")
                return 0

        def create_grid(self, grid_size=100):
            """Create interpolation grid"""
            try:
                x_min, x_max = min(self.X), max(self.X)
                y_min, y_max = min(self.Y), max(self.Y)

                x_range = x_max - x_min
                y_range = y_max - y_min

                x_min -= 0.1 * x_range
                x_max += 0.1 * x_range
                y_min -= 0.1 * y_range
                y_max += 0.1 * y_range

                xi = np.linspace(x_min, x_max, grid_size)
                yi = np.linspace(y_min, y_max, grid_size)
                zi = np.zeros((grid_size, grid_size))

                for i in range(grid_size):
                    for j in range(grid_size):
                        zi[j, i] = self.knn_interpolation(xi[i], yi[j])

                return xi, yi, zi
            except Exception as e:
                print(f"Grid creation error: {e}")
                return None, None, None

        def generate_contour_plot(self, title="Contour Map"):
            """Generate contour plot"""
            try:
                xi, yi, zi = self.create_grid()
                if xi is None:
                    return None

                fig, ax = plt.subplots(figsize=(12, 10))

                contour = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
                ax.contour(xi, yi, zi, levels=20,
                           colors='black', linewidths=0.5)

                scatter = ax.scatter(
                    self.X, self.Y, c='red', s=50, edgecolors='black')

                if self.well_names:
                    for i, name in enumerate(self.well_names):
                        ax.annotate(name, (self.X[i], self.Y[i]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, fontweight='bold')

                plt.colorbar(contour, ax=ax, label='Value')
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)

                return base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Plot generation error: {e}")
                return None

    def upload_file(self, file):
        """معالجة رفع ملف لـ KNN"""
        try:
            if not file or file.filename == '':
                return {'error': 'No file selected'}, 400

            if not allowed_file(file.filename):
                return {'error': 'Invalid file type'}, 400

            df = read_data_file(file)
            df_clean = clean_dataframe(df)

            response_data = {
                'filename': secure_filename(file.filename),
                'columns': list(df_clean.columns),
                'numeric_columns': list(df_clean.select_dtypes(include=[np.number]).columns),
                'row_count': len(df_clean),
                'preview_data': df_clean.head(10).to_dict('records')
            }

            return response_data, 200

        except Exception as e:
            print(f"KNN upload error: {e}")
            return {'error': str(e)}, 500

    def interpolate(self, file, request_data):
        """تنفيذ الاستيفاء KNN"""
        try:
            df = read_data_file(file)

            x_col = request_data.get('x_column')
            y_col = request_data.get('y_column')
            z_col = request_data.get('z_column')
            well_name_col = request_data.get('well_name_column')
            grid_size = request_data.get('grid_size', 100)
            n_neighbors = request_data.get('n_neighbors', 5)

            if not all([x_col, y_col, z_col]):
                return {'error': 'Missing required columns'}, 400

            X = df[x_col].tolist()
            Y = df[y_col].tolist()
            Z = df[z_col].tolist()

            well_names = None
            if well_name_col and well_name_col in df.columns:
                well_names = df[well_name_col].tolist()

            contour_map = self.ContourMap(X, Y, Z, well_names)

            plot_base64 = contour_map.generate_contour_plot(
                title=f"KNN Interpolation - {z_col}"
            )

            if plot_base64:
                return {
                    'success': True,
                    'plot': f"data:image/png;base64,{plot_base64}",
                    'data_points': len(X)
                }, 200
            else:
                return {'error': 'Failed to generate contour plot'}, 500

        except Exception as e:
            print(f"KNN interpolation error: {e}")
            return {'error': str(e)}, 500

# =============================================================================
# POLYY MULTI-Y-AXIS CHARTS TOOL - مستقلة تماماً
# =============================================================================


class PolyYChartTool:
    """أداة PolyY للمخططات متعددة المحاور"""

    def __init__(self):
        self.name = "polyy_charts"
        self.version = "1.0"

    class ChartGenerator:
        def __init__(self):
            self.figure_data = None

        def generate_chart(self, df, x_column, y_columns, chart_title="PolyY Chart"):
            """Generate multi-Y-axis chart using Plotly"""
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                # Create figure with secondary y-axes
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Add traces for each y column
                for i, y_col in enumerate(y_columns):
                    column_name = y_col['column']
                    chart_type = y_col['type']
                    color = y_col['color']

                    if column_name not in df.columns:
                        continue

                    # Create trace based on chart type
                    if chart_type == 'line':
                        trace = go.Scatter(
                            x=df[x_column],
                            y=df[column_name],
                            name=column_name,
                            line=dict(color=color),
                            yaxis=f"y{i+1}" if i > 0 else "y"
                        )
                    elif chart_type == 'scatter':
                        trace = go.Scatter(
                            x=df[x_column],
                            y=df[column_name],
                            mode='markers',
                            name=column_name,
                            marker=dict(color=color),
                            yaxis=f"y{i+1}" if i > 0 else "y"
                        )
                    elif chart_type == 'area':
                        trace = go.Scatter(
                            x=df[x_column],
                            y=df[column_name],
                            fill='tozeroy',
                            name=column_name,
                            line=dict(color=color),
                            yaxis=f"y{i+1}" if i > 0 else "y"
                        )
                    else:
                        continue

                    fig.add_trace(trace, secondary_y=(i > 0))

                # Update layout
                fig.update_layout(
                    title=chart_title,
                    xaxis_title=x_column,
                    template="plotly_dark",
                    height=600,
                    showlegend=True
                )

                # Update y-axes
                for i, y_col in enumerate(y_columns):
                    if i == 0:
                        fig.update_yaxes(
                            title_text=y_col['column'], secondary_y=False)
                    else:
                        fig.update_yaxes(
                            title_text=y_col['column'], secondary_y=True)

                # Convert to HTML
                plot_html = fig.to_html(
                    include_plotlyjs='cdn', config={'responsive': True})

                # Save files
                os.makedirs('static/polyy', exist_ok=True)

                # Save HTML file
                html_file_path = f'static/polyy/chart_{len(os.listdir("static/polyy"))}.html'
                with open(html_file_path, 'w', encoding='utf-8') as f:
                    f.write(plot_html)

                # Save PNG file
                png_file_path = f'static/polyy/chart_{len(os.listdir("static/polyy"))}.png'
                fig.write_image(png_file_path)

                return {
                    'plot_html': plot_html,
                    'html_file': f'/static/polyy/{os.path.basename(html_file_path)}',
                    'png_file': f'/static/polyy/{os.path.basename(png_file_path)}'
                }

            except Exception as e:
                print(f"PolyY chart generation error: {e}")
                return None

    def upload_file(self, file):
        """معالجة رفع ملف لـ PolyY"""
        try:
            if not file or file.filename == '':
                return {'error': 'No file selected'}, 400

            if not allowed_file(file.filename):
                return {'error': 'Invalid file type'}, 400

            df = read_data_file(file)
            df_clean = clean_dataframe(df)

            response_data = {
                'filename': secure_filename(file.filename),
                'all_columns': list(df_clean.columns),
                'numeric_columns': list(df_clean.select_dtypes(include=[np.number]).columns),
                'row_count': len(df_clean),
                'preview_data': df_clean.head(10).applymap(convert_to_serializable).to_dict('records')
            }

            return response_data, 200

        except Exception as e:
            print(f"PolyY upload error: {e}")
            return {'error': str(e)}, 500

    def generate_chart(self, file, request_data):
        """توليد مخطط PolyY"""
        try:
            x_column = request_data.get('x_column')
            chart_title = request_data.get('chart_title', 'PolyY Chart')
            y_columns_json = request_data.get('y_columns')

            if not y_columns_json:
                return {'error': 'No Y columns specified'}, 400

            try:
                y_columns = json.loads(y_columns_json)
            except json.JSONDecodeError:
                return {'error': 'Invalid Y columns format'}, 400

            df = read_data_file(file)

            # تنظيف البيانات
            df_clean = df.copy()
            for col in df_clean.columns:
                if df_clean[col].dtype in ['float64', 'int64']:
                    df_clean[col] = df_clean[col].fillna(0)
                else:
                    df_clean[col] = df_clean[col].fillna('')

            # توليد المخطط
            chart_generator = self.ChartGenerator()
            result = chart_generator.generate_chart(
                df_clean, x_column, y_columns, chart_title)

            if result:
                return {
                    'success': True,
                    **result
                }, 200
            else:
                return {'error': 'Failed to generate chart'}, 500

        except Exception as e:
            print(f"PolyY generation error: {e}")
            return {'error': str(e)}, 500

# =============================================================================
# تهيئة الأدوات - TOOLS INITIALIZATION
# =============================================================================


# إنشاء نسخ من الأدوات
knn_tool = KNNInterpolationTool()
polyy_tool = PolyYChartTool()

# =============================================================================
# ROUTES - مسارات API
# =============================================================================


@app.route('/')
def home():
    return jsonify({
        "message": "PetroAI Backend Server",
        "version": "1.0.0",
        "endpoints": {
            "knn": "/knn/upload, /knn/interpolate",
            "polyy": "/polyy/upload, /polyy/generate"
        }
    })

# =============================================================================
# KNN Routes - مسارات KNN
# =============================================================================


@app.route('/knn/upload', methods=['POST'])
def knn_upload_route():
    """رفع ملف البيانات لـ KNN"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    result, status_code = knn_tool.upload_file(file)
    return jsonify(result), status_code


@app.route('/knn/interpolate', methods=['POST'])
def knn_interpolate_route():
    """KNN interpolation endpoint"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    request_data = request.form.to_dict()

    result, status_code = knn_tool.interpolate(file, request_data)
    return jsonify(result), status_code

# =============================================================================
# POLYY Routes - مسارات PolyY
# =============================================================================


@app.route('/polyy/upload', methods=['POST'])
def polyy_upload_route():
    """رفع ملف البيانات لـ PolyY"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    result, status_code = polyy_tool.upload_file(file)
    return jsonify(result), status_code


@app.route('/polyy/generate', methods=['POST'])
def polyy_generate_route():
    """توليد مخطط PolyY متعدد المحاور"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    request_data = request.form.to_dict()

    result, status_code = polyy_tool.generate_chart(file, request_data)
    return jsonify(result), status_code

# =============================================================================
# HEALTH CHECK - فحص الحالة
# =============================================================================


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "PetroAI Backend",
        "timestamp": pd.Timestamp.now().isoformat(),
        "tools": {
            "knn": knn_tool.version,
            "polyy": polyy_tool.version
        }
    })

# =============================================================================
# تشغيل التطبيق - APPLICATION STARTUP
# =============================================================================


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

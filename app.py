import os
import base64
import io
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import math
from werkzeug.utils import secure_filename

# استخدام backend غير تفاعلي لـ matplotlib
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

# إعدادات التطبيق
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


def allowed_file(filename):
    """التحقق من نوع الملف"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


def clean_dataframe(df):
    """تنظيف DataFrame وتحويل قيم NaN إلى قيم مناسبة لـ JSON"""
    df_clean = df.copy()

    # تحويل قيم NaN في جميع الأعمدة
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].apply(
                lambda x: None if pd.isna(x) else x)
        else:
            df_clean[col] = df_clean[col].apply(
                lambda x: None if pd.isna(x) else str(x))

    return df_clean


def convert_to_serializable(obj):
    """تحويل الكائنات غير القابلة للتسلسل إلى قيم قابلة للتسلسل"""
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


class ContourMap:
    def __init__(self, X, Y, Z, well_names=None):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.well_names = well_names

    def knn_interpolation(self, target_x, target_y, n_neighbors=5):
        """KNN interpolation for a single point"""
        try:
            # حساب المسافات
            distances = np.sqrt((self.X - target_x)**2 +
                                (self.Y - target_y)**2)

            # الحصول على أقرب الجيران
            nearest_indices = np.argsort(distances)[:n_neighbors]

            # تجنب القسمة على الصفر
            weights = 1 / (distances.iloc[nearest_indices] + 1e-8)

            # حساب القيمة المستوفاة
            interpolated_value = np.sum(
                weights * self.Z.iloc[nearest_indices]) / np.sum(weights)

            # التأكد من أن القيمة رقمية
            if interpolated_value is None or np.isnan(interpolated_value):
                return 0.0
            return float(interpolated_value)

        except Exception as e:
            print(f"KNN interpolation error: {e}")
            return 0.0  # قيمة افتراضية في حالة الخطأ

    def plot_knn_map(self, target_x=None, target_y=None, n_neighbors=5, title="KNN Interpolation"):
        """Create contour map with optional target point"""
        try:
            fig, ax = plt.subplots(figsize=(12, 9))

            # إنشاء شبكة لرسم الكنتور
            x_range = self.X.max() - self.X.min()
            y_range = self.Y.max() - self.Y.min()

            # تحديد عدد النقاط بناءً على نطاق البيانات
            x_points = min(
                80, max(25, int(80 * x_range / (x_range + y_range))))
            y_points = min(
                80, max(25, int(80 * y_range / (x_range + y_range))))

            xi = np.linspace(self.X.min() - 0.05*x_range,
                             self.X.max() + 0.05*x_range, x_points)
            yi = np.linspace(self.Y.min() - 0.05*y_range,
                             self.Y.max() + 0.05*y_range, y_points)
            xi, yi = np.meshgrid(xi, yi)

            # الاستيفاء لنقاط الشبكة
            zi = np.zeros(xi.shape)
            for i in range(xi.shape[0]):
                for j in range(xi.shape[1]):
                    try:
                        zi[i, j] = self.knn_interpolation(
                            xi[i, j], yi[i, j], n_neighbors)
                    except:
                        zi[i, j] = np.nan

            # إنشاء رسم الكنتور
            contour = ax.contourf(xi, yi, zi, levels=15,
                                  alpha=0.8, cmap='viridis')
            contour_lines = ax.contour(xi, yi, zi, levels=12, linewidths=0.8,
                                       colors='black', alpha=0.5)
            ax.clabel(contour_lines, inline=True, fontsize=7, fmt='%.1f')

            # رسم نقاط البيانات الأصلية
            scatter = ax.scatter(self.X, self.Y, c=self.Z, s=60,
                                 edgecolors='white', linewidth=1, cmap='viridis')

            # رسم نقطة الهدف إذا تم توفيرها
            if target_x is not None and target_y is not None:
                try:
                    target_z = self.knn_interpolation(
                        target_x, target_y, n_neighbors)
                    ax.scatter([target_x], [target_y], c='red', s=150,
                               marker='*', edgecolors='black', linewidth=1.5,
                               label=f'Target Point\nZ = {target_z:.2f}')
                    ax.legend(loc='upper right', fontsize=9)
                except:
                    pass

            # إضافة أسماء الآبار إذا كانت متاحة
            if self.well_names is not None:
                for i, name in enumerate(self.well_names):
                    if name and i < len(self.X):
                        try:
                            ax.annotate(str(name), (self.X.iloc[i], self.Y.iloc[i]),
                                        xytext=(5, 5), textcoords='offset points',
                                        fontsize=6, alpha=0.7,
                                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
                        except:
                            pass

            plt.colorbar(contour, label='Z Value', shrink=0.8)
            ax.set_xlabel('X Coordinate', fontsize=11, fontweight='bold')
            ax.set_ylabel('Y Coordinate', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.2)

            # تحسين المظهر
            ax.ticklabel_format(useOffset=False, style='plain')
            plt.tight_layout()

            # حفظ الرسم إلى bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                        facecolor='#1a1a2e', edgecolor='none')
            buf.seek(0)
            plt.close(fig)

            return buf

        except Exception as e:
            print(f"Plotting error: {e}")
            # إنشاء رسم بديل في حالة الخطأ
            fig, ax = plt.subplots(figsize=(10, 7))
            scatter = ax.scatter(self.X, self.Y, c=self.Z, s=50, cmap='viridis',
                                 edgecolors='white', linewidth=0.8)
            plt.colorbar(scatter, label='Z Value')
            ax.set_xlabel('X Coordinate', fontweight='bold')
            ax.set_ylabel('Y Coordinate', fontweight='bold')
            ax.set_title('Data Points Distribution', fontweight='bold')
            ax.grid(True, alpha=0.2)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                        facecolor='#1a1a2e', edgecolor='none')
            buf.seek(0)
            plt.close(fig)
            return buf


@app.route('/')
def serve_index():
    return send_from_directory('.', 'knn.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return available columns"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload Excel files only.'}), 400

        # Read Excel file
        try:
            df = pd.read_excel(file)
        except Exception as e:
            return jsonify({'error': f'Error reading Excel file: {str(e)}'}), 400

        if df.empty:
            return jsonify({'error': 'The uploaded file is empty'}), 400

        # تنظيف البيانات
        df_clean = clean_dataframe(df)

        # Return column information
        numeric_cols = df_clean.select_dtypes(
            include=[np.number]).columns.tolist()
        text_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        all_cols = df_clean.columns.tolist()

        # تحويل المعاينة إلى شكل قابل للتسلسل
        preview_data = []
        for _, row in df_clean.head().iterrows():
            preview_data.append(
                {col: convert_to_serializable(row[col]) for col in all_cols})

        columns = {
            'numeric_columns': numeric_cols,
            'text_columns': text_cols,
            'all_columns': all_cols,
            'preview': preview_data,
            'total_rows': len(df_clean)
        }

        return jsonify(columns)

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/interpolate', methods=['POST'])
def interpolate():
    """Perform KNN interpolation and return results"""
    try:
        # التحقق من وجود ملف
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # الحصول على البيانات من form
        x_col = request.form.get('x_column')
        y_col = request.form.get('y_column')
        z_col = request.form.get('z_column')
        well_col = request.form.get('well_column', '')
        target_x = request.form.get('target_x')
        target_y = request.form.get('target_y')
        n_neighbors = request.form.get('n_neighbors', '5')

        # التحقق من البيانات المطلوبة
        if not all([x_col, y_col, z_col, target_x, target_y]):
            return jsonify({'error': 'Missing required parameters'}), 400

        try:
            target_x = float(target_x)
            target_y = float(target_y)
            n_neighbors = int(n_neighbors)
        except ValueError:
            return jsonify({'error': 'Invalid coordinate or neighbor values'}), 400

        # قراءة وتنظيف البيانات
        try:
            df = pd.read_excel(file)
            df_clean = clean_dataframe(df)
        except Exception as e:
            return jsonify({'error': f'Error reading data: {str(e)}'}), 400

        # التحقق من الأعمدة
        for col in [x_col, y_col, z_col]:
            if col not in df_clean.columns:
                return jsonify({'error': f'Column {col} not found in data'}), 400

        # إزالة الصفوف التي تحتوي على قيم مفقودة في الأعمدة المطلوبة
        df_filtered = df_clean.dropna(subset=[x_col, y_col, z_col])

        if len(df_filtered) == 0:
            return jsonify({'error': 'No valid data points after removing missing values'}), 400

        if len(df_filtered) < n_neighbors:
            return jsonify({'error': f'Not enough data points. Need at least {n_neighbors}, but only {len(df_filtered)} available'}), 400

        # استخراج البيانات
        X = pd.to_numeric(df_filtered[x_col], errors='coerce').dropna()
        Y = pd.to_numeric(df_filtered[y_col], errors='coerce').dropna()
        Z = pd.to_numeric(df_filtered[z_col], errors='coerce').dropna()

        # التأكد من أن جميع المصفوفات لها نفس الطول
        min_length = min(len(X), len(Y), len(Z))
        X = X.iloc[:min_length]
        Y = Y.iloc[:min_length]
        Z = Z.iloc[:min_length]

        well_names = None
        if well_col and well_col in df_filtered.columns:
            well_names = df_filtered[well_col].astype(str).tolist()[
                :min_length]
        else:
            # إنشاء أسماء آبار افتراضية إذا لم يتم توفيرها
            well_names = [f"Well_{i+1}" for i in range(min_length)]

        # إنشاء خريطة الكنتور
        contour_map = ContourMap(X, Y, Z, well_names)

        # تنفيذ الاستيفاء
        interpolated_z = contour_map.knn_interpolation(
            target_x, target_y, n_neighbors)

        # التأكد من أن القيمة رقمية
        if interpolated_z is None or np.isnan(interpolated_z):
            interpolated_z = 0.0
        else:
            interpolated_z = float(interpolated_z)

        # إنشاء الرسم
        plot_buffer = contour_map.plot_knn_map(
            target_x, target_y, n_neighbors,
            f"KNN Interpolation Map\nTarget: ({target_x:.1f}, {target_y:.1f}) | k={n_neighbors}"
        )

        # تحويل الرسم إلى base64
        plot_base64 = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')

        # الحصول على معلومات الجيران الأقرب
        distances = np.sqrt((X - target_x)**2 + (Y - target_y)**2)
        nearest_indices = np.argsort(distances)[:n_neighbors]
        nearest_neighbors = []

        for idx in nearest_indices:
            if idx < len(X):
                well_name = well_names[idx] if well_names else f"Well_{idx+1}"
                neighbor_info = {
                    'x': convert_to_serializable(X.iloc[idx]),
                    'y': convert_to_serializable(Y.iloc[idx]),
                    'z': convert_to_serializable(Z.iloc[idx]),
                    'distance': convert_to_serializable(distances.iloc[idx]),
                    'well_name': convert_to_serializable(well_name)
                }
                nearest_neighbors.append(neighbor_info)

        response = {
            'success': True,
            'interpolated_value': interpolated_z,
            'target_x': target_x,
            'target_y': target_y,
            'n_neighbors': n_neighbors,
            'nearest_neighbors': nearest_neighbors,
            'contour_plot': f"data:image/png;base64,{plot_base64}",
            'total_data_points': len(X)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Interpolation error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'PetroAI KNN Interpolation',
        'version': '1.0'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

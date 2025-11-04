import os
import base64
import io
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import math
import re
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
])

# إعدادات التطبيق
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls', 'csv'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


def allowed_file(filename):
    """التحقق من نوع الملف"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


def smart_column_detection(df):
    """اكتشاف ذكي لنوع البيانات في الأعمدة"""
    column_types = {}
    column_stats = {}

    for col in df.columns:
        # استخراج القيم الرقمية من الأعمدة المختلطة
        numeric_values = []
        text_values = []
        mixed_details = []

        for idx, value in enumerate(df[col].dropna()):
            if isinstance(value, (int, float)) and not pd.isna(value):
                numeric_values.append(float(value))
                mixed_details.append(
                    {'index': idx, 'value': value, 'type': 'numeric'})
            elif isinstance(value, str):
                # البحث عن أرقام في النصوص
                numbers = re.findall(r'-?\d+\.?\d*', value)
                if numbers:
                    numeric_val = float(numbers[0])
                    numeric_values.append(numeric_val)
                    mixed_details.append(
                        {'index': idx, 'value': value, 'extracted': numeric_val, 'type': 'text_with_number'})
                else:
                    text_values.append(value)
                    mixed_details.append(
                        {'index': idx, 'value': value, 'type': 'text_only'})

        # تحديد نوع العمول
        numeric_ratio = len(numeric_values) / \
            len(df[col]) if len(df[col]) > 0 else 0

        if numeric_ratio >= 0.8:
            col_type = 'numeric'
        elif numeric_ratio >= 0.4:
            col_type = 'mixed_numeric'
        else:
            col_type = 'text'

        column_types[col] = col_type
        column_stats[col] = {
            'numeric_count': len(numeric_values),
            'text_count': len(text_values),
            'total_count': len(df[col]),
            'numeric_ratio': numeric_ratio,
            'mixed_details': mixed_details[:10]  # أول 10 تفاصيل فقط
        }

    return column_types, column_stats


def extract_numeric_values(series):
    """استخراج القيم الرقمية من سلسلة تحتوي على نصوص وأرقام"""
    numeric_values = []
    indices = []

    for idx, value in enumerate(series):
        if pd.isna(value):
            continue

        if isinstance(value, (int, float)):
            numeric_values.append(float(value))
            indices.append(idx)
        elif isinstance(value, str):
            # البحث عن أول رقم في النص
            numbers = re.findall(r'-?\d+\.?\d*', value)
            if numbers:
                numeric_values.append(float(numbers[0]))
                indices.append(idx)

    return pd.Series(numeric_values, index=indices) if numeric_values else pd.Series(dtype=float)


def advanced_data_analysis(values):
    """تحليل متقدم للبيانات لتحديد النمط السائد"""
    if len(values) == 0:
        return {'pattern': 'unknown', 'confidence': 0.0}

    values_series = pd.Series(values)
    valid_values = values_series.dropna()

    if len(valid_values) == 0:
        return {'pattern': 'unknown', 'confidence': 0.0}

    # تحليل متقدم لأنواع البيانات
    pure_fractions = [x for x in valid_values if 0 < x < 1]  # كسور نقية
    likely_fractions = [x for x in valid_values if 0 < x < 2]  # كسور محتملة
    small_percentages = [x for x in valid_values if 1 <= x <= 20]  # نسب صغيرة
    medium_percentages = [
        x for x in valid_values if 20 < x <= 100]  # نسب متوسطة
    large_percentages = [
        x for x in valid_values if 100 < x <= 1000]  # نسب كبيرة
    very_large_numbers = [
        x for x in valid_values if x > 1000]  # أعداد كبيرة جداً

    # الإحصاءات
    stats = {
        'min': valid_values.min(),
        'max': valid_values.max(),
        'mean': valid_values.mean(),
        'std': valid_values.std(),
        'q25': valid_values.quantile(0.25),
        'q75': valid_values.quantile(0.75)
    }

    total_count = len(valid_values)

    # حساب النسب
    ratios = {
        'pure_fractions': len(pure_fractions) / total_count,
        'likely_fractions': len(likely_fractions) / total_count,
        'small_percentages': len(small_percentages) / total_count,
        'medium_percentages': len(medium_percentages) / total_count,
        'large_percentages': len(large_percentages) / total_count,
        'very_large': len(very_large_numbers) / total_count
    }

    # تحديد النمط السائد
    patterns = []

    # نمط الكسور
    if ratios['pure_fractions'] >= 0.6:
        patterns.append(('fractions', ratios['pure_fractions'] + 0.2))
    elif ratios['likely_fractions'] >= 0.7:
        patterns.append(('fractions', ratios['likely_fractions']))

    # نمط النسب المئوية
    if ratios['medium_percentages'] >= 0.6:
        patterns.append(('percentages', ratios['medium_percentages'] + 0.15))
    elif (ratios['small_percentages'] + ratios['medium_percentages']) >= 0.65:
        patterns.append(
            ('percentages', (ratios['small_percentages'] + ratios['medium_percentages']) * 0.9))

    # نمط الأعداد الكبيرة
    if ratios['very_large'] >= 0.5:
        patterns.append(('large_numbers', ratios['very_large']))

    # تحليل إحصائي إضافي
    if stats['mean'] < 0.5 and stats['max'] < 1.0:
        patterns.append(('fractions', 0.8))
    elif 1 < stats['mean'] < 100 and stats['max'] <= 100:
        patterns.append(('percentages', 0.7))

    # اختيار النمط الأكثر ثقة
    if patterns:
        best_pattern = max(patterns, key=lambda x: x[1])
        return {'pattern': best_pattern[0], 'confidence': best_pattern[1], 'stats': stats, 'ratios': ratios}
    else:
        # إذا لم يكن هناك نمط واضح، نستخدم الإحصاءات
        if stats['mean'] < 1.0:
            return {'pattern': 'fractions', 'confidence': 0.6, 'stats': stats, 'ratios': ratios}
        elif stats['mean'] < 100:
            return {'pattern': 'percentages', 'confidence': 0.6, 'stats': stats, 'ratios': ratios}
        else:
            return {'pattern': 'large_numbers', 'confidence': 0.5, 'stats': stats, 'ratios': ratios}


def smart_normalize_data(values):
    """تطبيع ذكي للبيانات - الغالبية تحكم الأقلية بذكاء عالي"""
    if len(values) == 0:
        return values, {'action': 'none', 'reason': 'no_data'}

    values_series = pd.Series(values)
    valid_values = values_series.dropna()

    if len(valid_values) == 0:
        return values, {'action': 'none', 'reason': 'no_valid_data'}

    # تحليل متقدم للبيانات
    analysis = advanced_data_analysis(values)
    pattern = analysis['pattern']
    confidence = analysis['confidence']
    stats = analysis['stats']

    print(
        f"🧠 Smart Analysis - Pattern: {pattern}, Confidence: {confidence:.2f}")
    print(
        f"📊 Stats - Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Mean: {stats['mean']:.4f}")
    print(f"📈 Ratios - {analysis['ratios']}")

    # قرارات الذكاء الاصطناعي
    if pattern == 'fractions' and confidence >= 0.6:
        print("✅ Decision: Converting all to FRACTIONS (majority rules)")
        result = convert_to_fractions(values_series)
        return result.tolist(), {
            'action': 'to_fractions',
            'confidence': confidence,
            'original_range': [stats['min'], stats['max']],
            'new_range': [result.min(), result.max()]
        }

    elif pattern == 'percentages' and confidence >= 0.6:
        print("✅ Decision: Converting all to PERCENTAGES (majority rules)")
        result = convert_to_percentages(values_series)
        return result.tolist(), {
            'action': 'to_percentages',
            'confidence': confidence,
            'original_range': [stats['min'], stats['max']],
            'new_range': [result.min(), result.max()]
        }

    elif pattern == 'large_numbers' and confidence >= 0.5:
        print("✅ Decision: Keeping as LARGE NUMBERS")
        return values, {
            'action': 'keep_large',
            'confidence': confidence,
            'reason': 'large_numbers_detected'
        }

    else:
        print("✅ Decision: No conversion needed - keeping original values")
        return values, {
            'action': 'none',
            'confidence': confidence,
            'reason': 'no_clear_pattern'
        }


def convert_to_fractions(series):
    """تحويل جميع القيم إلى كسور"""
    result = series.copy()
    for i, val in enumerate(result):
        if not pd.isna(val):
            if val >= 1.0:
                # إذا كانت قيمة كبيرة، نقسمها على 100
                result.iloc[i] = val / 100.0
            # إذا كانت بالفعل كسر، نتركها كما هي
    return result


def convert_to_percentages(series):
    """تحويل جميع القيم إلى نسب مئوية"""
    result = series.copy()
    for i, val in enumerate(result):
        if not pd.isna(val):
            if val < 1.0:
                # إذا كانت كسر، نضربها في 100
                result.iloc[i] = val * 100.0
            # إذا كانت بالفعل نسبة مئوية، نتركها كما هي
    return result


def auto_detect_coordinate_columns(df, column_types):
    """اكتشاف تلقائي ذكي لأعمدة الإحداثيات"""
    potential_columns = {
        'x': [], 'y': [], 'z': []
    }

    for col, col_type in column_types.items():
        if col_type in ['numeric', 'mixed_numeric']:
            numeric_values = extract_numeric_values(df[col])
            if len(numeric_values) > 0:
                stats = {
                    'min': numeric_values.min(),
                    'max': numeric_values.max(),
                    'range': numeric_values.max() - numeric_values.min(),
                    'mean': numeric_values.mean(),
                    'std': numeric_values.std()
                }

                # تحليل النمط لتحديد نوع العمود
                col_analysis = advanced_data_analysis(numeric_values.tolist())

                # إحداثيات X (عادة نطاق واسع، قيم كبيرة)
                if stats['range'] > 1000 and stats['min'] >= 0 and col_analysis['pattern'] == 'large_numbers':
                    score = stats['range'] / 1000 + col_analysis['confidence']
                    potential_columns['x'].append((col, score, stats))

                # إحداثيات Y (نطاق واسع أيضاً)
                elif stats['range'] > 1000 and stats['min'] >= 0:
                    score = stats['range'] / 1000 + col_analysis['confidence']
                    potential_columns['y'].append((col, score, stats))

                # قيم Z (عادة نطاق أصغر، كسور أو نسب)
                elif stats['range'] < 1000 or col_analysis['pattern'] in ['fractions', 'percentages']:
                    score = (1 / (stats['range'] + 1)) + \
                        col_analysis['confidence']
                    potential_columns['z'].append((col, score, stats))

    # ترتيب حسب الأفضلية واختيار الأفضل
    result = {}
    for coord_type, candidates in potential_columns.items():
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            result[coord_type] = candidates[0][0]
            print(
                f"🎯 Auto-detected {coord_type.upper()}: {candidates[0][0]} (score: {candidates[0][1]:.2f})")
        else:
            result[coord_type] = None

    return result


def clean_dataframe(df):
    """تنظيف وتحسين DataFrame بذكاء"""
    df_clean = df.copy()

    # اكتشاف أنواع الأعملة بذكاء
    column_types, column_stats = smart_column_detection(df_clean)

    print("🔍 Column Analysis Results:")
    for col, col_type in column_types.items():
        stats = column_stats[col]
        print(
            f"   {col}: {col_type} (numeric: {stats['numeric_count']}/{stats['total_count']})")

    # معالجة كل عميل حسب نوعه
    normalization_report = {}

    for col in df_clean.columns:
        if column_types.get(col) in ['numeric', 'mixed_numeric']:
            # استخراج القيم الرقمية
            numeric_series = extract_numeric_values(df_clean[col])
            if len(numeric_series) > 0:
                # تطبيع ذكي للبيانات
                normalized_values, normalization_info = smart_normalize_data(
                    numeric_series.tolist())
                normalization_report[col] = normalization_info

                # تحديث العميل بالقيم المعالجة
                new_series = pd.Series([None] * len(df_clean))
                valid_indices = numeric_series.index
                for i, idx in enumerate(valid_indices):
                    if i < len(normalized_values):
                        new_series[idx] = normalized_values[i]

                df_clean[col] = new_series

    print("📋 Normalization Report:")
    for col, report in normalization_report.items():
        print(
            f"   {col}: {report['action']} (confidence: {report.get('confidence', 0):.2f})")

    return df_clean, column_types, normalization_report


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
            return 0.0

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
            ax.clabel(contour_lines, inline=True, fontsize=7, fmt='%.3f')

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
                               label=f'Target Point\nZ = {target_z:.4f}')
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
            return jsonify({'error': 'Invalid file type. Please upload Excel or CSV files only.'}), 400

        # Read file based on extension
        try:
            if file.filename.lower().endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400

        if df.empty:
            return jsonify({'error': 'The uploaded file is empty'}), 400

        # تنظيف البيانات واكتشاف الأنواع
        df_clean, column_types, normalization_report = clean_dataframe(df)

        # الاكتشاف التلقائي للإحداثيات
        auto_columns = auto_detect_coordinate_columns(df_clean, column_types)

        # Return column information
        numeric_cols = [col for col, col_type in column_types.items()
                        if col_type in ['numeric', 'mixed_numeric']]
        text_cols = [col for col, col_type in column_types.items()
                     if col_type == 'text']
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
            'total_rows': len(df_clean),
            'auto_detected_columns': auto_columns,
            'column_types': column_types,
            'normalization_report': normalization_report
        }

        return jsonify(columns)

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/interpolate', methods=['POST'])
def interpolate():
    """Perform KNN interpolation and return results"""
    try:
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
            if file.filename.lower().endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            df_clean, column_types, normalization_report = clean_dataframe(df)
        except Exception as e:
            return jsonify({'error': f'Error reading data: {str(e)}'}), 400

        # التحقق من الأعمدة
        for col in [x_col, y_col, z_col]:
            if col not in df_clean.columns:
                return jsonify({'error': f'Column {col} not found in data'}), 400

        # استخراج البيانات الرقمية من الأعمدة
        X = extract_numeric_values(df_clean[x_col])
        Y = extract_numeric_values(df_clean[y_col])
        Z = extract_numeric_values(df_clean[z_col])

        # تطبيع ذكي لقيم Z
        Z_normalized, normalization_info = smart_normalize_data(Z.tolist())
        Z = pd.Series(Z_normalized)

        # التأكد من أن جميع المصفوفات لها نفس الطول
        min_length = min(len(X), len(Y), len(Z))
        X = X.iloc[:min_length]
        Y = Y.iloc[:min_length]
        Z = Z.iloc[:min_length]

        if len(X) == 0:
            return jsonify({'error': 'No valid numeric data found in the specified columns'}), 400

        if len(X) < n_neighbors:
            return jsonify({'error': f'Not enough data points. Need at least {n_neighbors}, but only {len(X)} available'}), 400

        well_names = None
        if well_col and well_col in df_clean.columns:
            well_names = df_clean[well_col].astype(str).tolist()[:min_length]
        else:
            well_names = [f"Well_{i+1}" for i in range(min_length)]

        # إنشاء خريطة الكنتور
        contour_map = ContourMap(X, Y, Z, well_names)

        # تنفيذ الاستيفاء
        interpolated_z = contour_map.knn_interpolation(
            target_x, target_y, n_neighbors)

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
            'total_data_points': len(X),
            'data_normalization_applied': normalization_info['action'] != 'none',
            'normalization_details': normalization_info
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Interpolation error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'PetroAI KNN Interpolation',
        'version': '2.0',
        'features': 'Advanced AI data normalization, CSV/Excel support, Smart pattern detection'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

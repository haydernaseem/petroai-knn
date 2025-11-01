import os
import base64
import io
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

# خدمة الملفات الثابتة


@app.route('/')
def serve_index():
    return send_from_directory('.', 'knn.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


class ContourMap:
    def __init__(self, X, Y, Z, well_names=None):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.well_names = well_names

    def knn_interpolation(self, target_x, target_y, n_neighbors=5):
        """KNN interpolation for a single point"""
        distances = np.sqrt((self.X - target_x)**2 + (self.Y - target_y)**2)
        nearest_indices = np.argsort(distances)[:n_neighbors]
        weights = 1 / (distances[nearest_indices] + 1e-8)
        interpolated_value = np.sum(
            weights * self.Z.iloc[nearest_indices]) / np.sum(weights)
        return interpolated_value

    def plot_knn_map(self, target_x=None, target_y=None, n_neighbors=5, title="KNN Interpolation"):
        """Create contour map with optional target point"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create grid for contour plot
        # قلل النقاط لتسريع العملية
        xi = np.linspace(self.X.min(), self.X.max(), 50)
        yi = np.linspace(self.Y.min(), self.Y.max(), 50)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate for grid points
        zi = np.zeros(xi.shape)
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                zi[i, j] = self.knn_interpolation(
                    xi[i, j], yi[i, j], n_neighbors)

        # Create contour plot
        contour = ax.contourf(xi, yi, zi, levels=15, alpha=0.7, cmap='viridis')
        ax.contour(xi, yi, zi, levels=15, linewidths=0.5,
                   colors='black', alpha=0.5)

        # Plot original data points
        scatter = ax.scatter(self.X, self.Y, c=self.Z, s=50,
                             edgecolors='black', cmap='viridis')

        # Plot target point if provided
        if target_x is not None and target_y is not None:
            target_z = self.knn_interpolation(target_x, target_y, n_neighbors)
            ax.scatter([target_x], [target_y], c='red', s=150,
                       marker='*', label=f'Target (Z={target_z:.2f})')
            ax.legend()

        # Add well names if available
        if self.well_names is not None:
            for i, name in enumerate(self.well_names):
                ax.annotate(name, (self.X.iloc[i], self.Y.iloc[i]), xytext=(5, 5),
                            textcoords='offset points', fontsize=8)

        plt.colorbar(contour, label='Z Value')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(title)

        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)  # تأكد من إغلاق figure لحفظ الذاكرة

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

        # Read Excel file
        df = pd.read_excel(file)

        # Return column information
        columns = {
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'text_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'all_columns': df.columns.tolist(),
            'preview': df.head().to_dict('records'),
            'total_rows': len(df)
        }

        return jsonify(columns)

    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500


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
        well_col = request.form.get('well_column')
        target_x = float(request.form.get('target_x'))
        target_y = float(request.form.get('target_y'))
        n_neighbors = int(request.form.get('n_neighbors', 5))

        # قراءة البيانات
        df = pd.read_excel(file)

        # التحقق من الأعمدة
        for col in [x_col, y_col, z_col]:
            if col not in df.columns:
                return jsonify({'error': f'Column {col} not found in data'}), 400

        # استخراج البيانات
        X = df[x_col]
        Y = df[y_col]
        Z = df[z_col]
        well_names = df[well_col].tolist(
        ) if well_col and well_col in df.columns else None

        # إنشاء خريطة الكنتور
        contour_map = ContourMap(X, Y, Z, well_names)

        # تنفيذ الاستيفاء
        interpolated_z = contour_map.knn_interpolation(
            target_x, target_y, n_neighbors)

        # إنشاء الرسم
        plot_buffer = contour_map.plot_knn_map(
            target_x, target_y, n_neighbors,
            f"KNN Interpolation (k={n_neighbors})"
        )

        # تحويل الرسم إلى base64
        plot_base64 = base64.b64encode(plot_buffer.getvalue()).decode('utf-8')

        # الحصول على معلومات الجيران الأقرب
        distances = np.sqrt((X - target_x)**2 + (Y - target_y)**2)
        nearest_indices = np.argsort(distances)[:n_neighbors]
        nearest_neighbors = []

        for idx in nearest_indices:
            neighbor_info = {
                'x': float(X.iloc[idx]),
                'y': float(Y.iloc[idx]),
                'z': float(Z.iloc[idx]),
                'distance': float(distances.iloc[idx]),
                'well_name': well_names[idx] if well_names else f"Well_{idx+1}"
            }
            nearest_neighbors.append(neighbor_info)

        response = {
            'success': True,
            'interpolated_value': float(interpolated_z),
            'target_x': target_x,
            'target_y': target_y,
            'n_neighbors': n_neighbors,
            'nearest_neighbors': nearest_neighbors,
            'contour_plot': f"data:image/png;base64,{plot_base64}",
            'total_data_points': len(df)
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
    app.run(host='0.0.0.0', port=port, debug=False)

🪵 Nesting App v2
Aplicación de escritorio para optimizar el espacio de trabajo de cortes CNC a partir de archivos DXF.
Soporta:
- Capas DXF (importación y exportación)
- Rotación automática para mejor aprovechamiento
- Reporte automático en TXT con métricas de optimización

🚀 Instalación Rápida
1. Clonar o descargar este repositorio
git clone https://github.com/tu-usuario/nesting-app.git
cd nesting-app

2. Instalar dependencias
Asegúrate de tener Python 3.9+ instalado.
pip install -r requirements.txt

3. Requisitos del archivo requirements.txt
PySide6
ezdxf
shapely

▶️ Uso
1. Ejecutar la app:
python nesting_app_v2.py

2. Cargar un archivo DXF
Cada figura puede estar en una capa diferente.
Las capas se mantendrán en el DXF optimizado.

3. Optimizar
Ajusta automáticamente las posiciones para maximizar el uso del espacio.
Rota las figuras cuando sea necesario.
Mantiene la distancia mínima entre elementos.

4. Resultados
Se genera un DXF optimizado llamado optimized_layout.dxf
Se crea un reporte automático en la carpeta reports/ con métricas:
% de aprovechamiento
Tiempo de procesamiento
Figuras colocadas

📂 Estructura del proyecto
nesting-app/
│
├── nesting_app_v2.py        # Código principal
├── requirements.txt         # Dependencias
├── README.md                # Este archivo
└── reports/                 # Se crea automáticamente para guardar reportes

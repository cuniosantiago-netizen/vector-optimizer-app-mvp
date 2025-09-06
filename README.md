# Nesting App MVP

Este es un **MVP** para una aplicación de escritorio que permite:
- Importar elementos en formato **DXF**
- Acomodarlos automáticamente en un espacio de trabajo
- Visualizar métricas de aprovechamiento del espacio
- Exportar el resultado nuevamente en formato **DXF**

La app está pensada para optimizar cortes en CNC y probar algoritmos de nesting 2D.

---

## Probar la aplicación en la nube (sin instalar nada)

1. **Abre este repositorio en GitHub**  
2. Haz clic en el botón verde **< > Code**  
3. Ve a la pestaña **Codespaces**  
4. Haz clic en **Create codespace on main**  
5. Espera a que cargue el entorno (toma 1-2 min)

Cuando termine:
- Abre la pestaña **PORTS**  
- Busca el puerto **6080** con el nombre *Escritorio Remoto (GUI)*  
- Haz clic en el ícono 🌐 para abrir el escritorio virtual  

En el escritorio virtual:  
```bash
cd /workspaces/nesting-app
python3 nesting_app.py

"""
Aplicación principal de análisis de radiografías dentales panorámicas.
Interfaz web desarrollada con Streamlit para análisis asistido por IA.

VERSIÓN CORREGIDA - Canvas con modo transform y sincronización mejorada
"""

# ============================================================
# IMPORTS CON MANEJO DE ERRORES
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import json
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Optional

# Import de PIL con manejo de error (por si hay instalación corrupta)
try:
    from PIL import Image
except ImportError as e:
    st.error(f"Error al importar PIL/Pillow: {e}")
    st.info("Intente reinstalar Pillow: pip uninstall pillow && pip install pillow")
    st.stop()

# ============================================================
# PARCHE DE COMPATIBILIDAD CRÍTICO
# Debe ejecutarse ANTES de importar streamlit_drawable_canvas
# Soluciona error: "module 'streamlit.elements.image' has no attribute 'image_to_url'"
# ============================================================
try:
    import streamlit.elements.image as st_image
    if not hasattr(st_image, 'image_to_url'):
        def _image_to_url(
            image,
            width=-1,
            clamp=False,
            channels='RGB',
            output_format='auto',
            image_id='',
            allow_emoji=False
        ):
            """
            Función de compatibilidad que reemplaza image_to_url removida en Streamlit 1.28+.
            Convierte una imagen PIL a una URL base64 data URI.
            """
            try:
                if isinstance(image, Image.Image):
                    pil_img = image
                elif isinstance(image, np.ndarray):
                    if image.dtype != np.uint8:
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                    if len(image.shape) == 2:
                        pil_img = Image.fromarray(image, mode='L').convert('RGB')
                    elif image.shape[2] == 4:
                        pil_img = Image.fromarray(image, mode='RGBA')
                    else:
                        pil_img = Image.fromarray(image, mode='RGB')
                else:
                    return ""
                
                if pil_img.mode == 'RGBA':
                    fmt = 'PNG'
                    mime = 'png'
                else:
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    fmt = 'PNG'
                    mime = 'png'
                
                buffered = BytesIO()
                pil_img.save(buffered, format=fmt)
                img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                return f"data:image/{mime};base64,{img_b64}"
                
            except Exception as e:
                print(f"Error en image_to_url patch: {e}")
                return ""
        
        st_image.image_to_url = _image_to_url
        
except Exception as e:
    print(f"⚠️ No se pudo aplicar parche de compatibilidad: {e}")

# Ahora sí importar el canvas (después del parche)
from streamlit_drawable_canvas import st_canvas

# ============================================================
# FUNCIÓN DE COMPATIBILIDAD PARA RERUN
# ============================================================
def safe_rerun():
    """Ejecuta rerun de forma compatible con múltiples versiones de Streamlit."""
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()

# Importar módulos propios
from utils import (
    load_image,
    apply_clahe_filter,
    apply_edge_detection,
    create_bone_density_heatmap,
    apply_bone_crest_highlight,
    bgr_to_rgb,
    numpy_to_pil_rgba
)
from analysis import (
    ToothDetector,
    CalibrationSystem,
    calculate_tooth_measurements,
    calculate_bone_crest_line,
    classify_dentition,
    draw_detections,
    draw_bone_crest_line,
    add_manual_tooth,
    generar_plantilla_estandar
)


# Configuración de la página
st.set_page_config(
    page_title="DentalScan AI - Análisis de Radiografías",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
    }
    .help-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-left: 4px solid #28a745;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        color: #155724;
        line-height: 1.6;
    }
    .help-box strong {
        color: #0d5524;
    }
    </style>
""", unsafe_allow_html=True)


def prepare_image_for_canvas(
    image_rgb: np.ndarray, 
    max_width: int = 800
) -> Tuple[Image.Image, int, int]:
    """
    Prepara una imagen NumPy para usarla como fondo del canvas.
    HARD FIX para Streamlit Cloud:
    - Fuerza modo RGB (no RGBA)
    - Redimensiona a máximo 800px de ancho
    - Usa LANCZOS para mejor calidad
    """
    if not isinstance(image_rgb, np.ndarray):
        raise ValueError("La imagen debe ser un numpy array")
    
    # Asegurar que sea RGB (3 canales)
    if len(image_rgb.shape) == 2:
        # Escala de grises -> RGB
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    elif len(image_rgb.shape) == 3 and image_rgb.shape[2] == 4:
        # RGBA -> RGB (quitar canal alpha)
        image_rgb = image_rgb[:, :, :3]
    
    # Asegurar uint8
    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        else:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    
    # Crear imagen PIL en modo RGB
    pil_image = Image.fromarray(image_rgb, mode='RGB')
    
    # REDIMENSIONADO AGRESIVO: máximo 800px de ancho
    orig_width, orig_height = pil_image.size
    
    if orig_width > max_width:
        # Calcular nueva altura manteniendo proporción
        ratio = max_width / orig_width
        new_height = int(orig_height * ratio)
        new_width = max_width
        
        # Redimensionar con LANCZOS (mejor calidad)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    # FORZAR RGB - El canvas falla con RGBA o P en Linux/Cloud
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    # Obtener dimensiones EXACTAS de la imagen final
    final_width, final_height = pil_image.size
    
    return pil_image, final_width, final_height


def detections_to_canvas_objects(
    detections: List[Dict], 
    scale_x: float = 1.0, 
    scale_y: float = 1.0
) -> Optional[Dict]:
    """
    Convierte detecciones a formato Fabric.js para el canvas.
    Incluye rectángulos editables Y etiquetas con el número de diente.
    """
    if not detections:
        return None
    
    canvas_objects = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        width = (x2 - x1) * scale_x
        height = (y2 - y1) * scale_y
        
        # Color según origen
        if det.get('template', False):
            stroke_color = "#00FFFF"  # Cian para plantilla
            fill_color = "rgba(0, 255, 255, 0.2)"
            text_color = "#00FFFF"
        elif det.get('manual', False):
            stroke_color = "#FFA500"  # Naranja para manual
            fill_color = "rgba(255, 165, 0, 0.2)"
            text_color = "#FFA500"
        else:
            stroke_color = "#00FF00"  # Verde para IA
            fill_color = "rgba(0, 255, 0, 0.2)"
            text_color = "#00FF00"
        
        # Rectángulo del diente
        rect_obj = {
            "type": "rect",
            "left": float(x1_scaled),
            "top": float(y1_scaled),
            "width": float(width),
            "height": float(height),
            "fill": fill_color,
            "stroke": stroke_color,
            "strokeWidth": 2,
            "selectable": True,
            "evented": True,
            "hasControls": True,
            "hasBorders": True,
            "hasRotatingPoint": True,
            "lockRotation": False,
            "tooth_id": det['id']
        }
        canvas_objects.append(rect_obj)
        
        # Etiqueta con número de diente (texto encima del rectángulo)
        label_obj = {
            "type": "text",
            "left": float(x1_scaled + width/2 - 8),  # Centrado horizontal
            "top": float(y1_scaled - 18),  # Encima del rectángulo
            "text": str(det['id']),
            "fontSize": 14,
            "fontWeight": "bold",
            "fontFamily": "Arial",
            "fill": text_color,
            "stroke": "#000000",
            "strokeWidth": 0.5,
            "selectable": False,  # No seleccionable (solo informativo)
            "evented": False,
            "tooth_id": det['id'],
            "isLabel": True  # Marcador para identificar etiquetas
        }
        canvas_objects.append(label_obj)
    
    return {"version": "4.4.0", "objects": canvas_objects}


def landmarks_to_canvas_objects(
    landmarks: List[Dict],
    scale_x: float = 1.0,
    scale_y: float = 1.0
) -> List[Dict]:
    """Convierte reparos anatómicos a objetos de canvas (círculos)."""
    canvas_objects = []
    
    landmark_colors = {
        "Conducto Mentoniano": "#FF6B6B",
        "Seno Maxilar": "#4ECDC4",
        "Otro": "#FFE66D"
    }
    
    for lm in landmarks:
        x = lm['x'] * scale_x
        y = lm['y'] * scale_y
        name = lm.get('name', 'Otro')
        color = landmark_colors.get(name, landmark_colors['Otro'])
        
        canvas_obj = {
            "type": "circle",
            "left": float(x - 10),
            "top": float(y - 10),
            "radius": 10,
            "fill": color,
            "stroke": "#FFFFFF",
            "strokeWidth": 2,
            "selectable": True,
            "evented": True,
            "hasControls": True,
            "hasBorders": True,
            "landmark_name": name,
            "landmark_id": lm.get('id', 0)
        }
        canvas_objects.append(canvas_obj)
    
    return canvas_objects


def canvas_to_detections_and_landmarks(
    canvas_result,
    existing_detections: List[Dict],
    existing_landmarks: List[Dict],
    scale_x: float = 1.0,
    scale_y: float = 1.0
) -> Tuple[List[Dict], List[Dict]]:
    """Convierte objetos del canvas a detecciones y landmarks."""
    if canvas_result is None or canvas_result.json_data is None:
        return existing_detections or [], existing_landmarks or []
    
    try:
        canvas_data = canvas_result.json_data
        if isinstance(canvas_data, str):
            canvas_data = json.loads(canvas_data)
        objects = canvas_data.get("objects", [])
    except Exception as e:
        return existing_detections or [], existing_landmarks or []
    
    detections = []
    landmarks = []
    
    existing_det_dict = {det['id']: det for det in (existing_detections or [])}
    
    tooth_id_counter = max([det['id'] for det in (existing_detections or [])], default=0) + 1
    landmark_id_counter = max([lm['id'] for lm in (existing_landmarks or [])], default=0) + 1
    
    for obj in objects:
        obj_type = obj.get("type", "")
        
        # Ignorar etiquetas de texto (son solo visuales)
        if obj.get("isLabel", False) or obj_type == "text":
            continue
        
        if obj_type == "rect":
            # Manejar transformaciones (scaleX, scaleY)
            obj_scale_x = float(obj.get("scaleX", 1.0))
            obj_scale_y = float(obj.get("scaleY", 1.0))
            
            left = float(obj.get("left", 0)) / scale_x
            top = float(obj.get("top", 0)) / scale_y
            width = float(obj.get("width", 0)) * obj_scale_x / scale_x
            height = float(obj.get("height", 0)) * obj_scale_y / scale_y
            
            tooth_id = obj.get("tooth_id")
            if tooth_id is None:
                tooth_id = tooth_id_counter
                tooth_id_counter += 1
            
            existing_det = existing_det_dict.get(tooth_id, {})
            
            detections.append({
                'bbox': [int(left), int(top), int(left + width), int(top + height)],
                'confidence': existing_det.get('confidence', 0.5),
                'id': tooth_id,
                'manual': True,
                'template': existing_det.get('template', False)
            })
            
        elif obj_type == "circle":
            left = float(obj.get("left", 0))
            top = float(obj.get("top", 0))
            radius = float(obj.get("radius", 10))
            
            center_x = (left + radius) / scale_x
            center_y = (top + radius) / scale_y
            
            landmark_id = obj.get("landmark_id")
            landmark_name = obj.get("landmark_name", "Punto Anatómico")
            
            if landmark_id is None:
                landmark_id = landmark_id_counter
                landmark_id_counter += 1
            
            landmarks.append({
                'id': landmark_id,
                'x': int(center_x),
                'y': int(center_y),
                'name': landmark_name
            })
    
    return detections, landmarks


def initialize_session_state():
    """Inicializa las variables de sesión si no existen."""
    defaults = {
        'image': None,
        'image_rgb': None,
        'detector': None,
        'calibration': CalibrationSystem(),
        'detections': [],
        'landmarks': [],
        'canvas_key': 0,
        'drawing_tool': 'rect',        # rect o circle
        'mouse_mode': 'draw',          # NUEVO: 'draw' o 'transform'
        'landmark_type': 'Conducto Mentoniano'
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def render_results_table(
    detections: List[Dict],
    landmarks: List[Dict],
    calibration: CalibrationSystem
) -> None:
    """Renderiza la tabla de resultados."""
    st.subheader("📋 Tabla de Resultados")
    
    tab_teeth, tab_landmarks = st.tabs(["🦷 Dientes", "📍 Reparos Anatómicos"])
    
    with tab_teeth:
        if detections:
            if calibration.calibrated:
                measurements_data = []
                for det in detections:
                    measures = calculate_tooth_measurements(det['bbox'], calibration)
                    if det.get('template'):
                        origin = "📋 Plantilla"
                    elif det.get('manual'):
                        origin = "✏️ Manual"
                    else:
                        origin = "🤖 IA"
                    measurements_data.append({
                        'ID': det['id'],
                        'Origen': origin,
                        'Ancho (mm)': measures['width_mm'],
                        'Alto (mm)': measures['height_total_mm'],
                        'Corona (mm)': measures['crown_height_mm'],
                        'Raíz (mm)': measures['root_height_mm'],
                        'Confianza': f"{det['confidence']:.2f}"
                    })
                
                df_teeth = pd.DataFrame(measurements_data)
                st.dataframe(df_teeth, use_container_width=True, hide_index=True)
                
                csv = df_teeth.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("📥 Descargar CSV", data=csv, 
                                   file_name=f"dientes_{len(detections)}.csv", mime="text/csv")
            else:
                st.warning("⚠️ Calibre el sistema para ver mediciones en milímetros.")
                data = []
                for det in detections:
                    if det.get('template'):
                        origin = "📋 Plantilla"
                    elif det.get('manual'):
                        origin = "✏️ Manual"
                    else:
                        origin = "🤖 IA"
                    data.append({
                        'ID': det['id'],
                        'Origen': origin,
                        'X1': det['bbox'][0],
                        'Y1': det['bbox'][1],
                        'X2': det['bbox'][2],
                        'Y2': det['bbox'][3],
                        'Ancho (px)': det['bbox'][2] - det['bbox'][0],
                        'Alto (px)': det['bbox'][3] - det['bbox'][1]
                    })
                df_teeth = pd.DataFrame(data)
                st.dataframe(df_teeth, use_container_width=True, hide_index=True)
        else:
            st.info("No hay dientes. Use 'Generar Plantilla' o dibuje manualmente.")
    
    with tab_landmarks:
        if landmarks:
            lm_data = [{'ID': lm['id'], 'Reparo': lm['name'], 'X': lm['x'], 'Y': lm['y'], 
                        'Estado': '✅ Marcado'} for lm in landmarks]
            df_landmarks = pd.DataFrame(lm_data)
            st.dataframe(df_landmarks, use_container_width=True, hide_index=True)
        else:
            st.info("No hay reparos anatómicos marcados.")


def main():
    """Función principal de la aplicación."""
    
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🦷 DentalScan AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análisis Inteligente de Radiografías Panorámicas</p>', unsafe_allow_html=True)
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # ---- Carga de imagen ----
        st.subheader("📁 Cargar Radiografía")
        uploaded_file = st.file_uploader(
            "Seleccione una imagen",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            help="Formatos: JPG, PNG, PDF"
        )
        
        if uploaded_file is not None:
            try:
                # Cargar imagen con optimización para Streamlit Cloud
                image_bgr = load_image(uploaded_file, max_dimension=1200)
                
                # Validar que la imagen se cargó correctamente
                if image_bgr is None or image_bgr.size == 0:
                    raise ValueError("La imagen cargada está vacía")
                
                # Guardar en session_state
                st.session_state.image = image_bgr.copy()  # Copia para evitar referencias
                st.session_state.image_rgb = bgr_to_rgb(image_bgr)
                
                # Mostrar info de la imagen
                h, w = image_bgr.shape[:2]
                st.success(f"✅ {uploaded_file.name} ({w}x{h}px)")
                
                # Inicializar detector si no existe
                if st.session_state.detector is None:
                    st.session_state.detector = ToothDetector(None)
                    
            except Exception as e:
                st.error(f"❌ Error al cargar imagen: {str(e)}")
                st.info("💡 Intente con otro archivo o formato (JPG, PNG, PDF)")
        
        st.divider()
        
        # ============================================================
        # NUEVO: ACCIÓN DEL MOUSE (Dibujar vs Mover/Editar)
        # ============================================================
        st.subheader("🕹️ Acción del Mouse")
        mouse_mode = st.radio(
            "Seleccione modo:",
            options=["✏️ Dibujar", "🔄 Mover/Editar"],
            index=0 if st.session_state.mouse_mode == 'draw' else 1,
            help="'Dibujar' crea nuevos objetos. 'Mover/Editar' permite seleccionar, mover y redimensionar."
        )
        
        if "Dibujar" in mouse_mode:
            st.session_state.mouse_mode = 'draw'
        else:
            st.session_state.mouse_mode = 'transform'
        
        # ---- Herramienta de dibujo (solo visible en modo Dibujar) ----
        if st.session_state.mouse_mode == 'draw':
            st.subheader("🎯 Herramienta de Dibujo")
            tool_option = st.radio(
                "Tipo de objeto:",
                options=["🦷 Dientes (Rectángulos)", "📍 Puntos Anatómicos (Círculos)"],
                index=0
            )
            
            if "Rectángulos" in tool_option:
                st.session_state.drawing_tool = "rect"
            else:
                st.session_state.drawing_tool = "circle"
                st.session_state.landmark_type = st.selectbox(
                    "Tipo de reparo:",
                    options=["Conducto Mentoniano", "Seno Maxilar", "Otro"],
                    index=0
                )
        
        st.divider()
        
        # ---- Filtros ----
        st.subheader("🔍 Filtros")
        filter_option = st.selectbox(
            "Filtro de imagen",
            ["Original", "CLAHE (Contraste)", "Detección de Bordes", "Mapa de Densidad Ósea"]
        )
        
        highlight_crest = st.checkbox("🦴 Resaltar Cresta Ósea", value=False)
        crest_intensity = 0.0
        if highlight_crest:
            crest_intensity = st.slider("Intensidad", 0.1, 1.0, 0.5, 0.1)
        
        st.divider()
        
        # ---- Acciones ----
        st.subheader("⚡ Acciones Rápidas")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Plantilla 28", use_container_width=True, help="Generar 28 dientes (dentadura permanente completa)"):
                if st.session_state.image is not None:
                    template = generar_plantilla_estandar(st.session_state.image)
                    st.session_state.detections = template
                    st.session_state.canvas_key += 1  # FORZAR RE-RENDER
                    st.success(f"✅ {len(template)} dientes generados")
                    safe_rerun()
                else:
                    st.warning("⚠️ Cargue imagen primero")
        
        with col2:
            if st.button("🤖 Detectar IA", use_container_width=True, help="Detectar con IA"):
                if st.session_state.image is not None and st.session_state.detector:
                    with st.spinner("Detectando..."):
                        st.session_state.detections = st.session_state.detector.detect(st.session_state.image)
                        st.session_state.canvas_key += 1  # FORZAR RE-RENDER
                        if st.session_state.detections:
                            st.success(f"✅ {len(st.session_state.detections)} dientes")
                        else:
                            st.warning("No detectados")
                        safe_rerun()
        
        if st.button("🗑️ Limpiar Todo", use_container_width=True):
            st.session_state.detections = []
            st.session_state.landmarks = []
            st.session_state.canvas_key += 1
            safe_rerun()
        
        st.divider()
        
        # ---- Calibración ----
        st.subheader("📏 Calibración")
        if st.session_state.calibration.calibrated:
            st.success(f"✅ {st.session_state.calibration.pixel_ratio:.4f} mm/px")
            if st.button("🔄 Reset Calibración"):
                st.session_state.calibration.reset()
                safe_rerun()
        else:
            st.warning("⚠️ No calibrado")
    
    # ==================== ÁREA PRINCIPAL ====================
    if st.session_state.image is None:
        st.info("👆 Cargue una radiografía desde el panel lateral para comenzar.")
        
        st.markdown("""
        ### 📋 Instrucciones:
        1. **Cargar Imagen**: Panel lateral (JPG, PNG, PDF)
        2. **Modo Mouse**: "Dibujar" para crear, "Mover/Editar" para ajustar
        3. **Generar**: Use plantilla o detección IA
        4. **Editar**: En modo "Mover/Editar", arrastre y redimensione los cuadros
        5. **Guardar**: Presione "Guardar Cambios" para aplicar ediciones
        """)
        return
    
    # ---- Procesar imagen según filtro ----
    display_image = st.session_state.image.copy()
    
    if filter_option == "CLAHE (Contraste)":
        display_image = apply_clahe_filter(display_image)
    elif filter_option == "Detección de Bordes":
        display_image = apply_edge_detection(display_image)
    elif filter_option == "Mapa de Densidad Ósea":
        display_image = create_bone_density_heatmap(display_image)
    
    if highlight_crest and crest_intensity > 0:
        display_image = apply_bone_crest_highlight(display_image, crest_intensity)
    
    display_image_rgb = bgr_to_rgb(display_image)
    
    # ==================== CANVAS INTERACTIVO ====================
    st.subheader("🎨 Canvas Interactivo")
    
    # ============================================================
    # INFORMACIÓN DE MODO ACTUAL
    # ============================================================
    if st.session_state.mouse_mode == 'transform':
        st.success("""
        **🔄 Modo Mover/Editar activo** (los números indican el ID de cada diente):  
        ✓ **Click** en un rectángulo para seleccionarlo (aparecen controles)  
        ✓ **Arrastrar** el centro para mover  
        ✓ **Arrastrar esquinas** para redimensionar  
        ✓ **Arrastrar círculo superior** para rotar 🔄  
        ✓ **Presionar Delete o Backspace** para ELIMINAR el seleccionado  
        ✓ **Guardar Cambios** para confirmar eliminaciones
        """)
    else:
        if st.session_state.drawing_tool == "rect":
            st.info("✏️ **Modo Dibujar Dientes**: Click y arrastre para crear rectángulos.")
        else:
            st.info(f"✏️ **Modo Dibujar Reparos**: Click para marcar '{st.session_state.landmark_type}'.")
    
    # ============================================================
    # HARD FIX PARA STREAMLIT CLOUD
    # ============================================================
    try:
        # Validar que la imagen existe y no está vacía
        if display_image_rgb is None or display_image_rgb.size == 0:
            raise ValueError("La imagen a mostrar está vacía")
        
        # Preparar imagen para canvas (máximo 800px ancho, modo RGB forzado)
        pil_background, canvas_width, canvas_height = prepare_image_for_canvas(
            display_image_rgb, 
            max_width=800
        )
        
        # VERIFICACIÓN: imagen PIL válida
        if pil_background is None:
            raise ValueError("No se pudo crear la imagen PIL")
        
        if pil_background.size[0] == 0 or pil_background.size[1] == 0:
            raise ValueError(f"Dimensiones inválidas: {pil_background.size}")
        
        # FORZAR RGB - Crítico para Linux/Cloud
        if pil_background.mode != "RGB":
            pil_background = pil_background.convert("RGB")
        
        # Las dimensiones del canvas DEBEN coincidir EXACTAMENTE con la imagen
        exact_width, exact_height = pil_background.size
        
        # ============================================================
        # DEBUG VISUAL - Ver si la imagen se carga correctamente
        # Si ves esta imagen pero no el canvas, el problema es el canvas
        # Si no ves esta imagen, el problema es la carga
        # ============================================================
        with st.expander("🔍 DEBUG: Vista previa de imagen (expandir para verificar)", expanded=False):
            st.image(pil_background, caption=f"Imagen cargada: {exact_width}x{exact_height}px, Modo: {pil_background.mode}", use_container_width=True)
            st.caption("✅ Si ves la imagen aquí, la carga funciona correctamente.")
        
        # Calcular escalas
        original_height, original_width = st.session_state.image.shape[:2]
        scale_x = exact_width / original_width
        scale_y = exact_height / original_height
        
        # ============================================================
        # PREPARAR OBJETOS INICIALES (detecciones + landmarks)
        # ============================================================
        initial_drawing = None
        canvas_objects = []
        
        if st.session_state.detections:
            det_objects = detections_to_canvas_objects(
                st.session_state.detections,
                scale_x, scale_y
            )
            if det_objects:
                canvas_objects.extend(det_objects.get("objects", []))
        
        if st.session_state.landmarks:
            lm_objects = landmarks_to_canvas_objects(
                st.session_state.landmarks,
                scale_x, scale_y
            )
            canvas_objects.extend(lm_objects)
        
        if canvas_objects:
            initial_drawing = {"version": "4.4.0", "objects": canvas_objects}
        
        # ============================================================
        # DETERMINAR DRAWING_MODE
        # ============================================================
        if st.session_state.mouse_mode == 'transform':
            drawing_mode = "transform"
        else:
            drawing_mode = st.session_state.drawing_tool
        
        # Colores según herramienta
        if st.session_state.drawing_tool == "rect":
            fill_color = "rgba(0, 255, 0, 0.15)"
            stroke_color = "#00FF00"
        else:
            landmark_colors = {
                "Conducto Mentoniano": "#FF6B6B",
                "Seno Maxilar": "#4ECDC4",
                "Otro": "#FFE66D"
            }
            stroke_color = landmark_colors.get(st.session_state.landmark_type, "#FFE66D")
            fill_color = stroke_color
        
        # ============================================================
        # CANVAS - Dimensiones EXACTAS de la imagen PIL
        # ============================================================
        canvas_result = st_canvas(
            fill_color=fill_color,
            stroke_width=2,
            stroke_color=stroke_color,
            background_image=pil_background,  # Imagen RGB directa
            update_streamlit=True,
            height=exact_height,  # EXACTO de la imagen
            width=exact_width,    # EXACTO de la imagen
            drawing_mode=drawing_mode,
            point_display_radius=10 if st.session_state.drawing_tool == "circle" else 0,
            key=f"canvas_{st.session_state.canvas_key}",
            initial_drawing=initial_drawing,
            display_toolbar=True
        )
        
        # Info del canvas
        st.caption(f"📐 {exact_width}x{exact_height}px | Modo: {drawing_mode.upper()} | Objetos: {len(canvas_objects)}")
        
        # ============================================================
        # BOTÓN GUARDAR CAMBIOS
        # ============================================================
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Guardar Cambios del Canvas", use_container_width=True, type="primary"):
                if canvas_result is not None and canvas_result.json_data:
                    new_detections, new_landmarks = canvas_to_detections_and_landmarks(
                        canvas_result,
                        st.session_state.detections,
                        st.session_state.landmarks,
                        scale_x, scale_y
                    )
                    st.session_state.detections = new_detections
                    st.session_state.landmarks = new_landmarks
                    st.session_state.canvas_key += 1  # Actualizar key para sincronizar
                    st.success(f"✅ Guardado: {len(new_detections)} dientes, {len(new_landmarks)} reparos")
                    safe_rerun()
                else:
                    st.warning("No hay cambios para guardar.")
        
        with col2:
            num_det = len(st.session_state.detections)
            num_lm = len(st.session_state.landmarks)
            st.info(f"📊 Estado actual: {num_det} dientes, {num_lm} reparos")
        
    except Exception as e:
        st.error(f"❌ Error al renderizar canvas: {e}")
        st.warning(f"**Detalles técnicos:** {type(e).__name__}: {str(e)}")
        st.info("💡 Mostrando imagen estática como alternativa. Si el problema persiste, verifique las dependencias.")
        
        # Mostrar imagen estática como fallback
        if display_image_rgb is not None and display_image_rgb.size > 0:
            st.image(display_image_rgb, use_container_width=True, caption="Vista estática (canvas no disponible)")
    
    # ==================== CALIBRACIÓN ====================
    if not st.session_state.calibration.calibrated and st.session_state.detections:
        st.divider()
        st.header("📏 Calibración del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tooth_options = {f"Diente #{det['id']}": det for det in st.session_state.detections}
            selected_label = st.selectbox("Seleccione diente de referencia:", list(tooth_options.keys()))
            selected_tooth = tooth_options[selected_label]
            bbox = selected_tooth['bbox']
            
            width_px = abs(bbox[2] - bbox[0])
            height_px = abs(bbox[3] - bbox[1])
            
            st.write(f"**Dimensiones:** {width_px}px x {height_px}px")
        
        with col2:
            cal_type = st.radio("Usar medida de:", ["Ancho", "Alto"], horizontal=True)
            
            if cal_type == "Ancho":
                ref_mm = st.number_input("Ancho real (mm)", 0.1, 50.0, 7.0, 0.1)
                ref_px = float(width_px)
            else:
                ref_mm = st.number_input("Alto real (mm)", 0.1, 50.0, 20.0, 0.1)
                ref_px = float(height_px)
            
            if st.button("✅ Calibrar Sistema", use_container_width=True):
                try:
                    st.session_state.calibration.calibrate(ref_px, ref_mm)
                    st.success("✅ Calibrado!")
                    safe_rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # ==================== RESULTADOS ====================
    st.divider()
    st.header("📊 Resultados del Análisis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dientes", len(st.session_state.detections))
    with col2:
        dentition = classify_dentition(len(st.session_state.detections))
        st.metric("Dentición", dentition)
    with col3:
        st.metric("Reparos", len(st.session_state.landmarks))
    with col4:
        status = "✅ Sí" if st.session_state.calibration.calibrated else "❌ No"
        st.metric("Calibrado", status)
    
    render_results_table(
        st.session_state.detections,
        st.session_state.landmarks,
        st.session_state.calibration
    )


if __name__ == "__main__":
    main()

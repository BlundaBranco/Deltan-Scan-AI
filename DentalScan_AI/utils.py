"""
Módulo de utilidades para procesamiento de imágenes y conversión de archivos.
Contiene funciones para convertir PDFs a imágenes y aplicar filtros de OpenCV.

VERSIÓN CORREGIDA - Incluye conversión robusta NumPy -> PIL para canvas
"""

import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO
from typing import Tuple


def pdf_to_image(pdf_file) -> np.ndarray:
    """
    Convierte la primera página de un PDF a una imagen numpy array.
    
    Args:
        pdf_file: Archivo PDF cargado (BytesIO o similar)
        
    Returns:
        np.ndarray: Imagen en formato BGR (OpenCV)
        
    Raises:
        Exception: Si el PDF no se puede leer o convertir
    """
    try:
        # Leer el PDF desde bytes
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Obtener la primera página
        if len(pdf_document) == 0:
            raise ValueError("El PDF no tiene páginas")
        
        page = pdf_document[0]
        
        # Convertir página a imagen (pixmap)
        # Usamos una resolución alta para mantener calidad
        mat = fitz.Matrix(2.0, 2.0)  # Escala 2x para mejor calidad
        pix = page.get_pixmap(matrix=mat)
        
        # Convertir a bytes PIL Image
        img_bytes = pix.tobytes("png")
        img_pil = Image.open(BytesIO(img_bytes))
        
        # Convertir PIL Image a numpy array (RGB)
        img_array = np.array(img_pil)
        
        # Convertir RGB a BGR (OpenCV usa BGR)
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            # Escala de grises
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        pdf_document.close()
        return img_bgr
        
    except Exception as e:
        raise Exception(f"Error al convertir PDF a imagen: {str(e)}")


def load_image(file) -> np.ndarray:
    """
    Carga una imagen desde un archivo (JPG, PNG) o convierte PDF.
    
    Args:
        file: Archivo cargado por Streamlit
        
    Returns:
        np.ndarray: Imagen en formato BGR (OpenCV), uint8, rango 0-255
    """
    # Leer el archivo como bytes
    file_bytes = file.read()
    
    # Detectar tipo de archivo
    file_extension = file.name.lower().split('.')[-1]
    
    if file_extension == 'pdf':
        # Convertir PDF usando BytesIO
        pdf_file = BytesIO(file_bytes)
        return pdf_to_image(pdf_file)
    else:
        # Convertir bytes a numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        # Decodificar imagen
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"No se pudo decodificar la imagen {file.name}")
        
        # Asegurar formato correcto
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        return img


def numpy_to_pil_rgba(
    image: np.ndarray, 
    is_bgr: bool = True
) -> Image.Image:
    """
    Convierte una imagen NumPy a PIL Image en modo RGBA.
    
    Esta función es CRÍTICA para el canvas de streamlit-drawable-canvas,
    que requiere imágenes PIL en modo RGBA para mostrar correctamente el fondo.
    
    Args:
        image: Imagen numpy array (BGR o RGB)
        is_bgr: Si True, la imagen está en formato BGR (OpenCV). Default True.
        
    Returns:
        PIL.Image en modo RGBA con canal alpha completo (255)
        
    Raises:
        ValueError: Si la imagen no es válida
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("La imagen debe ser un numpy array")
    
    # Clonar para no modificar original
    img = image.copy()
    
    # Asegurar uint8 en rango 0-255
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Manejar diferentes formatos de entrada
    if len(img.shape) == 2:
        # Escala de grises -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # Ya tiene canal alpha
        if is_bgr:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img_rgb = img  # Ya está en RGBA
        # Crear PIL directamente en RGBA
        return Image.fromarray(img_rgb, mode='RGBA')
    elif img.shape[2] == 3:
        # 3 canales (BGR o RGB)
        if is_bgr:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
    else:
        raise ValueError(f"Formato de imagen no soportado: shape={img.shape}")
    
    # Crear imagen PIL en RGB primero
    pil_rgb = Image.fromarray(img_rgb, mode='RGB')
    
    # Convertir a RGBA con alpha completo
    pil_rgba = pil_rgb.convert('RGBA')
    
    return pil_rgba


def apply_clahe_filter(image: np.ndarray) -> np.ndarray:
    """
    Aplica el filtro CLAHE (Contrast Limited Adaptive Histogram Equalization)
    para mejorar el contraste y resaltar detalles óseos.
    
    CLAHE es especialmente útil para radiografías porque:
    - Mejora el contraste local sin sobreexponer áreas
    - Resalta bordes y estructuras óseas
    - Facilita la visualización de pérdida ósea
    
    Args:
        image: Imagen en formato BGR
        
    Returns:
        np.ndarray: Imagen procesada con CLAHE
    """
    # Convertir a escala de grises si es color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Crear objeto CLAHE
    # clipLimit: límite de contraste (2.0-3.0 es bueno para radiografías)
    # tileGridSize: tamaño de la cuadrícula para adaptación local (8x8 es estándar)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    
    # Aplicar CLAHE
    clahe_image = clahe.apply(gray)
    
    # Convertir de vuelta a BGR para mantener consistencia
    clahe_bgr = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    
    return clahe_bgr


def apply_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Aplica detección de bordes usando Canny para resaltar estructuras óseas.
    
    Útil para visualizar:
    - Bordes de dientes
    - Cresta ósea alveolar
    - Estructuras anatómicas
    
    Args:
        image: Imagen en formato BGR
        
    Returns:
        np.ndarray: Imagen con bordes detectados superpuestos
    """
    # Convertir a escala de grises
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Aplicar desenfoque gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detección de bordes Canny
    # threshold1, threshold2: umbrales de histéresis (ajustados para radiografías)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Convertir bordes a BGR y superponer sobre imagen original
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combinar: bordes en verde sobre imagen original
    result = image.copy()
    # Donde hay bordes, poner color verde (0, 255, 0 en BGR)
    result[edges > 0] = [0, 255, 0]
    
    return result


def create_bone_density_heatmap(image: np.ndarray) -> np.ndarray:
    """
    Crea un mapa de calor (heatmap) que resalta áreas de densidad ósea.
    
    Usa la intensidad de píxeles como proxy de densidad:
    - Píxeles más claros = mayor densidad ósea
    - Píxeles más oscuros = menor densidad (posible pérdida ósea)
    
    Args:
        image: Imagen en formato BGR
        
    Returns:
        np.ndarray: Imagen con colormap aplicado (JET colormap)
    """
    # Convertir a escala de grises
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalizar a 0-255
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Aplicar colormap JET (azul = bajo, rojo = alto)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    # Mezclar con imagen original para mejor visualización
    blended = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    return blended


def apply_bone_crest_highlight(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Aplica un filtro agresivo de Laplacian/Sobel para resaltar la cresta ósea.
    
    Usa operadores Laplacian y Sobel para detectar bordes de manera más agresiva,
    creando un efecto tipo mapa de calor que facilita ver la pérdida ósea.
    
    Args:
        image: Imagen en formato BGR
        intensity: Intensidad del efecto (0.0 a 1.0). Por defecto 0.5
        
    Returns:
        np.ndarray: Imagen procesada con resaltado agresivo de cresta ósea
    """
    # Convertir a escala de grises
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Aplicar CLAHE primero para mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Aplicar desenfoque gaussiano para suavizar antes de detectar bordes
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # OPERADOR LAPLACIAN - Detecta cambios de intensidad (bordes)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)
    laplacian_abs = np.absolute(laplacian)
    laplacian_max = np.max(laplacian_abs)
    if laplacian_max > 0:
        laplacian_norm = np.uint8(255 * laplacian_abs / laplacian_max)
    else:
        laplacian_norm = np.zeros_like(gray, dtype=np.uint8)
    
    # OPERADOR SOBEL - Detecta gradientes en X e Y
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobel_x_abs = np.absolute(sobel_x)
    sobel_x_max = np.max(sobel_x_abs)
    if sobel_x_max > 0:
        sobel_x_norm = np.uint8(255 * sobel_x_abs / sobel_x_max)
    else:
        sobel_x_norm = np.zeros_like(gray, dtype=np.uint8)
    
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y_abs = np.absolute(sobel_y)
    sobel_y_max = np.max(sobel_y_abs)
    if sobel_y_max > 0:
        sobel_y_norm = np.uint8(255 * sobel_y_abs / sobel_y_max)
    else:
        sobel_y_norm = np.zeros_like(gray, dtype=np.uint8)
    
    # Combinar Sobel X e Y
    sobel_combined = cv2.addWeighted(sobel_x_norm, 0.5, sobel_y_norm, 0.5, 0)
    
    # Combinar Laplacian y Sobel
    edges_combined = cv2.addWeighted(laplacian_norm, 0.6, sobel_combined, 0.4, 0)
    
    # Aplicar umbral para resaltar solo bordes fuertes
    _, edges_thresh = cv2.threshold(edges_combined, 50, 255, cv2.THRESH_BINARY)
    
    # Crear mapa de calor
    heatmap = cv2.applyColorMap(edges_combined, cv2.COLORMAP_JET)
    
    # Convertir bordes a BGR
    edges_bgr = cv2.cvtColor(edges_thresh, cv2.COLOR_GRAY2BGR)
    edges_bgr[edges_thresh > 0] = [0, 255, 255]  # Amarillo
    
    # Mezclar
    result = cv2.addWeighted(image, 1.0 - (intensity * 0.7), heatmap, intensity * 0.5, 0)
    result = cv2.addWeighted(result, 1.0 - (intensity * 0.4), edges_bgr, intensity * 0.4, 0)
    
    return result


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convierte imagen de BGR (OpenCV) a RGB (Streamlit/PIL).
    
    Args:
        image: Imagen en formato BGR
        
    Returns:
        np.ndarray: Imagen en formato RGB
    """
    if len(image.shape) == 2:
        # Escala de grises, no necesita conversión
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image_for_canvas(
    image: np.ndarray, 
    max_size: int = 800
) -> Tuple[np.ndarray, float, float]:
    """
    Redimensiona una imagen para el canvas manteniendo proporción.
    
    Args:
        image: Imagen numpy array
        max_size: Tamaño máximo (ancho o alto)
        
    Returns:
        Tupla (imagen redimensionada, escala_x, escala_y)
    """
    height, width = image.shape[:2]
    
    if height <= max_size and width <= max_size:
        return image.copy(), 1.0, 1.0
    
    scale = min(max_size / height, max_size / width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    scale_x = new_width / width
    scale_y = new_height / height
    
    return resized, scale_x, scale_y

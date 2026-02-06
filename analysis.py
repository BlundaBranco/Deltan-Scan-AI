"""
Módulo de análisis y lógica de negocio.
Contiene la clase para detección de dientes con YOLOv8, calibración,
y cálculos de mediciones automáticas.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
from ultralytics import YOLO
import random


class ToothDetector:
    """
    Clase para detectar dientes en radiografías panorámicas usando YOLOv8.
    
    Si no encuentra el modelo entrenado, activa modo Mock que genera
    bounding boxes simulados para pruebas de interfaz.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_mock: bool = False):
        """
        Inicializa el detector de dientes.
        
        Args:
            model_path: Ruta al archivo .pt del modelo YOLOv8.
                       Si es None, intenta cargar yolov8n.pt (modelo genérico).
            use_mock: Si es True, fuerza el modo Mock. Por defecto False.
        """
        self.model_path = model_path
        self.model = None
        self.mock_mode = False
        
        # Si se fuerza modo Mock, activarlo directamente
        if use_mock:
            print("Modo Mock activado manualmente.")
            self.mock_mode = True
            return
        
        # Intentar cargar modelo personalizado si se especifica
        if model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                print(f"Modelo YOLOv8 cargado desde: {model_path}")
                return
            except Exception as e:
                print(f"Error al cargar modelo personalizado: {e}. Intentando modelo genérico...")
        
        # Intentar cargar modelo genérico yolov8n.pt (se descarga automáticamente)
        try:
            print("Cargando modelo YOLOv8 genérico (yolov8n.pt)...")
            self.model = YOLO('yolov8n.pt')  # Se descarga automáticamente si no existe
            print("✅ Modelo YOLOv8 genérico cargado correctamente")
        except Exception as e:
            print(f"Error al cargar modelo YOLOv8: {e}. Activando modo Mock como respaldo.")
            self.mock_mode = True
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detecta dientes en la imagen usando detección por contornos OpenCV.
        
        Los dientes son las partes más brillantes/densas en radiografías,
        por lo que detectamos contornos de áreas blancas brillantes.
        
        Args:
            image: Imagen en formato BGR (OpenCV)
            
        Returns:
            Lista de diccionarios con información de cada diente detectado:
            [
                {
                    'bbox': [x1, y1, x2, y2],  # Coordenadas del bounding box
                    'confidence': float,        # Confianza de la detección
                    'id': int                  # ID único del diente
                },
                ...
            ]
        """
        if self.mock_mode:
            return self._mock_detection(image)
        else:
            # Usar detección por contornos (más confiable para radiografías)
            return self._contour_detection(image)
    
    def _contour_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Detección de dientes usando contornos OpenCV.
        
        Los dientes son las partes más brillantes/densas en radiografías.
        Detecta contornos de áreas blancas brillantes y filtra por tamaño,
        ubicación (zona central de arcadas) y forma.
        
        Args:
            image: Imagen en formato BGR
            
        Returns:
            Lista de detecciones con bounding boxes
        """
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        height, width = image.shape[:2]
        
        # Aplicar CLAHE para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Aplicar desenfoque gaussiano para reducir ruido
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Calcular umbral más selectivo - usar percentil alto para detectar solo áreas muy brillantes
        # Los dientes son las estructuras más radiopacas (más blancas)
        intensity_percentile_90 = np.percentile(blurred, 90)  # Percentil 90 = muy brillante
        intensity_percentile_85 = np.percentile(blurred, 85)  # Percentil 85 como respaldo
        
        # Usar umbral alto para detectar solo dientes (muy brillantes)
        _, thresh_high = cv2.threshold(blurred, int(intensity_percentile_90), 255, cv2.THRESH_BINARY)
        
        # También usar Otsu pero invertido para áreas brillantes
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combinar umbrales altos
        combined = cv2.bitwise_or(thresh_high, thresh_otsu)
        
        # Operaciones morfológicas más agresivas para aislar dientes individuales
        # Usar kernel más pequeño para preservar forma de dientes
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((3, 3), np.uint8)
        
        # Cerrar pequeños agujeros dentro de dientes
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        # Abrir para separar dientes adyacentes
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_medium, iterations=2)
        
        # ROI RESTRINGIDA: Solo buscar en el rectángulo central
        # Ignorar: 15% izquierdo, 15% derecho, 10% superior, 10% inferior
        roi_left = int(width * 0.15)
        roi_right = int(width * 0.85)
        roi_top = int(height * 0.10)
        roi_bottom = int(height * 0.90)
        
        # Crear máscara para ROI
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[roi_top:roi_bottom, roi_left:roi_right] = 255
        
        # Aplicar máscara a la imagen procesada
        opened_masked = cv2.bitwise_and(opened, mask)
        
        # Encontrar contornos solo en la ROI
        contours, _ = cv2.findContours(opened_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        # Filtrar contornos por tamaño y forma - más restrictivo para dientes
        # Dientes típicos: área entre 0.1% y 2% de la imagen
        min_area = (width * height) * 0.001   # 0.1% del área (dientes pequeños pero reales)
        max_area = (width * height) * 0.02    # 2% del área (dientes grandes, evitar estructuras óseas)
        
        # Filtrar por tamaño mínimo (evitar ruido muy pequeño)
        min_width = width * 0.015   # 1.5% del ancho (dientes más grandes)
        min_height = height * 0.02  # 2% del alto (dientes más altos)
        max_width = width * 0.15    # Máximo 15% del ancho (evitar ramas mandibulares)
        max_height = height * 0.20   # Máximo 20% del alto (evitar estructuras grandes)
        
        tooth_id = 1
        
        for contour in contours:
            # Calcular área del contorno
            area = cv2.contourArea(contour)
            
            # Filtrar por área
            if min_area <= area <= max_area:
                # Obtener bounding box del contorno
                x, y, w, h = cv2.boundingRect(contour)
                
                # FILTRO DE VERTICALIDAD: Un diente humano es más alto que ancho
                if h <= w:
                    continue  # Descartar si width >= height (no es un diente)
                
                # Filtrar por tamaño (evitar estructuras muy grandes o muy pequeñas)
                if (min_width <= w <= max_width and 
                    min_height <= h <= max_height):
                    
                    # Verificar que el contorno esté dentro de la ROI
                    if not (roi_left <= x and x + w <= roi_right and
                            roi_top <= y and y + h <= roi_bottom):
                        continue  # Fuera de la ROI, saltar
                    
                    # Calcular aspect ratio
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Filtrar por forma: dientes tienen aspect ratio más específico
                    # Dientes típicos: más altos que anchos (0.3 a 1.0, ya que h > w)
                    if 0.25 <= aspect_ratio <= 1.0:
                        # Calcular "confianza" basada en la intensidad promedio del área
                        roi = gray[y:y+h, x:x+w]
                        avg_intensity = np.mean(roi)
                        
                        # Calcular también la intensidad máxima (dientes tienen picos brillantes)
                        max_intensity = np.max(roi)
                        
                        # Confianza basada en intensidad promedio y máxima
                        # Dientes son muy brillantes, así que mayor intensidad = mayor confianza
                        confidence_avg = min(avg_intensity / 255.0, 0.95)
                        confidence_max = min(max_intensity / 255.0, 0.95)
                        confidence = (confidence_avg * 0.7 + confidence_max * 0.3)  # Ponderado
                        
                        # Solo incluir si es suficientemente brillante (dientes son muy radiopacos)
                        if confidence > 0.4:  # Al menos 40% de brillo (más selectivo para dientes)
                            # Verificar que no sea una estructura demasiado grande (rama mandibular)
                            # Los dientes tienen un área más compacta
                            compactness = area / (w * h)  # Ratio área/área del bbox
                            if compactness > 0.3:  # Dientes tienen buena compactness
                                detections.append({
                                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                                    'confidence': float(confidence),
                                    'id': tooth_id
                                })
                                tooth_id += 1
        
        # Si no se detectaron suficientes dientes, intentar con umbral ligeramente más bajo
        if len(detections) < 10:  # Si detectó menos de 10 dientes, intentar método alternativo
            # Usar percentil 85 como respaldo (menos restrictivo pero aún selectivo)
            _, thresh_backup = cv2.threshold(blurred, int(intensity_percentile_85), 255, cv2.THRESH_BINARY)
            
            # Aplicar operaciones morfológicas
            kernel_backup = np.ones((3, 3), np.uint8)
            thresh_backup = cv2.morphologyEx(thresh_backup, cv2.MORPH_CLOSE, kernel_backup, iterations=1)
            thresh_backup = cv2.morphologyEx(thresh_backup, cv2.MORPH_OPEN, kernel_backup, iterations=2)
            
            contours_backup, _ = cv2.findContours(thresh_backup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Usar set para evitar duplicados
            existing_boxes = set()
            for det in detections:
                existing_boxes.add(tuple(det['bbox']))
            
            for contour in contours_backup:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Verificar que no sea duplicado
                    bbox_tuple = (int(x), int(y), int(x + w), int(y + h))
                    if bbox_tuple in existing_boxes:
                        continue
                    
                    # FILTRO DE VERTICALIDAD: Un diente humano es más alto que ancho
                    if h <= w:
                        continue  # Descartar si width >= height
                    
                    if (min_width <= w <= max_width and 
                        min_height <= h <= max_height):
                        
                        # Verificar que el contorno esté dentro de la ROI
                        if not (roi_left <= x and x + w <= roi_right and
                                roi_top <= y and y + h <= roi_bottom):
                            continue
                        
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.25 <= aspect_ratio <= 1.0:
                            roi = gray[y:y+h, x:x+w]
                            avg_intensity = np.mean(roi)
                            max_intensity = np.max(roi)
                            confidence_avg = min(avg_intensity / 255.0, 0.95)
                            confidence_max = min(max_intensity / 255.0, 0.95)
                            confidence = (confidence_avg * 0.7 + confidence_max * 0.3)
                            
                            # Umbral ligeramente más bajo para el segundo intento
                            if confidence > 0.35:
                                compactness = area / (w * h)
                                if compactness > 0.3:
                                    detections.append({
                                        'bbox': [int(x), int(y), int(x + w), int(y + h)],
                                        'confidence': float(confidence),
                                        'id': tooth_id
                                    })
                                    existing_boxes.add(bbox_tuple)
                                    tooth_id += 1
        
        return detections
    
    def _mock_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Genera detecciones simuladas para modo Demo.
        
        Crea bounding boxes aleatorios distribuidos horizontalmente
        simulando una arcada dental panorámica.
        """
        height, width = image.shape[:2]
        detections = []
        
        # Simular entre 20-32 dientes (rango típico)
        num_teeth = random.randint(20, 32)
        
        # Distribuir dientes horizontalmente (simulando arcada)
        # Los dientes están en la mitad inferior de la imagen
        y_center = int(height * 0.65)  # Centro vertical aproximado
        tooth_height = random.randint(int(height * 0.08), int(height * 0.15))
        tooth_width = random.randint(int(width * 0.02), int(width * 0.04))
        
        # Espaciar dientes horizontalmente
        spacing = width // (num_teeth + 1)
        
        for i in range(num_teeth):
            x_center = spacing * (i + 1) + random.randint(-spacing//4, spacing//4)
            x1 = max(0, x_center - tooth_width // 2)
            x2 = min(width, x_center + tooth_width // 2)
            y1 = max(0, y_center - tooth_height // 2)
            y2 = min(height, y_center + tooth_height // 2)
            
            # Simular confianza aleatoria
            confidence = random.uniform(0.6, 0.95)
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'id': i + 1
            })
        
        return detections


def generar_plantilla_estandar(image: np.ndarray) -> List[Dict]:
    """
    Genera una plantilla estándar de 28 dientes (14 arriba, 14 abajo)
    para una dentadura permanente completa, distribuidos horizontalmente.
    
    Esta función se usa cuando la detección automática falla o detecta
    muy pocos dientes, proporcionando una base para que el usuario ajuste.
    
    Args:
        image: Imagen en formato BGR o escala de grises
        
    Returns:
        Lista de 28 detecciones simuladas (plantilla estándar dentadura permanente)
    """
    height, width = image.shape[:2]
    detections = []
    
    # Tamaño de diente ajustado para 14 por arcada
    tooth_width = int(width * 0.035)   # 3.5% del ancho (más pequeño para caber 14)
    tooth_height = int(height * 0.12)  # 12% del alto
    
    # Separación entre dientes (más pequeña para 14 dientes)
    spacing = int(width * 0.025)  # 2.5% del ancho entre dientes
    
    # Calcular posición inicial (centrar 14 dientes)
    total_width = 14 * tooth_width + 13 * spacing
    start_x = (width - total_width) // 2
    
    # Márgenes de seguridad
    margin_left = int(width * 0.05)
    margin_right = width - int(width * 0.05)
    
    # Ajustar start_x si se sale de los márgenes
    if start_x < margin_left:
        start_x = margin_left
        # Recalcular spacing para que quepan
        available_width = margin_right - margin_left - (14 * tooth_width)
        spacing = max(2, available_width // 13)
    
    # Arcada superior (14 dientes) - Posición Y más arriba
    upper_y = int(height * 0.30)  # 30% desde arriba
    for i in range(14):
        x = start_x + i * (tooth_width + spacing)
        y = upper_y
        
        # Asegurar que no se salga de la imagen
        if x + tooth_width > width:
            x = width - tooth_width - 5
        
        detections.append({
            'bbox': [int(x), int(y), int(x + tooth_width), int(y + tooth_height)],
            'confidence': 0.5,
            'id': i + 1,
            'manual': True,
            'template': True
        })
    
    # Arcada inferior (14 dientes) - Posición Y más abajo
    lower_y = int(height * 0.55)  # 55% desde arriba
    for i in range(14):
        x = start_x + i * (tooth_width + spacing)
        y = lower_y
        
        # Asegurar que no se salga de la imagen
        if x + tooth_width > width:
            x = width - tooth_width - 5
        
        detections.append({
            'bbox': [int(x), int(y), int(x + tooth_width), int(y + tooth_height)],
            'confidence': 0.5,
            'id': i + 15,  # Continuar numeración desde 15
            'manual': True,
            'template': True
        })
    
    return detections


def add_manual_tooth(
    x: int, 
    y: int, 
    width: int, 
    height: int, 
    existing_detections: List[Dict]
) -> Dict:
    """
    Agrega un diente manualmente usando coordenadas proporcionadas por el usuario.
    
    Args:
        x: Coordenada X del centro o esquina superior izquierda
        y: Coordenada Y del centro o esquina superior izquierda
        width: Ancho del diente en píxeles
        height: Alto del diente en píxeles
        existing_detections: Lista de detecciones existentes para generar ID único
        
    Returns:
        Diccionario con la detección del diente agregado manualmente
    """
    # Generar ID único (más alto que los existentes)
    max_id = max([det['id'] for det in existing_detections], default=0)
    new_id = max_id + 1
    
    # Calcular bounding box (asumiendo x, y son esquina superior izquierda)
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    
    return {
        'bbox': [int(x1), int(y1), int(x2), int(y2)],
        'confidence': 1.0,  # Confianza máxima para dientes manuales
        'id': new_id,
        'manual': True  # Marca para identificar dientes agregados manualmente
    }


class CalibrationSystem:
    """
    Sistema de calibración para convertir píxeles a milímetros.
    
    El usuario selecciona un diente o línea de referencia y proporciona
    su tamaño real en mm. El sistema calcula el ratio píxel/mm.
    """
    
    def __init__(self):
        """Inicializa el sistema de calibración."""
        self.pixel_ratio = None  # mm por píxel
        self.reference_length_px = None  # Longitud en píxeles
        self.reference_length_mm = None  # Longitud real en mm
        self.calibrated = False
    
    def calibrate(self, reference_length_px: float, reference_length_mm: float):
        """
        Calibra el sistema con una medida de referencia.
        
        Args:
            reference_length_px: Longitud de referencia en píxeles
            reference_length_mm: Longitud real en milímetros
        """
        if reference_length_px <= 0 or reference_length_mm <= 0:
            raise ValueError("Las medidas deben ser mayores que cero")
        
        self.reference_length_px = reference_length_px
        self.reference_length_mm = reference_length_mm
        self.pixel_ratio = reference_length_mm / reference_length_px
        self.calibrated = True
    
    def pixels_to_mm(self, pixels: float) -> float:
        """
        Convierte píxeles a milímetros.
        
        Args:
            pixels: Medida en píxeles
            
        Returns:
            Medida en milímetros
            
        Raises:
            ValueError: Si el sistema no está calibrado
        """
        if not self.calibrated:
            raise ValueError("El sistema no está calibrado. Calibre primero.")
        
        return pixels * self.pixel_ratio
    
    def reset(self):
        """Resetea la calibración."""
        self.pixel_ratio = None
        self.reference_length_px = None
        self.reference_length_mm = None
        self.calibrated = False


def calculate_tooth_measurements(
    bbox: List[int],
    calibration: CalibrationSystem
) -> Dict[str, float]:
    """
    Calcula las mediciones de un diente en milímetros.
    
    Usa heurística: 35% superior = Corona, 65% inferior = Raíz.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        calibration: Sistema de calibración
        
    Returns:
        Diccionario con medidas:
        {
            'width_mm': ancho total,
            'height_total_mm': altura total,
            'crown_height_mm': altura de corona (35% superior),
            'root_height_mm': altura de raíz (65% inferior)
        }
    """
    x1, y1, x2, y2 = bbox
    
    # Calcular dimensiones en píxeles
    width_px = abs(x2 - x1)
    height_px = abs(y2 - y1)
    
    # Convertir a milímetros
    width_mm = calibration.pixels_to_mm(width_px)
    height_total_mm = calibration.pixels_to_mm(height_px)
    
    # Heurística: 35% superior = Corona, 65% inferior = Raíz
    crown_height_px = height_px * 0.35
    root_height_px = height_px * 0.65
    
    crown_height_mm = calibration.pixels_to_mm(crown_height_px)
    root_height_mm = calibration.pixels_to_mm(root_height_px)
    
    return {
        'width_mm': round(width_mm, 2),
        'height_total_mm': round(height_total_mm, 2),
        'crown_height_mm': round(crown_height_mm, 2),
        'root_height_mm': round(root_height_mm, 2)
    }


def calculate_bone_crest_line(detections: List[Dict], image_height: int) -> int:
    """
    Calcula la posición sugerida de la cresta ósea alveolar.
    
    Usa la posición media de la BASE de las coronas de los dientes
    (parte inferior de los bounding boxes, ajustada al 35% superior)
    como referencia, ya que la cresta ósea está donde terminan las coronas.
    
    Args:
        detections: Lista de detecciones de dientes
        image_height: Altura de la imagen en píxeles
        
    Returns:
        Coordenada Y sugerida para la línea de cresta ósea
    """
    if not detections:
        # Si no hay detecciones, usar posición media de la imagen
        return image_height // 2
    
    # Obtener las coordenadas Y de la base de las coronas
    # La corona ocupa el 35% superior del diente, así que la base de la corona
    # está en y1 + (y2 - y1) * 0.35
    y_positions = []
    for det in detections:
        y1, y2 = det['bbox'][1], det['bbox'][3]
        # Base de la corona = parte inferior del 35% superior
        crown_base_y = y1 + int((y2 - y1) * 0.35)
        y_positions.append(crown_base_y)
    
    # Calcular posición media de la base de las coronas
    mean_y = int(np.mean(y_positions))
    
    return mean_y


def classify_dentition(num_teeth: int) -> str:
    """
    Clasifica el tipo de dentición basado en el número de dientes.
    
    Heurística mejorada:
    - <= 20 dientes: Dentición Primaria
    - 21-27 dientes: Dentición Mixta
    - 28-32 dientes: Dentición Permanente
    
    Args:
        num_teeth: Número de dientes detectados
        
    Returns:
        Clasificación de la dentición
    """
    if num_teeth <= 20:
        return "Dentición Primaria"
    elif 21 <= num_teeth <= 27:
        return "Dentición Mixta"
    elif 28 <= num_teeth <= 32:
        return "Dentición Permanente"
    else:
        # Caso atípico (más de 32 o menos de lo esperado)
        return f"Dentición Atípica ({num_teeth} dientes)"


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    calibration: Optional[CalibrationSystem] = None
) -> np.ndarray:
    """
    Dibuja los bounding boxes de los dientes detectados sobre la imagen.
    
    Diferencia visualmente entre dientes detectados por IA (verde) y 
    dientes agregados manualmente (azul/cian).
    
    Args:
        image: Imagen original
        detections: Lista de detecciones
        calibration: Sistema de calibración (opcional, para mostrar medidas)
        
    Returns:
        Imagen con bounding boxes dibujados
    """
    result_image = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        confidence = det['confidence']
        tooth_id = det['id']
        is_manual = det.get('manual', False)
        
        # Color: verde para IA, azul/cian para manual
        if is_manual:
            color = (255, 255, 0)  # Cian en BGR (para manual)
            prefix = "M"  # Marca para manual
        else:
            color = (0, 255, 0)  # Verde en BGR (para IA)
            prefix = ""
        
        # Dibujar rectángulo con grosor diferente
        thickness = 3 if is_manual else 2
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
        
        # Etiqueta con ID y confianza
        label = f"{prefix}#{tooth_id}"
        if not is_manual:
            label += f" ({confidence:.2f})"
        
        # Si está calibrado, mostrar medidas
        if calibration and calibration.calibrated:
            measures = calculate_tooth_measurements(det['bbox'], calibration)
            label += f" | W:{measures['width_mm']:.1f}mm"
        
        # Fondo para el texto (para mejor legibilidad)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            result_image,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Texto
        cv2.putText(
            result_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # Texto negro
            1
        )
    
    return result_image


def draw_bone_crest_line(image: np.ndarray, y_position: int) -> np.ndarray:
    """
    Dibuja una línea horizontal sugerida para la cresta ósea.
    
    Args:
        image: Imagen original
        y_position: Coordenada Y donde dibujar la línea
        
    Returns:
        Imagen con línea dibujada
    """
    result_image = image.copy()
    height, width = image.shape[:2]
    
    # Línea roja punteada para cresta ósea
    color = (0, 0, 255)  # Rojo en BGR
    thickness = 2
    
    # Dibujar línea punteada (simulada con pequeños segmentos)
    dash_length = 20
    gap_length = 10
    
    x = 0
    while x < width:
        end_x = min(x + dash_length, width)
        cv2.line(result_image, (x, y_position), (end_x, y_position), color, thickness)
        x += dash_length + gap_length
    
    # Etiqueta
    label = "Cresta Ósea Sugerida"
    cv2.putText(
        result_image,
        label,
        (10, y_position - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )
    
    return result_image

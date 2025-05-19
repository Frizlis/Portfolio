import flet as ft
import cv2
import base64
import threading
import numpy as np
from ml_integration import import_model, inference_model

class VideoCaptureHandler:
    def __init__(self, page):
        self.page = page
        # Инициализация камеры
        self.cap = None
        self.running = False
        self.initialized = False        
        # Настройки
        self.mirror = False
        self.inference = False
        self.camera_resolution = (640, 480)
        # UI элементы
        self.start_btn = ft.ElevatedButton("Start", on_click=self.start_capture, disabled=True)
        self.stop_btn = ft.ElevatedButton("Stop", on_click=self.stop_capture, disabled=False)
        self.progress_bar = ft.ProgressBar(width=300, visible=False)
        self.status_text = ft.Text("", style=ft.TextThemeStyle.HEADLINE_SMALL, visible=False)
        self.image = ft.Image(src="no_video_signal.png", width=640, height=480, fit=ft.ImageFit.CONTAIN)

        # Запуск фоновой инициализации
        self._start_initialization()
        
    def _start_initialization(self):
        # Запуск фоновой инициализации
        self.progress_bar.visible = True
        self.status_text.value = "Поиск камеры..."
        self.status_text.visible = True
        self._safe_update()
        # Запуск потоков
        threading.Thread(target=self._preinit_camera, daemon=True).start() 
        self.model = import_model()
        self.classNames = self.model.names
        
    def _safe_update(self):
        """Безопасное обновление UI"""
        try:
            self.page.update()
        except Exception as e:
            print(f"Ошибка обновления UI: {str(e)}")

    def _preinit_camera(self):
        """Фоновая инициализация камеры"""
        self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            raise RuntimeError("Камера недоступна")
        # Предварительная настройка параметров
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
        # Тестовый кадр
        ret, _ = self.cap.read()
        if not ret:
            raise RuntimeError("Ошибка чтения кадра")
        # Освобождаем ресурсы
        self.cap.release()
          
        self.initialized = True
        self.progress_bar.visible = False
        self.status_text.visible = False
        self.start_btn.disabled = not self.initialized
        self._safe_update()

    def create_camera_tab(self):
        return ft.Tab(
            text="Камера",
            icon=ft.Icons.VIDEO_CAMERA_BACK,
            content=ft.Container(ft.Column(controls=[
                self.status_text,
                self.progress_bar,
                ft.Container(self.image, expand=True, width=1000),
                ft.Row([
                    self.start_btn,
                    self.stop_btn,
                ], 
                       alignment=ft.MainAxisAlignment.CENTER)
            ],
                expand=True,
                alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            expand=True,
            padding=ft.padding.symmetric(vertical=30, horizontal=30),
            )
        )
        
    def start_capture(self, e):
        if self.initialized and not self.running:
            self.running = True
            self.stop_btn.disabled = False
            self.cap.open(0, cv2.CAP_DSHOW)
            for _ in range(5):  # Пропуск кадров автонастройки
                self.cap.read()
            threading.Thread(target=self.update_frame, daemon=True).start()
            self._safe_update()

    def stop_capture(self, e):
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.image.src_base64 = None
        self.stop_btn.disabled = True
        # self.image.update()
        self._safe_update()

    def update_frame(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: continue
            
            if self.inference:
                frame = inference_model(self.model, self.classNames, frame)
                
            if self.mirror:
                frame = cv2.flip(frame, 1)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', frame)
            
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            if self.running:
                self.image.src_base64 = img_base64
                self.image.update()
    
    def inference_model(self, frame):
        prediction = self.model.predict(frame)
        
        for r in prediction:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # class confidence
                confidence = math.ceil((box.conf[0]*100))/100
                
                # class name
                cls = int(box.cls[0])
                color = (255/len(self.classNames)*cls, 255/len(self.classNames)*(len(self.classNames)-cls), 255*len(self.classNames)/(cls+1))
                
                # object details
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                thickness = 2
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, self.classNames[cls], [x1, y1-10], font, fontScale, color, thickness)
                cv2.putText(frame, str(confidence), [x2-80, y1+30], font, fontScale, color, thickness)           
        
        return frame

def main(page: ft.Page):
    page.title = "Camera App"
    page.window.height = 800    
    page.window.width = 800
    page.window.min_height = 800
    page.window.min_width = 800
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    
    # Инициализация обработчика камеры
    camera_handler = VideoCaptureHandler(page)
    # Элементы настроек
    mirror_switch = ft.Switch(label="Зеркальное отображение")
    inference_switch = ft.Switch(label="Детекция")
    resolution_dropdown = ft.Dropdown(
        label="Разрешение",
        options=[
            ft.dropdown.Option("640x480"),
            ft.dropdown.Option("1280x720"),
            ft.dropdown.Option("1920x1080")
        ],
        value="640x480"
    )

    # Обработчик применения настроек
    def apply_settings(e):
        camera_handler.mirror = mirror_switch.value
        camera_handler.inference = inference_switch.value
        # Здесь можно добавить изменение разрешения
        
    def change_theme(e):
        page.theme_mode = (ft.ThemeMode.LIGHT if page.theme_mode == ft.ThemeMode.DARK else ft.ThemeMode.DARK)
        e.control.selected = not e.control.selected
        page.update()

    # Создаем вкладки
    tabs = ft.Tabs(
        tabs=[
            ft.Tab(
                text="Настройки",
                icon=ft.Icons.SETTINGS,
                content=ft.Row([ 
                    ft.Column([
                        ft.Text("Настройки камеры", size=20),
                        mirror_switch,
                        inference_switch,
                        resolution_dropdown,
                        ft.ElevatedButton("Применить настройки", on_click=apply_settings)
                    ], spacing=15),
                    ft.IconButton(icon=ft.Icons.SUNNY, selected_icon=ft.Icons.MODE_NIGHT, on_click=change_theme, selected=False,),
                ], 
                               vertical_alignment=ft.CrossAxisAlignment.START,
                               alignment=ft.MainAxisAlignment.SPACE_AROUND)
            ),
            camera_handler.create_camera_tab()
        ],
        expand=True
    )
    
    page.add(tabs)

ft.app(target=main)
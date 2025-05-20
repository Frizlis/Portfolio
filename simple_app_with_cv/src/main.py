import flet as ft
import cv2
import base64
import threading
import numpy as np
import time
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
        self.camera_resolution = (1920, 1080)
        # UI элементы
        self.start_btn = ft.ElevatedButton("Start", on_click=self.start_capture, disabled=True)
        self.stop_btn = ft.ElevatedButton("Stop", on_click=self.stop_capture, disabled=False)
        self.progress_bar = ft.ProgressBar(width=300, visible=False)
        self.status_text = ft.Text("", style=ft.TextThemeStyle.HEADLINE_SMALL, visible=False)
        self.image = ft.Image(src="D:\\Python\\Portfolio\\simple_app_with_cv\\no_video_signal.png", 
                              width=640, 
                              height=480, 
                              fit=ft.ImageFit.CONTAIN,)

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
        
        self.status_text.value = "Камера определена и готова к запуску."          
        self.initialized = True
        self.progress_bar.visible = False        
        self.start_btn.disabled = not self.initialized
        self._safe_update()

    def create_camera(self):
        return ft.Container(
            content=ft.Container(ft.Column(controls=[
                self.status_text,
                self.progress_bar,
                ft.Container(self.image, expand=True, width=1920),
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
            ), 
            expand=True,
        )
        
    def start_capture(self, e):
        if self.initialized and not self.running:
            self.running = True
            self.stop_btn.disabled = False
            self.cap.open(0, cv2.CAP_DSHOW)
            for _ in range(5):  # Пропуск кадров автонастройки
                self.cap.read()
            threading.Thread(target=self.update_frame, daemon=True).start()
            self.status_text.visible = False 
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
        value="1920x1080"
    )    
    
    def open_dialog(e):
        settings_dialog.open = True
        page.update()
        
    def change_theme(e):
        page.theme_mode = (ft.ThemeMode.LIGHT if page.theme_mode == ft.ThemeMode.DARK else ft.ThemeMode.DARK)
        e.control.selected = not e.control.selected
        page.update()
        
    def close_dialog(e):
        settings_dialog.open = False
        page.update()
        
        # Обработчик применения настроек
    def apply_settings(e):
        camera_handler.mirror = mirror_switch.value
        camera_handler.inference = inference_switch.value
        try:
            camera_handler.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_dropdown.value.split("x")[0]) 
            camera_handler.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_dropdown.value.split("x")[1])
            print("Разрешение изменено успешно!")
        except Exception as ex:
            print(f"Ошибка: {e}.")
        print(camera_handler.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(camera_handler.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        close_dialog(e)
        # Здесь можно добавить изменение разрешения

    settings_dialog = ft.AlertDialog(
        title=ft.Text("Settings"),
        content=ft.Column([
                        ft.Row([ft.Text("Настройки камеры", size=20),
                                ft.IconButton(icon=ft.Icons.SUNNY, 
                                              selected_icon=ft.Icons.MODE_NIGHT, 
                                              on_click=change_theme)]),
                        mirror_switch,
                        inference_switch,
                        resolution_dropdown,
                    ], 
                          tight=True,
                          spacing=15),
         actions=[
            ft.TextButton("Сохранить", on_click=apply_settings),
            ft.TextButton("Отмена", on_click=close_dialog)
        ]
    )
    
    page.overlay.append(settings_dialog)
    
    page.appbar = ft.AppBar(
        actions=[
            ft.IconButton(
                icon=ft.Icons.SETTINGS,
                on_click=open_dialog,
            )
        ]
    )
    
    page.add(camera_handler.create_camera())

ft.app(target=main)
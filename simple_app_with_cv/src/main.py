import flet as ft
import cv2
import base64
import threading
from ml_integration import import_model, inference_model, draw_gird
import os
import time

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
        self.gride = False
        self.camera_resolution = (1920, 1080)
        # UI элементы
        self.start_btn = ft.ElevatedButton("Start", on_click=self.start_capture, color=ft.Colors.WHITE, bgcolor=ft.Colors.CYAN_ACCENT_700, disabled=True)
        self.stop_btn = ft.ElevatedButton("Stop", on_click=self.stop_capture, color=ft.Colors.WHITE, bgcolor=ft.Colors.CYAN_ACCENT_700, disabled=False)
        self.reset_cam = ft.IconButton(icon=ft.Icons.UPDATE, on_click=self._start_initialization, icon_color=ft.Colors.WHITE, bgcolor=ft.Colors.CYAN_ACCENT_700, disabled=False)
        self.progress_bar = ft.ProgressBar(width=300, visible=False)
        self.status_text = ft.Text("", style=ft.TextThemeStyle.HEADLINE_SMALL, weight=ft.FontWeight.BOLD, visible=False, color=ft.Colors.BLUE_900)
        self.text_container = ft.Container(self.status_text, border=ft.border.all(2, ft.Colors.CYAN_ACCENT_700), border_radius=25, padding=ft.padding.symmetric(horizontal=10, vertical=5))
        self.image = ft.Image(src="simple_app_with_cv/src/assets/Camera_waiting.png", 
                              fit=ft.ImageFit.CONTAIN,
                              expand=True,
                              height=1080,
                              width=1940)
        self.image_container = ft.Container(self.image, expand=True, alignment=ft.alignment.center, 
                                            padding=0, margin=0, border_radius=25, 
                                            height=1080, width=1940, 
                                            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,)

        # Запуск фоновой инициализации
        self._start_initialization()
        
    def _start_initialization(self, e=None):
        # Запуск фоновой инициализации
        self.progress_bar.visible = True
        self.text_container.visible = True
        self.status_text.value = "Поиск камеры..."
        self.status_text.visible = True   
        self.image.src_base64 = None
        self.image.src = "simple_app_with_cv/src/assets/Camera_waiting.png"
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

    def _error_camera(self, message):
            self.status_text.value = message
            self.image.src = "simple_app_with_cv/src/assets/Camera_no_signal.png"
            self.progress_bar.visible = False
            self.stop_btn.disabled = True
            self._safe_update()

    def _preinit_camera(self):
        """Фоновая инициализация камеры"""
        self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not self.cap.isOpened(): 
            self._error_camera("Камера недоступна")
            raise RuntimeError("Камера недоступна")
        # Предварительная настройка параметров
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
        # Тестовый кадр
        ret, _ = self.cap.read()
        if not ret:
            self._error_camera("Ошибка чтения кадра")
            raise RuntimeError("Ошибка чтения кадра")
        # Освобождаем ресурсы
        self.cap.release()
        
        self.status_text.value = "Камера определена и готова к запуску"          
        self.initialized = True
        self.progress_bar.visible = False        
        self.start_btn.disabled = not self.initialized
        self.image.src = "simple_app_with_cv/src/assets/Camera_ready.png"
        self._safe_update()

    def create_camera(self):
        return ft.Container(
            content=ft.Container(ft.Column(controls=[
                self.text_container,
                self.progress_bar,
                self.image_container,        
                ft.Row([
                    self.start_btn,
                    self.stop_btn,
                    self.reset_cam,
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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
            for _ in range(5):  # Пропуск кадров автонастройки
                self.cap.read()
            threading.Thread(target=self.update_frame, daemon=True).start()
            self.status_text.visible = False 
            self.text_container.visible = False
            self.image_container.shadow = ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.Colors.BLUE_GREY_300,
                offset=ft.Offset(0, 0),
                blur_style=ft.ShadowBlurStyle.OUTER,
            )
            self._safe_update()

    def stop_capture(self, e):
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.image.src_base64 = None
        self.stop_btn.disabled = True
        self.image_container.shadow = None
        self._safe_update()

    def update_frame(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: continue
            
            if self.mirror:
                frame = cv2.flip(frame, 1)
            
            if self.inference:
                frame = inference_model(self.model, self.classNames, frame)
                
            if self.gride:
                frame = draw_gird(frame)         
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', frame)
            
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            if self.running:
                self.image.src_base64 = img_base64
                self.image.update()
        
        if not self.cap.isOpened():
            self.image.src_base64 = None
            self.image.src = "simple_app_with_cv/src/assets/Camera_no_signal.png"
        
    def camera_set(self, W, H):
        self.camera_resolution = (W, H)
        if not self.cap.isOpened():
            self.cap.open(0, cv2.CAP_DSHOW)        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
        return (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    mirror_switch = ft.Ref[ft.Switch]()
    inference_switch = ft.Ref[ft.Switch]()
    grid_switch = ft.Ref[ft.Switch]()
    resolution_dropdown = ft.Ref[ft.Dropdown]()
    
    def open_dialog(e):
        mirror_switch.current.value = camera_handler.mirror
        inference_switch.current.value = camera_handler.inference
        grid_switch.current.value = camera_handler.gride
        try:
            resolution_dropdown.current.value = str(int(camera_handler.cap.get(cv2.CAP_PROP_FRAME_WIDTH))) + "x" + str(int(camera_handler.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        except:
            resolution_dropdown.current.value = None
        settings_dialog.open = True
        page.update()
    
    def resolution_change(e):
        snack_bar = ft.SnackBar(                
                content=ft.Text("Ошибка: возможно ваша камера не поддерживает данное разрешение", color=ft.Colors.WHITE),
                bgcolor=ft.Colors.RED_700,
                open=True
            )
        try:
            W = int(resolution_dropdown.current.value.split("x")[0])
            H = int(resolution_dropdown.current.value.split("x")[1])
            new_W, new_H = camera_handler.camera_set(W, H)
            if W==new_W and H == new_H:
                snack_bar.content = ft.Text("Разрешение изменено успешно!", color=ft.Colors.WHITE)
                snack_bar.bgcolor = ft.Colors.CYAN_ACCENT_700
            else:
                snack_bar.content = ft.Text(f"Ошибка: Разрешение не поддерживается. Текущее разрешение {int(new_W)}x{int(new_H)}", color=ft.Colors.WHITE)
                snack_bar.bgcolor = ft.Colors.YELLOW_700
                resolution_dropdown.current.value = f"{int(new_W)}x{int(new_H)}"
        except Exception as exc:
            print(exc)
            snack_bar.content = ft.Text("Ошибка:  Дождитесь определения камеры. Возможно ваша камера не поддерживает данное разрешение.", color=ft.Colors.WHITE)
            snack_bar.bgcolor = ft.Colors.RED_700
        page.add(snack_bar)
        page.update()
        time.sleep(3)
        snack_bar.open = False
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
        camera_handler.mirror = mirror_switch.current.value
        camera_handler.inference = inference_switch.current.value
        camera_handler.gride = grid_switch.current.value
        close_dialog(e)
        # Здесь можно добавить изменение разрешения

    settings_dialog = ft.AlertDialog(
        title=ft.Text("Settings"),
        content=ft.Column([
            ft.Row([
                ft.Text("Настройки камеры", size=20),
                ft.IconButton(
                    icon=ft.Icons.SUNNY, 
                    selected_icon=ft.Icons.MODE_NIGHT, 
                    on_click=change_theme)
            ]),
            ft.Switch(ref=mirror_switch, label="Зеркальное отображение"),
            ft.Switch(ref=grid_switch, label="Сетка"),
            ft.Switch(ref=inference_switch, label="Детекция"),
            ft.Dropdown(
                ref=resolution_dropdown,
                label="Quality",
                options=[
                    ft.dropdown.Option("320x240"),
                    ft.dropdown.Option("640x480"),
                    ft.dropdown.Option("1280x720"),
                    ft.dropdown.Option("1920x1080")
                ],
                value="640x480",
                on_change = resolution_change,
            )], 
            tight=True,
            spacing=15),
        actions=[
            ft.TextButton("Сохранить", on_click=apply_settings),
            ft.TextButton("Отмена", on_click=close_dialog),
        ]
    )
    
    page.overlay.append(settings_dialog)
    
    page.appbar = ft.AppBar(
        actions=[
            ft.IconButton(
                icon=ft.Icons.SETTINGS,
                on_click=open_dialog,
                icon_color=ft.Colors.BLUE_900,
            )
        ]
    )
    
    page.add(camera_handler.create_camera())

ft.app(target=main)
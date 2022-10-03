from kivy.app import App
from kivy.properties import (
    NumericProperty, ListProperty, StringProperty, BooleanProperty, DictProperty
)
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.switch import Switch
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from tensorflow import keras
from math import floor, ceil


class UI(BoxLayout):
    frames_per_second = NumericProperty(60.0)

    screen_ratio = NumericProperty(Window.size[0] / Window.size[1])
    capture_button_size = NumericProperty(min(Window.size) * 0.2)
    capture_button_source = StringProperty('camera_button_white.png')
    switch_cam_source = StringProperty('refresh.png')

    camera_size = ListProperty([1, 1])
    camera_display_size = ListProperty([1, 1])
    camera_display_pos = ListProperty([1, 1])
    screen_size = ListProperty([Window.size[0], Window.size[1]])

    camera_display = Image()
    continuous_mode = Switch()

    color = DictProperty({'gray': [80, 80, 80], 'red': [255, 0, 0],
                          'orange': [255, 69, 0], 'green': [0, 255, 0],
                          'yellow': [0, 0, 255], 'white': [1, 1, 1]})
    verify_colors = ListProperty([[80, 80, 80], [80, 80, 80], [80, 80, 80], [80, 80, 80]])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cam_index = 0
        self.max_cam_index = 10
        self.capture = cv2.VideoCapture(self.cam_index)
        if self.capture is None:
            self.switch_cam()

        self.picture = 0
        self.clicked = False
        self.nn_model = keras.models.load_model('nn_model')
        self.verify(np.zeros((300, 300, 3)))

        self.prediction = []
        self.predictions = []
        self.N = 10
        self.My_Clock = Clock
        self.My_Clock.schedule_interval(self.update, 1 / self.frames_per_second)


    def update(self, *args):
        # get window size and set capture_button_size
        self.capture_button_size = min(Window.size) * 0.1
        self.screen_size = [Window.size[0], Window.size[1]]
        self.screen_ratio = Window.size[0] / Window.size[1]

        # update camera_display according to the window size
        if self.width < self.height:
            self.camera_display_size = [self.width, self.width * self.camera_size[0] / self.camera_size[1]]
            self.camera_display_pos = [0, self.capture_button_size + 30]
        else:
            new_height = self.height - self.capture_button_size - 30
            new_width = new_height * self.camera_size[1] / self.camera_size[0]
            self.camera_display_size = [new_width, new_height]
            self.camera_display_pos = [(Window.size[0] - new_width)/2, self.capture_button_size + 30]

        # if photo is taken don't get new pic
        if not self.clicked or self.continuous_mode.active:
            self.get_pic()

        # continuously verify
        if self.continuous_mode and self.clicked:
            self.verify(self.picture)
            if len(self.predictions) == self.N:
                self.update_colors()

    def get_pic(self):
        success, frame = self.capture.read()
        if not success:
            print('fail')
            return

        self.camera_size = np.shape(frame)
        buf = cv2.flip(frame, 0).tobytes()

        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")

        self.picture = frame
        self.camera_display.texture = texture

    def capture_click(self):
        self.clicked = not self.clicked

        if self.clicked:
            if self.continuous_mode.active:
                self.capture_button_source = 'camera_button_red.png'
            else:
                self.capture_button_source = 'reload.png'
                self.verify(self.picture)
                self.update_colors()
        else:
            self.prediction = [0, 0, 0, 0]
            self.predictions.clear()
            self.capture_button_source = 'camera_button_white.png'
            self.update_colors()

    def switch_callback(self):
        # continuous_mode needs to be changed manualy, switch doesn't do it auto
        self.continuous_mode.active = not self.continuous_mode.active
        self.capture_button_source = 'camera_button_white.png'
        self.clicked = False
        self.prediction = [0, 0, 0, 0]
        self.predictions.clear()
        self.update_colors()

    def verify(self, img):
        img = self.resize_image(img, 300)
        prediction = self.nn_model.predict(img, batch_size=None, verbose=0, steps=None,callbacks=None,
                                           workers=8, use_multiprocessing=True).round(0)

        if self.continuous_mode.active:
            self.predictions.append(prediction[0])
            if len(self.predictions) > self.N:
                del self.predictions[0]
            self.prediction = list(np.sum(self.predictions, axis=0))
            one_hot = self.prediction.index(np.max(self.prediction))
            self.prediction = np.eye(4)[one_hot]
        else:
            self.prediction = prediction[0]

    def resize_image(self, img, new_size):
        height = np.size(img, axis=0)
        width = np.size(img, axis=1)

        if height > width:
            new_width = floor(new_size * width / height)
            img = cv2.resize(img, dsize=(new_width, new_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.copyMakeBorder(img, top=0, bottom=0, left=floor((new_size - new_width) / 2),
                                     right=ceil((new_size - new_width) / 2),
                                     borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            new_height = floor(new_size * height / width)
            img = cv2.resize(img, dsize=(new_size, new_height), interpolation=cv2.INTER_CUBIC)
            img = cv2.copyMakeBorder(img, top=floor((new_size - new_height) / 2),
                                     bottom=ceil((new_size - new_height) / 2),
                                     left=0, right=0,
                                     borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return np.reshape(img, (1, new_size, new_size, 3))

    def update_colors(self):
        if self.prediction[0] == 1:
            self.verify_colors = [self.color['green'], self.color['green'], self.color['green'], self.color['green']]
        elif self.prediction[1] == 1:
            self.verify_colors = [self.color['gray'], self.color['orange'], self.color['orange'], self.color['orange']]
        elif self.prediction[2] == 1:
            self.verify_colors = [self.color['gray'], self.color['gray'], self.color['orange'], self.color['orange']]
        elif self.prediction[3] == 1:
            self.verify_colors = [self.color['gray'], self.color['gray'], self.color['gray'], self.color['red']]
        else:
            self.verify_colors = [self.color['gray'], self.color['gray'], self.color['gray'], self.color['gray']]

    def switch_cam(self):
        self.My_Clock.unschedule(self.update)
        for i in range(self.max_cam_index):
            self.cam_index += 1
            self.cam_index %= self.max_cam_index
            self.capture = cv2.VideoCapture(self.cam_index)
            success, frame = self.capture.read()
            if success:
                break

        self.My_Clock.schedule_interval(self.update, 1 / self.frames_per_second)


class MaskDetectionApp(App):
    def build(self):
        return UI()


if __name__ == '__main__':
    MaskDetectionApp().run()

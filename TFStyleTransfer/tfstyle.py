#!/usr/bin/env python
# coding: utf-8

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap


def clickable(widget):
    class Filter(QtWidgets.QWizard):
        clicked = pyqtSignal()

        def eventFilter(self, obj, event):
            if obj == widget:
                if event.type() == QEvent.MouseButtonRelease:
                    if obj.rect().contains(event.pos()):
                        self.clicked.emit()
                        return True
            return False

    start_filter = Filter(widget)
    widget.installEventFilter(start_filter)
    return start_filter.clicked


class Ui(QtWidgets.QWizard):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi("style.ui", self)

        self.pushButton.clicked.connect(self.input_user_photo)
        clickable(self.label_5).connect(self.input_user_style)
        clickable(self.label_style1).connect(self.input_style1)
        clickable(self.label_style2).connect(self.input_style2)
        clickable(self.label_style3).connect(self.input_style3)

        self.startTf.clicked.connect(self.do_action)
        self.progressBar.setMaximum(100)
        self.timer = QBasicTimer()
        self.step = 0

    def input_user_photo(self):
        file_name_f = QtWidgets.QFileDialog.getOpenFileName(self, "Выбор изображения", "/", "Image Files (*.png *.jpg)")
        self.lineEdit.setText(file_name_f[0])
        userinfo_f = QPixmap(file_name_f[0]).scaled(300, 200)
        self.label_2.setPixmap(userinfo_f)
        global target_image_path
        target_image_path = file_name_f[0]

    def input_user_style(self):
        file_name_s = QtWidgets.QFileDialog.getOpenFileName(self, "Выбор изображения", "/", "Image Files (*.png *.jpg)")
        userinfo_s = QPixmap(file_name_s[0]).scaled(300, 200)
        self.label_5.setPixmap(userinfo_s)
        global style_reference_image_path
        style_reference_image_path = file_name_s[0]

    def input_style1(self):
        global style_reference_image_path
        style_reference_image_path = "img/s1.jpg"
        self.next()

    def input_style2(self):
        global style_reference_image_path
        style_reference_image_path = "img/s2.jpg"
        self.next()

    def input_style3(self):
        global style_reference_image_path
        style_reference_image_path = "img/s3.jpg"
        self.next()

    def timerEvent(self, e):

        if self.step >= 100:
            self.timer.stop()
            self.startTf.setText('Сохранить изображение')
            photo_result = "C:/Users/User/PycharmProjects/TFTest/img/my_result_at_iteration_1.png"
            icon_photo_result = QPixmap(photo_result).scaled(300, 200)
            self.label_7.setPixmap(icon_photo_result)
            # self.startTf.clicked(self.outputImage)
            return

        self.step = self.step + 10
        self.progressBar.setValue(self.step)

    def output_image(self):
        image_path = "C:/Users/User/PycharmProjects/TFTest/img/my_result_at_iteration_1.png"
        QtWidgets.QFileDialog.getSaveFileName(self, "Выбор изображения", image_path, "Image Files (*.png *.jpg)")

    def do_action(self):

        if self.timer.isActive():
            self.timer.stop()
            self.startTf.setText('Начать обработку')
        else:
            self.timer.start(100, self)
            self.startTf.setText('Прервать обработку')

            # Начало процесса обработки изображения
            import tensorflow as tf

            print("Start process...")

            # Отключение предупреждения, не включается AVX/FMA
            import os

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

            # определение начальных переменных
            from tensorflow.keras.preprocessing.image import load_img, img_to_array

            # target_image_path = 'img/target.jpg'
            # print(target_image_path)
            # target_image_path - путь к изображению, которое будет трансформироваться
            width, height = load_img(target_image_path).size
            # размеры генерируемого изображения
            img_height = 400
            img_width = int(width * img_height / height)

            # вспомогательные функции
            import numpy as np
            from tensorflow.keras.applications import vgg19

            def preprocess_image(image_path):
                p_image = load_img(image_path, target_size=(img_height, img_width))
                p_image = img_to_array(p_image)
                p_image = np.expand_dims(p_image, axis=0)
                p_image = vgg19.preprocess_input(p_image)
                return p_image

            def process_image(process_photo):
                # нулевое центрирование путем удаления среднего значения
                # пиксела из ImageNet. Это отменяет преобразование,
                # выполненное vgg19.preprocess_input
                process_photo[:, :, 0] += 103.939
                process_photo[:, :, 1] += 116.779
                process_photo[:, :, 2] += 123.68
                # Конвертация изображения из BGR
                # в RGB. Также является частью обратного порядка vgg19.preprocess_input
                process_photo = process_photo[:, :, ::-1]
                process_photo = np.clip(process_photo, 0, 255).astype('uint8')
                return process_photo

            # изменение размера изображения
            # style_reference_image_path = 'img/transfer_style_reference.jpg'
            width, height = load_img(style_reference_image_path).size
            img_height = 400
            img_width = int(width * img_height / height)

            # загрузка VGG
            from tensorflow.keras import backend as backeras

            target_image = backeras.constant(preprocess_image(target_image_path))
            style_reference_image = backeras.constant(preprocess_image(style_reference_image_path))
            # Заготовка, куда будет помещено сгенерированное изображение
            combination_image = backeras.placeholder((1, img_height, img_width, 3))
            # Объединение трех изображений в один пакет
            input_tensor = backeras.concatenate([target_image, style_reference_image, combination_image], axis=0)
            # Конструирование сети VGG19 с пакетом из трех изображений на
            # входе. В модель будут загружены веса, полученные в результате
            # обучения на наборе ImageNet
            model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
            print('Model loaded.')

            # Функция потерь содержимого

            def content_loss(base, combination):
                return backeras.sum(backeras.square(combination - base))

            # Функция потерь стиля

            def gram_matrix(process_photo):
                features = backeras.batch_flatten(backeras.permute_dimensions(process_photo, (2, 0, 1)))
                gram = backeras.dot(features, backeras.transpose(features))
                return gram

            def style_loss(style, combination):
                s = gram_matrix(style)
                c = gram_matrix(combination)
                channels = 3
                size = img_height * img_width
                return backeras.sum(backeras.square(s - c)) / (4. * (channels ** 2) * (size ** 2))

            # Функция общей потери вариации

            def total_variation_loss(process_photo):
                a = backeras.square(process_photo[:, :img_height - 1, :img_width - 1, :] - process_photo[:, 1:, :img_width - 1, :])
                b = backeras.square(process_photo[:, :img_height - 1, :img_width - 1, :] - process_photo[:, :img_height - 1, 1:, :])
                return backeras.sum(backeras.pow(a + b, 1.25))

            # Функция общей потери вариации

            loss = backeras.variable(0.)
            try:
                outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
                layer_features = outputs_dict['block1_conv1']
                style_reference_features = layer_features[1, :, :, :]
                combination_features = layer_features[2, :, :, :]
                style_loss(style_reference_features, combination_features)
            except ValueError:
                #Словарь, отображающий имена слоев в тензоры активаций
                outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
                # Слой, используемый для вычисления потерь содержимого
                content_layer = 'block5_conv2'
                # Слой, используемый для вычисления потерь стиля
                style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
                # Веса для вычислениясреднего
                # взвешенного по компонентам потерь
                total_variation_weight = 1e-4
                style_weight = 1.
                content_weight = 0.025

                # Величина потерь определяется сложением всех
                # компонентов с этой переменной
                loss = backeras.variable(0.)
                #Добавление потери содержимого
                layer_features = outputs_dict[content_layer]
                target_image_features = layer_features[0, :, :, :]
                combination_features = layer_features[2, :, :, :]
                loss = loss + content_weight * content_loss(target_image_features, combination_features)

                # Добавление потери стиля для каждого целевого уровня
                for layer_name in style_layers:
                    layer_features = outputs_dict[layer_name]
                    style_reference_features = layer_features[1, :, :, :]
                    combination_features = layer_features[2, :, :, :]
                    sl = style_loss(style_reference_features, combination_features)
                    loss += (style_weight / len(style_layers)) * sl

                # Добавление общей потери вариации
                loss += total_variation_weight * total_variation_loss(combination_image)

            # Подготовка процедуры градиентного спуска

            tf.compat.v1.disable_eager_execution()
            # Получение градиентов сгенерированного
            # изображения относительно потерь
            grads = backeras.gradients(loss, combination_image)[0]
            try:
                # Функция дляполучения значений
                # текущих потерь и градиентов
                fetch_loss_and_grads = backeras.function([combination_image], [loss, grads])
            except ValueError:
                target_image = backeras.constant(preprocess_image(target_image_path))
                style_reference_image = backeras.constant(preprocess_image(style_reference_image_path))
                combination_image = backeras.placeholder((1, img_height, img_width, 3))

                input_tensor = backeras.concatenate([target_image, style_reference_image, combination_image], axis=0)

                model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
                outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
                content_layer = 'block5_conv2'
                style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
                total_variation_weight = 1e-4
                style_weight = 1.
                content_weight = 0.025

                loss = backeras.variable(0.)
                layer_features = outputs_dict[content_layer]
                target_image_features = layer_features[0, :, :, :]
                combination_features = layer_features[2, :, :, :]
                loss = loss + content_weight * content_loss(target_image_features, combination_features)

                for layer_name in style_layers:
                    layer_features = outputs_dict[layer_name]
                    style_reference_features = layer_features[1, :, :, :]
                    combination_features = layer_features[2, :, :, :]
                    sl = style_loss(style_reference_features, combination_features)
                    loss += (style_weight / len(style_layers)) * sl

                loss += total_variation_weight * total_variation_loss(combination_image)
                grads = backeras.gradients(loss, combination_image)[0]
                fetch_loss_and_grads = backeras.function([combination_image], [loss, grads])

            # Этот класс обертывает fetch_loss_and_grads
            # и позволяет получать потери и
            # градиенты вызовами двух отдельных методов,
            # как того требует реализация
            # оптимизатора из SciPy
            class Evaluator(object):

                def __init__(self):
                    self.loss_value = None
                    self.grads_values = None

                def loss(self, process_photo):
                    assert self.loss_value is None
                    process_photo = process_photo.reshape((1, img_height, img_width, 3))
                    outs = fetch_loss_and_grads([process_photo])
                    loss_value = outs[0]
                    grad_values = outs[1].flatten().astype('float64')
                    self.loss_value = loss_value
                    self.grad_values = grad_values
                    return self.loss_value

                def grads(self, process_photo):
                    assert self.loss_value is not None
                    grad_values = np.copy(self.grad_values)
                    self.loss_value = None
                    self.grad_values = None
                    return grad_values

            evaluator = Evaluator()

            # Цикл передачи стиля

            from scipy.optimize import fmin_l_bfgs_b
            from matplotlib.pyplot import imsave
            import time

            result_prefix = 'img/my_result'
            iterations = 20
            # Первичное состояние: целевое изображение
            x = preprocess_image(target_image_path)
            # Проеобразование изображение, потому что scipy.optimize.
            # fmin_l_bfgs_b могут обрабатывать только плоские векторы
            x = x.flatten()
            for i in range(iterations):
                print('Start of iteration', i)
                start_time = time.time()
                # Выполняет оптимизацию L-BFGS по пикселам генерируемого изображения, чтобы минимизировать потерю стиля.
                # Нужно передать функцию, которая вычисляет потерю, и функцию, которая вычисляет
                # градиенты, как два отдельных аргумента
                x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
                print('Current loss value:', min_val)
                # Сохранение текущего сгенерированного изображения
                img = x.copy().reshape((img_height, img_width, 3))
                img = process_image(img)
                filename = result_prefix + '_at_iteration_%d.png' % i
                imsave(filename, img)
                print('Image saved as', filename)
                end_time = time.time()
                print('Iteration %d completed in %ds' % (i, end_time - start_time))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())

import cv2
import itertools
import numpy as np
import random
import os

class Stack():
    def __init__(self):
        self.item = []

    def push(self, value):
        self.item.append(value)

    def pop(self):
        return self.item.pop()

    def size(self):
        return len(self.item)

    def isEmpty(self):
        return self.size() == 0

    def clear(self):
        self.item = []

class RegionGrow():
    def __init__(self, image_path, threshold):
        self.image_path = image_path
        self.threshold = float(threshold)
        self.read_image()
        self.h, self.w = self.im_gray.shape
        self.passed_by = np.zeros((self.h, self.w), np.int32)
        self.current_region = 0
        self.iterations = 0
        self.stack = Stack()

    def read_image(self):
        im = cv2.imread(self.image_path)
        if im is None:
            raise ValueError(f"Erro ao carregar imagem: {self.image_path}")
        im = cv2.GaussianBlur(im, (5, 5), 0)
        self.im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        self.im_gray = self.im_hsv[:, :, 2]  # canal V
        self.im = im
        self.original = im.copy()

    def get_neighbours(self, x0, y0):
        return [
            (x, y)
            for i, j in itertools.product((-1, 0, 1), repeat=2)
            if (i, j) != (0, 0) and self.in_bounds(x := x0 + i, y := y0 + j)
        ]

    def apply(self):
        for x0 in range(0, self.h, 5):
            for y0 in range(0, self.w, 5):
                if self.passed_by[x0, y0] == 0 and self.im_gray[x0, y0] < 80:
                    self.current_region += 1
                    self.stack.push((x0, y0))
                    self.passed_by[x0, y0] = self.current_region

                    while not self.stack.isEmpty():
                        x, y = self.stack.pop()
                        self.bfs(x, y)
                        self.iterations += 1

        print(f"Total de regiões: {self.current_region} | Iterações: {self.iterations}")
        self.export_results()

    def bfs(self, x0, y0):
        region_num = self.passed_by[x0, y0]
        mean_val = int(self.im_gray[x0, y0])
        var = self.threshold

        for x, y in self.get_neighbours(x0, y0):
            if self.passed_by[x, y] == 0:
                diff = abs(int(self.im_gray[x, y]) - mean_val)
                if diff < var:
                    self.passed_by[x, y] = region_num
                    self.stack.push((x, y))

    def in_bounds(self, x, y):
        return 0 <= x < self.h and 0 <= y < self.w

    def export_results(self):
        mask = np.where(self.passed_by > 0, 255, 0).astype(np.uint8)
        cv2.imwrite("mascara.png", mask)

        overlay = self.original.copy()
        overlay[mask == 255] = (0, 0, 255)  # vermelho
        result = cv2.addWeighted(self.original, 0.7, overlay, 0.3, 0)
        cv2.imwrite("segmentado.png", result)

        print("Arquivos salvos: 'mascara.png' e 'segmentado.png'")

        # Redimensiona imagens para visualização lado a lado
        max_height = 400
        scale = max_height / self.original.shape[0]
        new_w = int(self.original.shape[1] * scale)
        new_h = max_height

        orig_resized = cv2.resize(self.original, (new_w, new_h))
        result_resized = cv2.resize(result, (new_w, new_h))

        side_by_side = np.hstack((orig_resized, result_resized))

        cv2.imshow("Original | Segmentado", side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ---------- USO DIRETO AQUI ----------
if __name__ == "__main__":
    imagem = "imgs/ol2.jpg"  # Caminho da imagem
    threshold = 5              # Threshold de crescimento (experimente 3 a 10)

    rg = RegionGrow(imagem, threshold)
    rg.apply()

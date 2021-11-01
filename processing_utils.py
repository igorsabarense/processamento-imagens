""" __author__ = "Bruno Rodrigues, Igor Sabarense e Raphael Nogueira"
    __date__ = "2021"
"""
import cv2
import numpy as np
from keras.utils.np_utils import normalize
from scipy.interpolate import interpolate
from scipy.ndimage import interpolation


def find_white_background(imgArr, threshold=0.1815):
    """ Retorna a cor do fundo da imagem ( preto ou branco )
    :param imgArr: npArray
    :param threshold:  float
    :returns: True or False: bool
    """

    background = np.array([255, 255, 255])
    percent = (imgArr == background).sum() / imgArr.size
    if percent >= threshold or percent == 0 or percent <= 0.001:
        return True
    else:
        return False


def sort_contours(contours):
    """ Retorna os contornos encontrados de forma ordenada da esquerda para direita
    :param contours: cv2.Countours
    :returns: countours: cv2.Countours
    """
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                            key=lambda b: b[1][i]))

    return contours


def resize_image(img, size=(18, 18)):
    """  Retorna a imagem reajustada para o valor escolhido , caso não passado, o valor default é 18,18
    :param img: npArray - imagem a ser ajustada o tamanho
    :param size: tuple(x,y)- tamanho que a imagem deve ser ajustada ( tupla )
    :returns: img np.array
    """

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation_ = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC

    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation_)


def interpolate_projection(projection, size=32):
    """  Retorna a projeção interpolada para o valor escolhido, caso não especificado, o valor default é 32.
        :param size: tamanho a ser interpolado
        :param projection : lista
    """
    interpol_func = interpolate.interp1d(np.arange(0, len(projection)), projection)
    stretch_array = interpol_func(np.linspace(0.0, len(projection) - 1, size))
    return stretch_array.tolist()


def getHorizontalProjectionProfile(image):
    """  Retorna a projeção horizontal somando os valores no eixo x
           :param image : npArray
    """
    horizontal_projection = np.sum(image, axis=1)
    return horizontal_projection.tolist()


def getVerticalProjectionProfile(image):
    """  Retorna a projeção vertical somando os valores no eixo y
          :param image : npArray
    """
    vertical_projection = np.sum(image, axis=0)
    return vertical_projection.tolist()


def moments(image):
    """  Retorna a matriz de covariancia
              :param image : npArray
              :return mu_vector, covariance_matrix
    """
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]  # A trick in numPy to create a mesh grid
    totalImage = np.sum(image)  # sum of pixels
    m0 = np.sum(c0 * image) / totalImage  # mu_x
    m1 = np.sum(c1 * image) / totalImage  # mu_y
    m00 = np.sum((c0 - m0) ** 2 * image) / totalImage  # var(x)
    m11 = np.sum((c1 - m1) ** 2 * image) / totalImage  # var(y)
    m01 = np.sum((c0 - m0) * (c1 - m1) * image) / totalImage  # covariance(x,y)
    mu_vector = np.array([m0, m1])  # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00, m01], [m01, m11]])  # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix


def deskew(image):
    """  Retorna a imagem alinhada
                  :param image : npArray
                  :returns image
    """
    c, v = moments(image)
    alpha = v[0, 1] / v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape) / 2.0
    offset = c - np.dot(affine, ocenter)
    return interpolation.affine_transform(image, affine, offset=offset)


def process_projection_profile(arg):
    """  Retorna a projeção concatenada ( vertical / horizontal ) e normalizada da imagem
         :param arg : npArray -> imagem a ser transformada em sua projeção.
         :return vh : npArray -> projecão concatenada.
    """
    img = deskew(arg.copy())
    vertical_proj = getVerticalProjectionProfile(img.copy())
    horizontal_proj = getHorizontalProjectionProfile(img.copy())
    vh = interpolate_projection(vertical_proj) + interpolate_projection(horizontal_proj)
    return normalize(vh, axis=0)[0]

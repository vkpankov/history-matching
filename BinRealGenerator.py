import typing
import utils
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

class BinRealGenerator:

    def __cov_matrix_init__(self, covariance_matrix_path: str, U_cache_file: str, S_cache_file: str, Vh_cache_file: str, param_count: int) -> None:
        cov_matrix = np.load(covariance_matrix_path)
        self.cov_mean = cov_matrix.mean(axis=1, keepdims=True)
        centered_cov_matrix = cov_matrix - self.cov_mean
        Y = 1/np.sqrt(cov_matrix.shape[1] - 1) * centered_cov_matrix
        if(not U_cache_file==""):
            U, S, Vh = np.load(U_cache_file), np.load(S_cache_file), np.load(Vh_cache_file) 
        else:
            U, S, Vh = np.linalg.svd(Y, full_matrices=False)

        np.save("U", U)
        np.save("S", S)
        np.save("Vh", Vh)
       
        self.U_l = U[:,:param_count]
        self.Sigma_L = np.diag(S[:param_count])

    def __get_pca_real__(self, ksi):
        pca_real = (self.U_l.dot(self.Sigma_L)).dot(ksi)
        pca_real = pca_real.reshape(pca_real.shape[0],1) + self.cov_mean
        pca_real = pca_real.reshape((92,92))
        pca_real = pca_real - pca_real.min()
        pca_real = pca_real/pca_real.max()*255
        return pca_real

    def get_ksi_by_image(self, img):
        return np.linalg.inv(self.Sigma_L).dot(self.U_l.T).dot(img.reshape(92*92,1) - self.cov_mean)

    def opca_transform(self, img, gamma):

        mask0 = img <= gamma/2
        mask1 =img > 1 - gamma/2
        mask = np.logical_and(img > gamma/2, img < 1 - gamma/2)
        img[mask0] = 0
        img[mask1] = 1
        img[mask] = (img[mask] - gamma/2)/(1-gamma)
        return img
    def hardthreshold_transform(self, img):
        mask0 = img <= 0.5
        mask1 =img > 0.5
        img[mask0] = 0
        img[mask1] = 1    
        return img

    def __init__(self, covariance_matrix_path: str, param_model_path: str, param_count: int, sand_perm_val: float, mud_perm_val: float, U_cache_file: str = "", S_cache_file: str = "", Vh_cache_file:str = "") -> None:
        self.param_count = param_count
        self.sand_perm_val = sand_perm_val
        self.mud_perm_val = mud_perm_val
        self.__cov_matrix_init__(covariance_matrix_path, U_cache_file, S_cache_file, Vh_cache_file, param_count)
        self.transformer = utils.load_model(param_model_path)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    def get_real(self, ksi = None, transform_func = "opca", gamma = 0.7):
        if(ksi is None):
            ksi = np.random.normal(size = self.param_count)

        content_image = self.__get_pca_real__(ksi)
        content_image = self.image_transform(content_image)
        content_image = content_image.unsqueeze(0).float()

        out = self.transformer(content_image.to("cuda"))

        img = out[0].cpu().detach().clone().clamp(0, 255).numpy()
        img = img.astype("float")
        bitarray = np.array(img)[0,:,:]
        bitarray = bitarray - bitarray.min()
        bitarray = bitarray / bitarray.max()

        if(transform_func == "opca"):
            threshold_mod  = self.opca_transform(bitarray, gamma)
        else:
            threshold_mod = self.hardthreshold_transform(bitarray)

        permiabilities = 200*np.exp(np.log(10)*threshold_mod)
        porosities = 0.25 * threshold_mod + 0.15 * (1 - threshold_mod)

        return permiabilities, porosities
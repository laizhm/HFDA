from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.fft
import numpy as np

class PASTA_OR:
    """
    Apply PASTA augmentation
    - Proportional Amplitude Spectrum Training Augmentation for Syn-to-Real Domain Generalization
    """

    def __init__(
        self,
        mode="prop",
        alpha=3.0,
        k=2,
        beta=0.25,
    ):
        self.mode = mode
        self.alpha = alpha
        self.k = k
        self.beta = beta

    def __call__(self, img):
        """Call function to apply PASTA to images.

        Args:
            img (PIL image): input image

        Returns:
            aug_img (PIL image): PASTA augmented image
        """

        img = transforms.ToTensor()(img)
        fft_src = torch.fft.fftn(img, dim=[-2, -1])
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)

        X, Y = amp_src.shape[1:]
        X_range, Y_range = None, None

        if X % 2 == 1:
            X_range = np.arange(-1 * (X // 2), (X // 2) + 1)
        else:
            X_range = np.concatenate(
                [np.arange(-1 * (X // 2) + 1, 1), np.arange(0, X // 2)]
            )

        if Y % 2 == 1:
            Y_range = np.arange(-1 * (Y // 2), (Y // 2) + 1)
        else:
            Y_range = np.concatenate(
                [np.arange(-1 * (Y // 2) + 1, 1), np.arange(0, Y // 2)]
            )

        XX, YY = np.meshgrid(Y_range, X_range)

        exp = self.k
        lin = self.alpha
        offset = self.beta

        if self.mode == "prop":
            inv = np.sqrt(np.square(XX) + np.square(YY))
            inv *= (1 / inv.max()) * lin
            inv = np.power(inv, exp)
            inv = np.tile(inv, (3, 1, 1))
            inv += offset
            prop = np.fft.fftshift(inv, axes=[-2, -1])
            amp_src = amp_src * np.random.normal(np.ones(prop.shape), prop)
        else:
            prop = offset
            amp_src = amp_src * np.random.normal(np.ones(XX.shape), prop)

        aug_img = amp_src * torch.exp(1j * pha_src)
        aug_img = torch.fft.ifftn(aug_img, dim=[-2, -1])
        aug_img = torch.real(aug_img)
        aug_img = torch.clip(aug_img, 0, 1)
        aug_img = transforms.ToPILImage()(aug_img)
        return aug_img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nmode={self.mode},\n"
        repr_str += "alpha="
        repr_str += f"{self.alpha},\n"
        repr_str += "k="
        repr_str += f"{self.k},\n"
        repr_str += f"beta={self.beta})"
        return repr_str
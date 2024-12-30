from benchopt import BaseObjective, safe_import_context
from benchopt.config import get_data_path

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    import deepinv as dinv
    from deepinv.loss.metric import PSNR, SSIM
    from deepinv.utils.demo import load_degradation


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Denoising priors"

    # URL of the main repo for this benchmark.
    url = "https://github.com/tomMoral/benchmark_denoisers"

    min_benchmark_version = "1.6.1"

    requirements = ["pip::deepinv"]

    parameters = {
        'task': ["denoising", "blur"],
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.6"

    def set_data(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        assert images.ndim == 4, "Images data should be 4D"

        self.n_images = len(images)
        if self.n_images == 1:
            self.images = self.images_test = images
        else:
            self.images = images[:1]
            self.images_test = images[1:]

        if self.task == "denoising":
            self.physic = dinv.physics.Denoising(sigma=0.1)
        elif self.task == "blur":
            kernel_torch = load_degradation(
                "Levin09.npy", get_data_path() / "kernels", index=1
            )
            self.physic = dinv.physics.BlurFFT(
                img_size=self.images.shape[1:],
                filter=kernel_torch,
                noise_model=dinv.physics.GaussianNoise(sigma=0.1),
            )

    def evaluate_result(self, denoiser):
        noise_levels = torch.logspace(-2, 0, 5)

        res = []
        psnr, ssim = PSNR(), SSIM()
        for sigma in noise_levels:
            noisy_images = self.images_test + sigma * torch.randn_like(
                self.images_test
            )
            denoised_images = denoiser(noisy_images, sigma.item())

            psnr_img = psnr(denoised_images, self.images_test).detach().numpy()
            ssim_img = ssim(denoised_images, self.images_test).detach().numpy()
            res.extend([
                dict(PSNR=psnr_i, SSIM=ssim_i, sigma=sigma.item(), id_img=i)
                for i, (psnr_i, ssim_i) in enumerate(zip(psnr_img, ssim_img))
            ])

        return res

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(denoiser=lambda x, _: x)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            images=self.images,
        )

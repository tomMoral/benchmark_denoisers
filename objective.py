from benchopt import BaseObjective, safe_import_context
from benchopt.config import get_data_path
from time import perf_counter

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    import deepinv as dinv
    from deepinv.loss.metric import PSNR, SSIM
    from deepinv.utils.demo import load_degradation
    from deepinv.optim import optim_builder


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
        'sigma': [1e-3, 1e-2, 1e-1, 1]
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
            self.physics = dinv.physics.Denoising(sigma=self.sigma)
            self.algo = lambda denoiser: (lambda y, _: denoiser(y, self.sigma))
        elif self.task == "blur":
            kernel_torch = load_degradation(
                "Levin09.npy", get_data_path() / "kernels", index=1
            ).unsqueeze(0).unsqueeze(0)
            self.physics = dinv.physics.BlurFFT(
                img_size=self.images.shape[1:],
                filter=kernel_torch,
                noise_model=dinv.physics.GaussianNoise(sigma=self.sigma),
            )

            L = self.physics.compute_norm(
                torch.randn_like(self.images), tol=1e-4
            )
            step_size, lmbd = 1 / L, self.sigma * L

            def algo(denoiser):
                prior = dinv.optim.PnP(denoiser)
                algo = optim_builder(
                    "HQS", prior=prior, data_fidelity=dinv.optim.L2(),
                    max_iter=100,
                    params_algo=dict(step_size=step_size, g_param=lmbd)
                )
                algo.eval()
                return algo

            self.algo = algo
        else:
            raise ValueError(f"Unknown task {self.task}")

        if torch.cuda.is_available():
            self.images = self.images.cuda()
            self.images_test = self.images_test.cuda()
            self.physics = self.physics.cuda()

    def evaluate_result(self, denoiser):

        y = self.physics(self.images_test)

        restoration = self.algo(denoiser)

        with torch.no_grad():
            t_start = perf_counter()
            denoised_images = restoration(y, self.physics)
            runtime = perf_counter() - t_start

        psnr_img = PSNR()(denoised_images, self.images_test)
        psnr_img = psnr_img.detach().cpu().numpy()
        ssim_img = SSIM()(denoised_images, self.images_test)
        ssim_img = ssim_img.detach().cpu().numpy()
        return [
            {
                'PSNR': psnr_i,
                'SSIM': ssim_i,
                'id_img': i,
                'runtime': runtime,
            } for i, (psnr_i, ssim_i) in enumerate(zip(psnr_img, ssim_img))
        ]

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

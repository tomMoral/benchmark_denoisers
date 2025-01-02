from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import timm  # noqa: F401
    from deepinv.models import SwinIR


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'SwinIR'
    requirements = ["pip::timm"]

    parameters = {
        'pretrained': ["download", None]
    }

    sampling_strategy = 'run_once'

    def set_objective(self, images):
        self.images = images

        self.denoiser = SwinIR(
            pretrained=self.pretrained, device=images.device
        )

    def run(self, _):
        pass

    def get_result(self):
        return dict(denoiser=self.denoiser)

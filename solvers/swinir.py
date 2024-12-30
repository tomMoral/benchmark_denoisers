from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from deepinv.models import SwinIR


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'SwinIR'

    parameters = {
        'pretrained': ["download", None]
    }

    sampling_strategy = 'run_once'

    def set_objective(self, images):
        self.images = images

        self.denoiser = SwinIR(pretrained=self.pretrained)

    def run(self, _):
        pass

    def get_result(self):
        return dict(denoiser=self.denoiser)

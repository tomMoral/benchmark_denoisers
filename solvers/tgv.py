from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    # Make it possible to check if pywt is installed
    import pywt  # noqa: F401
    from deepinv.models import TGVDenoiser


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'TGV'

    sampling_strategy = 'run_once'

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = ["pip::PyWavelets"]

    def set_objective(self, images):
        self.images = images

        self.denoiser = TGVDenoiser()

    def run(self, _):
        pass

    def get_result(self):
        return dict(denoiser=self.denoiser)

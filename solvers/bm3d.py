from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from deepinv.models import BM3D


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'BM3D'

    sampling_strategy = 'run_once'

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = ["pip::bm3d"]

    def set_objective(self, images):
        self.images = images

        self.denoiser = BM3D()

    def run(self, _):
        pass

    def get_result(self):
        return dict(denoiser=self.denoiser)

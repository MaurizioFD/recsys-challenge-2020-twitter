from abc import abstractmethod
from abc import ABC
import scipy.sparse as sps
import RootPath
import pathlib as pl


class CSR_SparseMatrix(ABC):
    ROOT_PATH = RootPath.get_root().joinpath("Dataset/Sparse/")

    def __init__(self, matrix_name: str):
        self.matrix_name = matrix_name
        self.path = pl.Path(f"{CSR_SparseMatrix.ROOT_PATH}/sparse/{self.matrix_name}.npz")
        self.sim_path = pl.Path(f"{CSR_SparseMatrix.ROOT_PATH}/sparse/{self.matrix_name}_sim.npz")

    def has_matrix(self):
        return self.path.is_file()

    def load_matrix(self):
        assert self.has_matrix(), f"The Matrix {self.matrix_name} does not exists. Create it first."
        matrix = sps.load_npz(self.path)
        return matrix

    @abstractmethod
    def create_matrix(self):
        pass

    def load_or_create(self):
        if not self.has_matrix():
            self.create_matrix()
        return self.load_matrix()

    def save_matrix(self, matrix):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sps.save_npz(self.path, matrix)

    def load_similarity(self):
        if self.path.is_file():
            matrix = sps.load_npz(self.sim_path)
            return matrix
        else:
            raise Exception("File not found. Generate the similarity first")

    def save_similarity(self, sim_matrix):
        self.sim_path.parent.mkdir(parents=True, exist_ok=True)
        sps.save_npz(self.sim_path, sim_matrix)
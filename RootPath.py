import pathlib as pl


def get_root():
    return pl.Path(__file__).parent.absolute()


def get_dataset_path():
    return pl.Path(__file__).parent.joinpath("Dataset/").absolute()

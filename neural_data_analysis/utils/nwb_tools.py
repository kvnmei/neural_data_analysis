from pathlib import Path


def get_nwb_files(path: Path) -> list[Path]:
    """
    Get all NWB files in a dataset directory downloaded from DANDI.
    Assumes that the directory contains directories for each subject and the NWB files
    are in the subject directories. Each NWB file in a subject directory should be a separate session
    for the same subject.

    Args:
        path (Path): path to DANDI dataset directory containing subject directories

    Returns:
        nwb_sessin_files (list[Path]): list of NWB file names
    """
    if not isinstance(path, Path):
        path = Path(path)
    nwb_session_files = sorted(path.glob("sub-*/*.nwb"))
    return nwb_session_files


if __name__ == "__main__":
    get_nwb_files(Path("/home/kevinmei/Projects/Bang-Youre-Dead-Movie/data/000623"))

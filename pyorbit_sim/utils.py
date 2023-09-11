import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time

import orbit_mpi


def ensure_path_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def delete_files_not_folders(path):
    for root, folders, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))


def git_url():
    url = None
    try:
        url = (
            subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            .decode("utf-8")
            .strip()
        )
        if url and url.endswith(".git"):
            url = url[:-4]
    except subprocess.CalledProcessError:
        pass
    return url


def is_git_clean():
    clean = False
    try:
        clean = (
            False if subprocess.check_output(["git", "status", "--porcelain"]) else True
        )
    except subprocess.CalledProcessError:
        pass
    return clean


def git_revision_hash():
    commit = None
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        pass
    return str(commit)


def file_hash(filename):
    sha1_hash = hashlib.sha1()
    with open(filename, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha1_hash.update(byte_block)
    return sha1_hash.hexdigest()


def url_style(url, text=None):
    if text is None:
        text = url
    return {"text": text, "url": url}


class ScriptManager:
    def __init__(self, outdir=None, filepath=None):
        """
        Parameters
        ----------
        outdir : str
            All output files/folders will go here.
        filepath : str
            Full path to script.
        """
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        
        main_rank = 0
        datestamp = time.strftime("%Y-%m-%d")
        timestamp = time.strftime("%y%m%d%H%M%S")
        self.datestamp = orbit_mpi.MPI_Bcast(datestamp, orbit_mpi.mpi_datatype.MPI_CHAR, main_rank, _mpi_comm)
        self.timestamp = orbit_mpi.MPI_Bcast(timestamp, orbit_mpi.mpi_datatype.MPI_CHAR, main_rank, _mpi_comm)
        
        self.git_hash, self.git_url = self.get_git()
        
        self.filepath = filepath
        self.path = pathlib.Path(self.filepath)
        self.script_name = self.path.stem
        self.outdir = os.path.join(
            outdir, 
            self.path.as_posix().split("scripts/")[1].split(".py")[0], 
            self.timestamp,
        )
    
    def make_dirs(self):
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def get_git(self):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        _git_hash = git_revision_hash()
        _git_url = "{}/commit/{}".format(git_url(), _git_hash)
        if _mpi_rank == 0:
            if _git_hash and git_url and is_git_clean():
                print("Repository is clean.")
                print("Code should be available at {}".format(_git_url))
            else:
                print("Unknown git revision.")
        return _git_hash, _git_url
    
    def get_filename(self, filename):
        return os.path.join(self.outdir, filename)
    
    def save_script_copy(self):
        filename = self.get_filename("{}.py".format(self.script_name))
        shutil.copy(self.path.absolute().as_posix(), filename)
        
    def get_info(self):
        info = {
            "git_hash": self.git_hash,
            "git_url": self.git_url,
            "outdir": self.outdir,
            "script_name": self.script_name,
            "timestamp": self.timestamp,
            "datestamp": self.datestamp,
        }
        return info
    
    def get_logger(self, save=True, disp=True, filename="log.txt"):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if save:
            filename = self.get_filename(filename)
            info_file_handler = logging.FileHandler(filename, mode="a")
            info_file_handler.setLevel(logging.INFO)
            logger.addHandler(info_file_handler)
        if disp:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
        return logger
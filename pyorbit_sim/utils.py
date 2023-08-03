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
    
    def __init__(self, datadir=None, path=None, timestamp=None, datestamp=None, script_path_in_outdir=True):
        """
        Parameters
        ----------
        datadir : str
            All output files/folders will go here.
        path : pathlib.Path
            The path to the script.
        timestamp : str
            Timestamp when ScriptManager was created (YYMMDDHHSSMM). This is
            appended to each output file, along with the script name. For example,
            "230812062403-sim_coords.py" if the script name is 'sim'.
        datestamp : str
            Datestamp when ScriptManager was created (YYYY-MM-DD).
        script_path_in_outdir : bool
            Suppose the script is '/scripts/ring/sim.py` and datadir is
            '/home/sim_data/' and timestamp is '230812062403':
            
            True: outdir is '/home/sim_data/ring/sim/230812062403/'.
            False: outdir is '/home/sim_data/'.
        """
        self.datadir = datadir
        self.timestamp = timestamp
        self.datestamp = datestamp
        self.path = path
        self.script_name = self.path.stem
        self.git_hash, self.git_url = self.get_git()
        self.prefix = "{}-{}".format(self.timestamp, self.script_name)
        self.script_path_in_outdir = script_path_in_outdir
        if self.script_path_in_outdir:
            self.outdir = os.path.join(
                self.datadir, 
                self.path.as_posix().split("scripts/")[1].split(".py")[0], 
                self.datestamp,
            )
        else:
            self.outdir = self.datadir
    
    def make_outdir(self):
        """Create output directory."""
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def get_git(self):
        """Retreive git information."""
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
    
    def get_filename(self, filename, sep="_"):
        """Append prefix to filename.
        
        Example: 'data.txt' --> '230812062403-sim_data.txt'.
        """
        return os.path.join(self.outdir, "{}{}{}".format(self.prefix, sep, filename))
    
    def save_script_copy(self):
        """Save a timestamped copy of the script"""
        shutil.copy(self.path.absolute().as_posix(), self.get_filename(".py", sep=""))
        
    def get_info(self):
        """Return info dictionary."""
        info = {
            "git_hash": self.git_hash,
            "git_url": self.git_url,
            "outdir": self.outdir,
            "script_name": self.script_name,
            "timestamp": self.timestamp,
            "datestamp": self.datestamp,
        }
        return info

    def save_info(self):   
        """Save info dictionary."""
        info = self.get_info()
        file = open(self.get_filename("info.txt"), "w")
        for key in sorted(info):
            file.write("{}: {}\n".format(key, info[key]))
        file.close()
        return info
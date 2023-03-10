import os
import pathlib
import shutil
import subprocess
import sys
import time


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
    
    def __init__(self, datadir=None, path=None):
        self.datadir = datadir
        self.path = path
        self.script_name = self.path.stem
        self.datestamp = time.strftime("%Y-%m-%d")
        self.timestamp = time.strftime("%y%m%d%H%M%S")
        self.git_hash, self.git_url = self.get_git()
        self.prefix = "{}-{}".format(self.timestamp, self.script_name)
        self.outdir = os.path.join(
            self.datadir, 
            self.path.as_posix().split("scripts/")[1].split(".py")[0], 
            self.datestamp,
        )
        ensure_path_exists(self.outdir)
        print("Output directory: {}".format(self.outdir))
        print("Output file prefix: {}".format(self.prefix))
                        
    def get_git(self):
        _git_hash = git_revision_hash()
        _git_url = "{}/commit/{}".format(git_url(), _git_hash)
        if _git_hash and git_url and is_git_clean():
            print("Repository is clean.")
            print("Code should be available at {}".format(_git_url))
        else:
            print("Unknown git revision.")
        return _git_hash, _git_url
    
    def get_filename(self, filename, sep="_"):
        return os.path.join(self.outdir, "{}{}{}".format(self.prefix, sep, filename))
    
    def save_script_copy(self):
        shutil.copy(self.path.absolute().as_posix(), self.get_filename(".py", sep=""))
        
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

    def save_info(self):      
        info = self.get_info()
        file = open(self.get_filename("info.txt"), "w")
        for key in sorted(info):
            file.write("{}: {}\n".format(key, info[key]))
        file.close()
        return info
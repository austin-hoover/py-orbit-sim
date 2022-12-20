import os
import sys
import subprocess


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

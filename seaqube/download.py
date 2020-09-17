import urllib.request
from os.path import join, isfile, isdir, exists
from os import mkdir, system
from tqdm import tqdm

import gzip
import shutil

from seaqube.package_config import package_path, log
import subprocess

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class ExternalDownload:
    def __call__(self, what):
        if what == "fasttext-en-pretrained":
            self.__download_fasttext_en_pretrained()

        elif what == "spacy-en-pretrained":
            self.__download_spacy_en_pretrained()

        else:
            raise ValueError(f"The download you want to perform is not implemented (what={what})")

    def __download_url(self, url, path):
        """
        Based on https://stackoverflow.com/a/44712152. It is used for interactive downloads.
        Args:
            url: download url
            path: path to store on disk

        Returns: None

        """
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=path, reporthook=t.update_to)

    def __download_spacy_en_pretrained(self):
        #subprocess.check_output('python -m spacy download en_core_web_sm', shell=True, universal_newlines=True)
        system('python -m spacy download en_core_web_sm')

    def __download_fasttext_en_pretrained(self):
        lang = "en"
        ft_dir = join(package_path, 'augmentation', 'data', 'fasttext_en')
        if not exists(ft_dir):
            mkdir(ft_dir)

        gz_path = join(package_path, 'augmentation', 'data', 'fasttext_en', f'cc.{lang}.300.bin.gz')
        bin_path = join(package_path, 'augmentation', 'data', 'fasttext_en', f'cc.{lang}.300.bin')

        log.info(f"Download: {gz_path}")
        self.__download_url(f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.bin.gz", gz_path)

        with gzip.open(gz_path, 'rb') as f_in:
            with open(bin_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

downloader = ExternalDownload()

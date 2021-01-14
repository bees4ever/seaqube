"""
Copyright (c) 2021 by Benjamin Manns
This file is part of the Semantic Quality Benchmark for Word Embeddings Tool in Python (SeaQuBe).
:author: Benjamin Manns
"""

import tempfile
import urllib.request
from os.path import join, isfile, isdir, exists
from os import mkdir, system
from tqdm import tqdm
import gzip
import shutil
from seaqube.package_config import package_path, log


class DownloadProgressBar(tqdm):
    """
    A simple tqdm based progress bar of downloading data.
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class ExternalDownload:
    """
    Makes it possible to download external data with a one-line. Some data are not pre-installed because of their size.
    This classe implements all easy-downloadable data like external packages or pre-trained models.
    """
    def __call__(self, what):
        if what == "fasttext-en-pretrained":
            self.__download_fasttext_en_pretrained()

        elif what == "spacy-en-pretrained":
            self.__download_spacy_en_pretrained()
        
        elif what == "vec4ir":
            self.__install_vec4ir()

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

    def __install_vec4ir(self):
        tmp = tempfile.mkdtemp()
        system('cd ' + tmp + ' && git clone https://github.com/bees4ever/vec4ir.git && cd vec4ir/ && pip install -e .')

    def __download_fasttext_en_pretrained(self):
        lang = "en"
        
        data_dir = join(package_path, 'augmentation', 'data')
        if not exists(data_dir):
            mkdir(data_dir)

        ft_dir = join(data_dir, 'fasttext_en')
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

Installation (Development)
==========================

Download and install NLTK from http://nltk.org/install.html if you do not already have it.

Move `masc.py` into ``your-path-to-nltk/corpus/reader/``

In ``nltk/corpus/__init__.py`` add this code after the import statements,

::

    masc = LazyCorpusLoader('oanc_masc', MascCorpusReader, r'(?!\.).*\.txt', encoding='utf-8')


In ``nltk/corpus/reader/__init__.py`` add this line,

::

    from nltk.corpus.reader.masc import *

and add ``'MascCorpusReader'`` (with single quotes) to the list of Corpus readers in the ``__all__ =  [`` statement.

Make sure you have the ``oanc_masc`` zip file and/or folder within your ``nltk_data/corpora`` directory.

The NLTK data folder is probably in your home folder. Inside of the MASC folder be the written and spoken directories.

You can download MASC 3.0.0 from http://anc.org but you need to rename the directory to ``oanc_masc`` and move it to ``nltk_data/corpora``.

Usage
=====

Open a python shell, such as IDLE,

::

    import nltk

To use the MascCorpusReader type:

::

    nltk.corpus.masc

For example, to see the list of files in the MASC directory type

::

    nltk.corpus.masc.fileids()

For more example usage see http://nltk.sourceforge.net/corpus.html

# Natural Language Toolkit: ISO GrAF Corpus Reader
#
# Copyright (C) 2001-2014 NLTK Project
# Author: Stephen Matysik <smatysik@gmail.com> (original)
#         Jared Suttles <jaredks@gmail.com> (heavily modified)
# URL: <http://nltk.org/>
# For license informations, see LICENSE.TXT

"""
A reader for corpora that consist of documents in
the ISO GrAF format.
"""

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from collections import namedtuple

from nltk.corpus.reader.util import *
from nltk.corpus.reader.api import *


class MascCorpusReader(CorpusReader):
    """
    Reader for corpora that consist of documents from MASC collection.
    Paragraphs, sentences, words, nouns, verbs, and other annotations
    are contained within MASC.
    """
    def __init__(self, root, fileids=r'(?!\.).*', encoding='utf-8'):
        """
        Construct a new MASC corpus reader for a set of documents
        located at the given root directory.  Example usage:

            >>> root = '/...path to corpus.../'
            >>> reader = MascCorpusReader(root)

        :param root: The root directory for this corpus.
        :param fileids: A list of regexp specifying the fileids in
            this corpus.
        :param encoding: The encoding used for the text files in the corpus.
        """
        CorpusReader.__init__(self, root, fileids, encoding)
        self._fileids = [f for f in self._fileids
                         if 'data' in f and f.endswith('.txt')]

    def raw(self, fileids=None):
        """
        :return: the given file(s) as a single string.
        :rtype: str
        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, basestring):
            fileids = [fileids]
        return concat([self.open(f).read() for f in fileids])

    def words(self, fileids=None):
        """
        :return: the given file(s) as a list of words
            and punctuation symbols.
        :rtype: list(str)
        """
        return concat([MascCorpusView(fileid, 'penn', 'tok')
                       for fileid in self.abspaths(fileids)])

    def sents(self, fileids=None):
        """
        :return: the given file(s) as a list of
            sentences or utterances, each encoded as a list of word
            strings.
        :rtype: list(list(str))
        """
        return concat([MascCorpusView(fileid, 's', ('s', 'head'))
                       for fileid in self.abspaths(fileids)])

    def paras(self, fileids=None):
        """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as lists of word strings.
        :rtype: list(list(list(str)))
        """
        return concat([MascCorpusView(fileid, 'logical', 'p')
                       for fileid in self.abspaths(fileids)])

    def tagged_words(self, fileids=None):
        """
        :return: the given file(s) as a list of tagged
            words and punctuation symbols, encoded as tuples
            ``(word,tag)``.
        :rtype: list(tuple(str,str))
        """
        return concat([MascCorpusView(fileid, 'penn', 'tok', pos=True)
                       for fileid in self.abspaths(fileids)])

    def tagged_sents(self, fileids=None):
        """
        :return: the given file(s) as a list of
            sentences, each encoded as a list of ``(word,tag)`` tuples.
        :rtype: list(list(tuple(str,str)))
        """
        return concat([MascCorpusView(fileid, 's', ('s', 'head'), pos=True)
                       for fileid in self.abspaths(fileids)])

    def tagged_paras(self, fileids=None):
        """
        :return: the given file(s) as a list of
            paragraphs, each encoded as a list of sentences, which are
            in turn encoded as lists of ``(word,tag)`` tuples.
        :rtype: list(list(list(tuple(str,str))))
        """
        return concat([MascCorpusView(fileid, 'logical', 'p', pos=True)
                       for fileid in self.abspaths(fileids)])

    def nouns(self, fileids=None):
        """
        :return: the given file(s) as a list of noun chunks.
        :rtype: list(str)
        """
        return concat([MascCorpusView(fileid, 'nc', 'nchunk')
                       for fileid in self.abspaths(fileids)])

    def verbs(self, fileids=None):
        """
        :return: the given file(s) as a list of verb chunks.
        :rtype: list(str)
        """
        return concat([MascCorpusView(fileid, 'vc', 'vchunk')
                       for fileid in self.abspaths(fileids)])

    def persons(self, fileids=None):
        """
        :return: the given file(s) as a list of persons.
        :rtype: list(str)
        """
        return concat([MascCorpusView(fileid, 'ne', 'person')
                       for fileid in self.abspaths(fileids)])

    def locations(self, fileids=None):
        """
        :return: the given file(s) as a list of locations.
        :rtype: list(str)
        """
        return concat([MascCorpusView(fileid, 'ne', 'location')
                       for fileid in self.abspaths(fileids)])

    def dates(self, fileids=None):
        """
        :return: the given file(s) as a list of dates.
        :rtype: list(str)
        """
        return concat([MascCorpusView(fileid, 'ne', 'date')
                       for fileid in self.abspaths(fileids)])

    def orgs(self, fileids=None):
        """
        :return: the given file(s) as a list of organizations.
        :rtype: list(str)
        """
        return concat([MascCorpusView(fileid, 'ne', 'org')
                       for fileid in self.abspaths(fileids)])


class MascCorpusView(StreamBackedCorpusView):
    _block_size = 500

    def __init__(self, fileid, annotation, labels, pos=False):
        StreamBackedCorpusView.__init__(self, fileid, encoding='utf-8')
        self._path = fileid[:-4]
        self._type = annotation
        self._labels = (labels,) if isinstance(labels, basestring) else labels
        self._pos = pos
        self._saved_text = ''

    def read_block(self, stream):
        """
        :param stream: A file stream
        :return: A block of annotated text.
        """
        byte_start = stream.tell()
        char_start = self._byte_to_char.get(byte_start, byte_start)
        char_end = char_start + self._block_size
        char_start -= len(self._saved_text)
        subset, overlap, char_end = self._get_subset(char_start, char_end)
        byte_end = self._char_to_byte.get(char_end, char_end)
        read_size = byte_end - byte_start
        if read_size <= 0:
            read_size = self._block_size
        text = self._saved_text + stream.read(read_size)
        block = self._block_reader(subset, text, char_start)
        self._saved_text = text[-overlap:]
        return block

    def _open(self):
        """
        Overrides superclass method for opening the file stream. Does
        any work that only needs to be done once for the instance:
            1. Parses XMLs into one or more _Stream of _MascAnnotation
            2. Generates the discrepency dicts
        """
        StreamBackedCorpusView._open(self)
        self._annotations = self._get_annotations(self._type, self._labels)
        self._block_reader = self._read_block
        if self._type == 'logical':
            self._block_reader = self._read_para_block
            self._annotations_sents = self._get_annotations('s', ('s', 'head'))
            self._annotations_words = self._get_annotations('penn', 'tok')
        elif self._type == 's':
            self._block_reader = self._read_sent_block
            self._annotations_words = self._get_annotations('penn', 'tok')
        self._get_disc(self._stream)

    def _get_annotations(self, annotation_type, labels):
        """
        Calls _annotations to generate ``(offsets, pos)`` tuples then
        maps _MascAnnotation onto each, wraps as a _Stream object, and
        sorts based on the offsets for internal use.
        """
        annotations = _annotations(self._path, annotation_type, labels)
        annotations = _Stream(_MascAnnotation(*a) for a in annotations)
        annotations.sort(key=lambda x: x.offsets)
        return annotations

    def _get_disc(self, stream):
        """
        Uses the file stream to create two discrepency mappings,
        both as dictionaries:
            1. self._char_to_byte uses key = character number,
                                       entry = byte number
            2. self._byte_to_char uses key = byte number,
                                       entry = character number
        """
        self._char_to_byte = {}
        self._byte_to_char = {}
        stream.read()
        end = stream.tell()
        stream.seek(0)
        for i in range(end + 1):
            if i != stream.tell():
                self._char_to_byte[i] = stream.tell()
                self._byte_to_char[stream.tell()] = i
            stream.read(1)
        stream.seek(0)

    def _get_subset(self, adjusted_char_start, char_end):
        """
        Find the subset of offsets that are within our search area. Also will
        produce the final character end offset to help us know how much to read.
        :return: A list of annotations, overlap size, and last character index.
        """
        subset = []
        highest_curr = lowest_next = None
        for annotation in self._annotations:
            start, end = annotation.offsets
            if adjusted_char_start <= start:
                highest_curr = max(highest_curr, end)
                subset.append(annotation)
                if end > char_end:
                    try:
                        lowest_next, _ = next(iter(self._annotations)).offsets
                        self._annotations.back(1)
                        break
                    except StopIteration:
                        pass
        if lowest_next is None or lowest_next > highest_curr:
            overlap_size = 0
        else:
            overlap_size = highest_curr - lowest_next
        return subset, overlap_size, highest_curr or char_end

    def _get_word(self, text, offsets, adjust):
        """
        :return: string of text between given offsets, accounting for
            adjust, without newline characters and multiple spaces.
        :rtype: str
        """
        start, end = offsets
        w = text[start-adjust:end-adjust]
        return ' '.join(w.encode('utf-8').translate(None, '\n').split())

    def _read_block(self, subset, text, adjust):
        block = []
        for annotation in subset:
            word = self._get_word(text, annotation.offsets, adjust)
            block.append((word, annotation.pos) if self._pos else word)
        return block

    def _read_sent_block(self, subset, text, adjust):
        block = []
        for s_annotation in subset:
            s = s_annotation.offsets
            sent = []
            for w_annotation in self._annotations_words:
                w = w_annotation.offsets
                if s[0] <= w[0] and w[1] <= s[1]:
                    word = self._get_word(text, w, adjust)
                    sent.append((word, w_annotation.pos) if self._pos else word)
                elif s[1] < w[1]:
                    break
            self._annotations_words.back(len(sent) + 1)
            block.append(sent)
        return block

    def _read_para_block(self, subset, text, adjust):
        block = []
        for p_annotation in subset:
            p = p_annotation.offsets
            para = []
            for s_annotation in self._annotations_sents:
                s = s_annotation.offsets
                if p[0] <= s[0] and s[1] <= p[1]:
                    sent = []
                    for w_annotation in self._annotations_words:
                        w = w_annotation.offsets
                        if s[0] <= w[0] and w[1] <= s[1]:
                            word = self._get_word(text, w, adjust)
                            sent.append(
                                (word, w_annotation.pos) if self._pos else word
                            )
                        elif s[1] < w[1]:
                            break
                    self._annotations_words.back(len(sent) + 1)
                    para.append(sent)
                elif p[1] < s[1]:
                    break
            self._annotations_words.back(sum(len(sent) for sent in para) + 1)
            self._annotations_sents.back(len(para) + 1)
            if para:
                block.append(para)
        return block


class _Stream(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)
        self.position = 0

    def __iter__(self):
        for i in range(self.position, len(self)):
            self.position = i + 1
            yield self[i]

    def back(self, n):
        self.position = self.position - n if n < self.position else 0


_MascAnnotation = namedtuple('Annotation', ['offsets', 'pos'])


def _get_offsets(node):
    """
    Offsets to be used to index text for the given type and id.
    """
    offsets = []
    if node.links:
        for link in node.links:
            for region in link:
                offsets.extend(region.anchors)
        return min(offsets), max(offsets)
    elif node.out_edges:
        for edge in node.out_edges:
            offset = _get_offsets(edge.to_node)
            if offset is not None:
                offsets.extend(offset)
        return min(offsets), max(offsets)
    return None


def _parse_xmls(path, annotation_type):
    try:
        import graf
    except ImportError:
        msg = ("graf-python is required to parse MASC xml files.\n\n"
               "You can download using the pip command (if you have pip),\n\n"
               "    pip install graf-python\n\n"
               "More information is available at "
               "http://media.cidles.eu/poio/graf-python/\n")
        raise ImportError(msg)
    parser = graf.GraphParser()
    return parser.parse(path + '-{0}.xml'.format(annotation_type))


def _annotations(path, annotation_type, labels):
    """
    Given a base name (file path up to an extension), an annotation type
    (ie. penn, seg, s,...) and a labels iterable, yield tuples of
    ``(offsets, pos)`` with valid offsets and label.
    """
    parsed_xmls = _parse_xmls(path, annotation_type)
    for node in parsed_xmls.nodes:
        annotation = next(iter(node.annotations))
        if annotation.label in labels:
            offsets = _get_offsets(node)
            if offsets is not None and offsets[0] < offsets[1]:
                yield offsets, annotation.features.get('msd')

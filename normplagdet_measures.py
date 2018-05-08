#! /usr/bin/python
# Copyright (C) 2009 webis.de. All rights reserved.
"""Plagiarism detection performance measures.

This module implements the measures recall, precision, and granularity
as described by the authors of [1]. The measures can be calculated
macro-averaged and micro-averaged with the respective functions; each
one's parameters are iterables of reference plagiarism cases and
plagiarism detections. The latter are compared with the former in order
to determine how accurate the cases have been detected with respect to
different ideas of detection accuracy. Moreover, the function
plagdet_score combines recall, precision and granularity values to a
single value which allows to rank plagiarism detection algorithms.

The parameters 'cases' and 'detections' both must contain instances of
the Annotation class, a 7-tuple consisting of the following exemplified
values:
>>> Annotation('suspicious-document00001.txt', 10000, 1000, \
...            'source-document00001.txt', 5000, 1100, True)
where the first three values reference a section of text in a
suspicious document by means of char offset and length, and likewise
the following three values reference a section of text in a source
document. The last value specifies whether the annotation is to be
treated as an external detection or as an intrinsic detection. In the
latter case, the precedin values should be set to '', 0, 0, respectively.

Finally, this module contains functions to extract plagiarism
annotations from XML documents which contain tags with a given name
attribute, and with values corresponding to those required by the
Annotation class, e.g., the XML format used in the PAN benchmarking
workshops [2,3].

[1]  Martin Potthast, Benno Stein, Alberto Barron-Cedeno, and Paolo Rosso.
     An Evaluation Framework for Plagiarism Detection.
     In Proceedings of the 23rd International Conference on Computational
     Linguistics (COLING 2010), Beijing, China. August 2010. ACL.

[2]  Martin Potthast, Benno Stein, Andreas Eiselt, Alberto Barron-Cedeno,
     and Paolo Rosso. Overview of the 1st International Competition on
     Plagiarism Detection. In Benno Stein, Paolo Rosso, Efstathios
     Stamatatos, Moshe Koppel, and Eneko Agirre, editors, SEPLN 2009
     Workshop on Uncovering Plagiarism, Authorship, and Social Software
     Misuse (PAN 09), pages 1-9, September  2009. CEUR-WS.org.
     ISSN 1613-0073.

[3]  Martin Potthast, Benno Stein, Andreas Eiselt, Alberto Barron-Cedeno,
     and Paolo Rosso. Overview of the 2nd International Benchmarking
     Workshop on Plagiarism Detection. In Benno Stein, Paolo Rosso,
     Efstathios Stamatatos, and Moshe Koppel, editors, Proceedings of
     PAN at CLEF 2010: Uncovering Plagiarism, Authorship, and Social
     Software Misuse, September 2010.
"""



__author__ = "Martin Potthast"
__email__ = "martin.potthast at uni-weimar dot de"
__version__ = "1.3"
__all__ = ["macro_avg_recall_and_precision", "micro_avg_recall_and_precision",
           "granularity", "plagdet_score", "Annotation"]


from collections import namedtuple
from functools import reduce
import getopt
import glob
import math
from numpy.ma import zeros, sum as npsum
import os
import sys
import unittest
import xml.dom.minidom


TREF, TOFF, TLEN = 'this_reference', 'this_offset', 'this_length'
SREF, SOFF, SLEN = 'source_reference', 'source_offset', 'source_length'
EXT = 'is_external'
Annotation = namedtuple('Annotation', [TREF, TOFF, TLEN, SREF, SOFF, SLEN, EXT])
TREF, TOFF, TLEN, SREF, SOFF, SLEN, EXT = list(range(7))


class MultiAnnotation:
    def __init__(self, ref):
        self.ref = ref
        self.begin = []
        self.end = []
        self.length = 0

    def __len__(self):
        return reduce((lambda l, i: l + self.end[i] - self.begin[i]), range(self.length), 0)


plag_path_ = None


def macro_avg_recall_and_precision(cases, detections):
    """Returns tuple (rec, prec); the macro-averaged recall and precision of the
       detections in detecting the plagiarism cases"""
    return macro_avg_recall(cases, detections), \
           macro_avg_precision(cases, detections)


def micro_avg_recall_and_precision(cases, detections):
    """Returns tuple (rec, prec); the micro-averaged recall and precision of the
       detections in detecting the plagiarism cases"""
    return micro_avg_recall(cases, detections), \
           micro_avg_precision(cases, detections)


def granularity(cases, detections):
    """Granularity of the detections in detecting the plagiarism cases."""
    if len(detections) == 0:
        return 1
    detections_per_case = list()
    case_index = index_annotations(cases)
    det_index = index_annotations(detections)
    for tref in case_index:
        cases, detections = case_index[tref], det_index.get(tref, False)
        if not detections:  # No detections for document tref.
            continue
        for case in cases:
            num_dets = sum((is_overlapping(case, det) for det in detections))
            detections_per_case.append(num_dets)
    detected_cases = sum((num_dets > 0 for num_dets in detections_per_case))
    if detected_cases == 0:
        return 1
    return sum(detections_per_case) / detected_cases


def plagdet_score(rec, prec, gran):
    """Combines recall, precision, and granularity to a allow for ranking."""
    if (rec == 0 and prec == 0) or prec < 0 or rec < 0 or gran < 1:
        return 0
    return ((2 * rec * prec) / (rec + prec)) / math.log(1 + gran, 2)


def macro_avg_precision(cases, detections):
    """The macro-averaged precision of the detections in detecting the plagiarism cases."""
    # Observe the difference to calling 'macro_avg_recall(cases, detections)'.
    return macro_avg_recall(detections, cases)


def macro_avg_recall(cases, detections, debug=False):
    """Recall of the detections in detecting plagiarism cases."""
    if len(cases) == 0 and len(detections) == 0:
        return 1
    elif len(cases) == 0 or len(detections) == 0:
        return 0
    num_cases, recall_per_case = len(cases), list()
    case_index = index_annotations(cases)
    det_index = index_annotations(detections)
    for tref in case_index:
        cases, detections = case_index[tref], det_index.get(tref, False)
        if not detections:  # No detections for document tref.
            continue
        for case in cases:
            src_len = count_chars_in_doc(case[SREF], False)
            susp_len = count_chars_in_doc(case[TREF], True)
            recall_per_case.append(case_recall(case, detections, src_len, susp_len))
    if debug:
        return sum(recall_per_case) / num_cases, recall_per_case
    else:
        return sum(recall_per_case) / num_cases


def micro_avg_precision(cases, detections):
    """The micro-averaged precision of the detections in detecting the plagiarism cases."""
    return micro_avg_recall(detections, cases)


def micro_avg_recall(cases, detections):
    """The micro-averaged recall of the detections in detecting the plagiarism cases."""
    if len(cases) == 0 and len(detections) == 0:
        return 1
    elif len(cases) == 0 or len(detections) == 0:
        return 0
    cases_detections = true_detections(cases, detections)
    num_summands, denom_summands = [], []
    for ref, off, length in ([TREF, TOFF, TLEN], [SREF, SOFF, SLEN]):
        docs = index_annotations(cases, ref)
        num_summand, denom_summand = 0, 0
        for doc in docs:
            doc_cases = docs[doc]
            doc_detections = list(set(map(lambda case: cases_detections[case], doc_cases)))

            doc_cases_multiannotation = collapse_annotations(doc_cases, doc, off, length)
            doc_detections_multiannotation = collapse_annotations(doc_detections, doc, off, length)
            doc_intersection = intersect_multiannotations(doc_cases_multiannotation, doc_detections_multiannotation)

            doc_length = count_chars_in_doc(doc, False)
            cases_length = len(doc_cases_multiannotation)
            detections_length = len(doc_detections_multiannotation)
            intersection_length = len(doc_intersection)

            a = max(0, cases_length + detections_length - doc_length)
            b = min(cases_length, detections_length)
            q = intersection_length - a
            l = cases_length - a
            w = (b - a) / doc_length

            num_summand += q * w
            denom_summand += l * w
        num_summands.append(num_summand)
        denom_summands.append(denom_summand)
    return (num_summands[0] + num_summands[1]) / (denom_summands[0] + denom_summands[1])


def collapse_annotations(annotations, ref, xoff, xlen):
    """Returns a multi-annotation consisting of annotations in one document."""
    ma = MultiAnnotation(ref)
    for ann in annotations:
        insert_annotation_into_multi(ann, ma, xoff, xlen)
    return ma


def insert_annotation_into_multi(ann, ma, xoff, xlen):
    """Inserts an annotation inside a multi-annotation."""
    begin = 0
    while begin < ma.length and ann[xoff] > ma.end[begin]:
        begin += 1
    end = begin
    while end < ma.length and ann[xoff] + ann[xlen] >= ma.begin[end]:
        end += 1
    new_begin = min(ann[xoff], ma.begin[begin])
    new_end = max(ann[xoff] + ann[xlen], ma.end[end - 1])
    ma.begin = [ma.begin[i] for i in range(begin)] + [new_begin] + [ma.begin[i] for i in range(end, ma.length)]
    ma.end = [ma.end[i] for i in range(begin)] + [new_end] + [ma.end[i] for i in range(end, ma.length)]
    ma.length = len(ma.begin)


def intersect_multiannotations(ma1, ma2):
    """Intersect multi-annotations ma1 and ma2."""
    ma = MultiAnnotation(ma1.ref)
    i1, i2 = 0, 0
    while i1 < ma1.length and i2 < ma2.length:
        if ma1.end[i1] <= ma2.begin[i2]:
            i1 += 1
            continue
        if ma2.end[i2] <= ma1.begin[i1]:
            i2 += 1
            continue
        ma.begin.append(max(ma1.begin[i1], ma2.begin[i2]))
        ma.end.append(min(ma1.end[i1], ma2.end[i2]))
        if ma1.end[i1] < ma2.end[i2]:
            i1 += 1
        else:
            i2 += 1
    return ma


len_cache = {}
def count_chars_in_doc(file_path, is_susp):
    global len_cache
    if (file_path, is_susp) not in len_cache:
        if is_susp:
            file_path_ = plag_path_ + "/../../susp/" + file_path
        else:
            file_path_ = plag_path_ + "/../../src/" + file_path
        len_cache[file_path, is_susp] = len(open(file_path_).read())

    return len_cache[file_path, is_susp]


def case_recall(case, detections, src_len, susp_len):
    """Recall of the detections in detecting the plagiarism case."""

    Rs_detections = [det for det in detections if is_overlapping(case, det)]

    normalized_num = 0.
    normalized_denom = 0.
    for xref, xoff, xlen, dlen in ((TREF, TOFF, TLEN, susp_len), (SREF, SOFF, SLEN, src_len)):
        # Calculate number of chars from the detections that cover the case.
        detections_len = count_chars2(Rs_detections, xref, xoff, xlen)

        # Calculate intersection length between the case and the detections.
        intersection_len = overlapping_chars(case, detections, xoff, xlen)

        # Estimate lower (a) and upper (b) bounds of intersection length.
        a = max(0, detections_len + case[xlen] - dlen)
        b = min(detections_len, case[xlen])

        # Normalize overlap size according to:
        #   1) overlap size lower and upper bounds,
        #   2) document length.
        q = intersection_len - a
        l = case[xlen] - a
        w = (b - a) / dlen

        # Calculate part of case recall.
        normalized_num += q * w
        normalized_denom += l * w

    # Calculate normalized case recall.
    if normalized_denom == 0:
        return 1
    else:
        return normalized_num / normalized_denom


def true_detections(cases, detections):
    """Recreates the detections so that only true detections remain and so that
       the true detections are reduced to the passages that actually overlap
       with the respective detected case."""
    true_dets = dict()
    case_index = index_annotations(cases)
    det_index = index_annotations(detections)
    for tref in case_index:
        cases, detections = case_index[tref], det_index.get(tref, False)
        if not detections:  # No detections for document tref.
            continue
        for case in cases:
            case_dets = (det for det in detections if is_overlapping(case, det))
            true_dets.setdefault(case, []).extend(case_dets)
    return true_dets


def overlap_annotation(ann1, ann2):
    """Returns an Annotation that annotates overlaps between ann1 and ann2."""
    tref, sref, ext = ann1[TREF], ann1[SREF], ann1[EXT] and ann2[EXT]
    toff, tlen, soff, slen = 0, 0, 0, 0
    if is_overlapping(ann1, ann2):
       toff, tlen = overlap_chars(ann1, ann2, TOFF, TLEN)
       if ext:
           soff, slen = overlap_chars(ann1, ann2, SOFF, SLEN)
    return Annotation(tref, toff, tlen, sref, soff, slen, ext)


def overlap_chars(ann1, ann2, xoff, xlen):
    """Returns the overlapping passage between ann1 and ann2, given the keys
       xoff and xlen."""
    overlap_start, overlap_length = 0, 0
    max_ann = ann1 if ann1[xoff] >= ann2[xoff] else ann2
    min_ann = ann1 if ann1[xoff] < ann2[xoff] else ann2
    if min_ann[xoff] + min_ann[xlen] > max_ann[xoff]:
       overlap_start = max_ann[xoff]
       overlap_end = min(min_ann[xoff] + min_ann[xlen], \
                         max_ann[xoff] + max_ann[xlen])
       overlap_length = overlap_end - overlap_start
    return overlap_start, overlap_length


def count_chars(annotations):
    """Returns the number of chars covered by the annotations, while counting
       overlapping chars only once."""
    num_chars = count_chars2(annotations, TREF, TOFF, TLEN)
    num_chars += count_chars2(annotations, SREF, SOFF, SLEN)
    return num_chars


def count_chars2(annotations, xref, xoff, xlen):
    """Returns the number of chars covered by the annotations with regard to
       the keys xref, xoff, and xlen."""
    num_chars = 0
    if len(annotations) == 0:
        return 0
    max_length = max((ann[xoff] + ann[xlen] for ann in annotations))
    char_bits = zeros(max_length, dtype=bool)
    xref_index = index_annotations(annotations, xref)
    for xref in xref_index:
        annotations = xref_index[xref]
        char_bits[:] = False
        for ann in annotations:
            char_bits[ann[xoff]:ann[xoff] + ann[xlen]] = True
        num_chars += npsum(char_bits)
    return num_chars


def overlapping_chars(ann1, annotations, xoff, xlen):
    """Returns the number of chars in ann1 that overlap with the annotations."""
    annotations = [ann2 for ann2 in annotations if is_overlapping(ann1, ann2)]
    if len(annotations) == 0 or not isinstance(ann1, Annotation):
        return 0
    overlaps = zeros(ann1[xlen], dtype=bool)
    for ann2 in annotations:
        mark_overlapping_chars(overlaps, ann1, ann2, xoff, xlen)
    return npsum(overlaps)


def mark_overlapping_chars(char_bits, ann1, ann2, xoff, xlen):
    """Sets the i-th boolean in char_bits to true if ann2 overlaps with the i-th
       char in ann1, respecting the given xoff and xlen index."""
    offset_difference = ann2[xoff] - ann1[xoff]
    overlap_start = min(max(0, offset_difference), ann1[xlen])
    overlap_end = min(max(0, offset_difference + ann2[xlen]), ann1[xlen])
    char_bits[overlap_start:overlap_end] = True


def is_overlapping(ann1, ann2):
    """Returns true iff the ann2 overlaps with ann1."""
    detected = ann1[TREF] == ann2[TREF] and \
               ann2[TOFF] + ann2[TLEN] > ann1[TOFF] and \
               ann2[TOFF] < ann1[TOFF] + ann1[TLEN]
    if ann1[EXT] == True and ann2[EXT] == True:
        detected = detected and ann1[SREF] == ann2[SREF] and \
                   ann2[SOFF] + ann2[SLEN] > ann1[SOFF] and \
                   ann2[SOFF] < ann1[SOFF] + ann1[SLEN]
    return detected


def index_annotations(annotations, xref=TREF):
    """Returns an inverted index that maps references to annotation lists."""
    index = dict()
    for ann in annotations:
        index.setdefault(ann[xref], []).append(ann)
    return index


def extract_annotations_from_files(path, tagname):
    """Returns a set of plagiarism annotations from XML files below path."""
    if not os.path.exists(path):
        print("Path not accessible:", path)
        sys.exit(2)
    annotations = set()
    xmlfiles = glob.glob(os.path.join(path, '*.xml'))
    xmlfiles.extend(glob.glob(os.path.join(path, os.path.join('*', '*.xml'))))
    for xmlfile in xmlfiles:
        annotations.update(extract_annotations_from_file(xmlfile, tagname))
    return annotations


def extract_annotations_from_file(xmlfile, tagname):
    """Returns a set of plagiarism annotations from an XML file."""
    doc = xml.dom.minidom.parse(xmlfile)
    annotations = set()
    if not doc.documentElement.hasAttribute('reference'):
        return annotations
    t_ref = doc.documentElement.getAttribute('reference')
    for node in doc.documentElement.childNodes:
        if node.nodeType == xml.dom.Node.ELEMENT_NODE and \
           node.hasAttribute('name') and \
           node.getAttribute('name').endswith(tagname):
            ann = extract_annotation_from_node(node, t_ref)
            if ann:
                annotations.add(ann)
    return annotations


def extract_annotation_from_node(xmlnode, t_ref):
    """Returns a plagiarism annotation from an XML feature tag node."""
    if not (xmlnode.hasAttribute('this_offset') and \
            xmlnode.hasAttribute('this_length')):
        return False
    t_off = int(xmlnode.getAttribute('this_offset'))
    t_len = int(xmlnode.getAttribute('this_length'))
    s_ref, s_off, s_len, ext = '', 0, 0, False
    if xmlnode.hasAttribute('source_reference') and \
       xmlnode.hasAttribute('source_offset') and \
       xmlnode.hasAttribute('source_length'):
        s_ref = xmlnode.getAttribute('source_reference')
        s_off = int(xmlnode.getAttribute('source_offset'))
        s_len = int(xmlnode.getAttribute('source_length'))
        ext = True
    return Annotation(t_ref, t_off, t_len, s_ref, s_off, s_len, ext)


class TestPerfMeasures(unittest.TestCase):
    """Unit tests for the plagiarism detection performance measures."""

    ann1 = Annotation('tref1', 0, 100, 'sref1', 0, 100, True)
    ann2 = Annotation('tref1', 0, 100, '', 0, 0, False)
    ann3 = Annotation('tref1', 100, 100, 'sref1', 100, 100, True)
    ann4 = Annotation('tref1', 0, 200, 'sref1', 0, 200, True)
    ann5 = Annotation('tref1', 0, 1, 'sref1', 0, 1, True)
    ann6 = Annotation('tref1', 99, 1, 'sref1', 99, 1, True)
    ann7 = Annotation('tref2', 0, 100, 'sref2', 0, 100, True)
    ann8 = Annotation('tref2', 0, 100, '', 0, 0, False)
    ann9 = Annotation('tref2', 50, 100, 'sref2', 50, 100, True)
    ann10 = Annotation('tref2', 25, 75, 'sref2', 25, 75, True)

    def test_macro_averaged_recall(self):
        self.assertEqual(1, macro_avg_recall([], []))
        self.assertEqual(0, macro_avg_recall(['sth'], []))
        self.assertEqual(0, macro_avg_recall([], ['sth']))
        self.assertEqual(1, macro_avg_recall([self.ann1], [self.ann1]))
        self.assertEqual(1, macro_avg_recall([self.ann2], [self.ann2]))
        self.assertEqual(0.5, macro_avg_recall([self.ann1, self.ann7], \
                                               [self.ann1]))
        self.assertEqual(0.5, macro_avg_recall([self.ann2, self.ann8], \
                                               [self.ann2]))
        self.assertEqual(0, macro_avg_recall([self.ann1], [self.ann7]))
        self.assertEqual(0, macro_avg_recall([self.ann2], [self.ann8]))

    def test_case_recall(self):
        self.assertEqual(0, case_recall(self.ann1, []))
        self.assertEqual(1, case_recall(self.ann1, [self.ann1]))
        self.assertEqual(0.5, case_recall(self.ann1, [self.ann2]))
        self.assertEqual(0, case_recall(self.ann1, [self.ann3]))
        self.assertEqual(1, case_recall(self.ann1, [self.ann4]))
        self.assertEqual(1, case_recall(self.ann1, [self.ann4, self.ann7]))
        self.assertEqual(0, case_recall(self.ann1, [self.ann7, self.ann9]))
        self.assertEqual(0.5, case_recall(self.ann7, [self.ann9]))
        self.assertEqual(0.75, case_recall(self.ann7, [self.ann10]))
        self.assertEqual(0.75, case_recall(self.ann7, [self.ann9, self.ann10]))

    def test_macro_averaged_precision(self):
        self.assertEqual(1, macro_avg_precision([], []))
        self.assertEqual(0, macro_avg_precision(['sth'], []))
        self.assertEqual(0, macro_avg_precision([], ['sth']))
        self.assertEqual(1, macro_avg_precision([self.ann1], [self.ann1]))
        self.assertEqual(1, macro_avg_precision([self.ann2], [self.ann2]))
        self.assertEqual(1, macro_avg_precision([self.ann1, self.ann7], \
                                                [self.ann1]))
        self.assertEqual(1, macro_avg_precision([self.ann2, self.ann8], \
                                                [self.ann2]))
        self.assertEqual(0.5, macro_avg_precision([self.ann1], [self.ann4]))
        self.assertEqual(1, macro_avg_precision([self.ann7], [self.ann10]))
        self.assertEqual(1, macro_avg_precision([self.ann7], [self.ann10]))
        self.assertEqual(0.75, macro_avg_precision([self.ann7], \
                                                   [self.ann9, self.ann10]))
        self.assertEqual(0.25, macro_avg_precision([self.ann1], \
                                                   [self.ann3, self.ann4]))

    def test_granularity(self):
        self.assertEqual(1, granularity([], []))
        self.assertEqual(1, granularity([self.ann1], [self.ann2]))
        self.assertEqual(1, granularity([self.ann1], [self.ann2, self.ann3]))
        self.assertEqual(2, granularity([self.ann1],
                                        [self.ann2, self.ann3, self.ann4]))
        self.assertEqual(1.5, granularity([self.ann1, self.ann3],
                                          [self.ann2, self.ann4]))

    def test_plagdet_score(self):
        self.assertEqual(0, plagdet_score(-1, 0, 0))
        self.assertEqual(0, plagdet_score(0, -1, 0))
        self.assertEqual(0, plagdet_score(0, 0, -1))
        self.assertEqual(0, plagdet_score(0, 0, 1))
        self.assertEqual(0, plagdet_score(0, 1, 1))
        self.assertEqual(0, plagdet_score(1, 0, 1))
        self.assertEqual(1, plagdet_score(1, 1, 1))
        self.assertEqual(2 / 3, plagdet_score(0.5, 1, 1))
        self.assertEqual(2 / 3, plagdet_score(1, 0.5, 1))
        self.assertAlmostEqual(0.63092975, plagdet_score(1, 1, 2))
        self.assertAlmostEqual(0.23659865, plagdet_score(0.25, 0.75, 2))

    def test_is_overlapping(self):
        self.assertTrue(is_overlapping(self.ann1, self.ann2))
        self.assertFalse(is_overlapping(self.ann1, self.ann3))
        self.assertTrue(is_overlapping(self.ann1, self.ann4))
        self.assertFalse(is_overlapping(self.ann1, self.ann7))
        self.assertFalse(is_overlapping(self.ann1, self.ann8))
        self.assertFalse(is_overlapping(self.ann1, self.ann9))
        self.assertFalse(is_overlapping(self.ann1, self.ann10))
        self.assertTrue(is_overlapping(self.ann1, self.ann5))
        self.assertTrue(is_overlapping(self.ann1, self.ann6))

    def test_index_annotations(self):
        index = index_annotations([self.ann1, self.ann7, self.ann2, self.ann8])
        self.assertEqual([self.ann1, self.ann2], index.get('tref1'))
        self.assertEqual([self.ann7, self.ann8], index.get('tref2'))


def usage():
    """Prints command line usage manual."""
    print("""\
Usage: perfmeasures.py [options]

Options:
      --micro      Compute micro-averaged recall and precision,
                   default: macro-averaged recall and precision
  -p, --plag-path  Path to the XML files with plagiarism annotations
      --plag-tag   Tag name suffix of plagiarism annotations,
                   default: 'plagiarism'
  -d, --det-path   Path to the XML files with detection annotations
      --det-tag    Tag name of the detection annotations,
                   default: 'detected-plagiarism'
  -h, --help       Show this message
""")


def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["micro", "plag-path=", "plag-tag=", "det-path=",
                        "det-tag=", "help"]
        opts, _ = getopt.getopt(sys.argv[1:], "p:d:h", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)
    micro_averaged = False
    plag_path, det_path = "undefined", "undefined"
    plag_tag_name, det_tag_name = "plagiarism", "detected-plagiarism"
    for opt, arg in opts:
        if opt in ("--micro"):
            micro_averaged = True
        elif opt in ("-p", "--plag-path"):
            plag_path = arg
        elif opt == "--plag-tag":
            plag_tag_name = arg
        elif opt in ("-d", "--det-path"):
            det_path = arg
        elif opt == "--det-tag":
            det_tag_name = arg
        elif opt in ("-h", "--help"):
            usage()
            sys.exit()
        else:
            assert False, "Unknown option."
    if plag_path == "undefined":
        print("Plagiarism path undefined. Use option -p or --plag-path.")
        sys.exit()
    if det_path == "undefined":
        print("Detections path undefined. Use option -d or --det-path.")
        sys.exit()
    return (micro_averaged, plag_path, plag_tag_name, det_path, det_tag_name)


def main(micro_averaged, plag_path, plag_tag_name, det_path, det_tag_name):
    global plag_path_
    plag_path_ = plag_path
    """Main method of this module."""
    print('Reading', plag_path)
    cases = extract_annotations_from_files(plag_path, plag_tag_name)
    print('Reading', det_path)
    detections = extract_annotations_from_files(det_path, det_tag_name)
    print('Processing... (this may take a while)')
    rec, prec = 0, 0
    if micro_averaged:
        rec, prec = micro_avg_recall_and_precision(cases, detections)
    else:
        rec, prec = macro_avg_recall_and_precision(cases, detections)
    gran = granularity(cases, detections)
    print('Plagdet Score', plagdet_score(rec, prec, gran))
    print('Recall', rec)
    print('Precision', prec)
    print('Granularity', gran)


if __name__ == '__main__':
    main(*parse_options())



import os
import sys
import csv
import nltk
import regex
import pickle
import collections
import numpy as np
import pandas as pd

from lxml import etree
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

assert len(sys.argv) == 5

data_dir = sys.argv[1]
tasks_dir = sys.argv[2]
out_det_dir = sys.argv[3]
model_type = sys.argv[4]

assert model_type in ("m1", "m2")

nltk.download("stopwords")
nltk.download("punkt")

# Word tokenization.

token_regexp = regex.compile("(?u)\\b(\\p{L}+|\d+)\\b")
stop_words = set(stopwords.words("english"))


def tokenize(text):
    toks = token_regexp.findall(text)
    return list(filter(lambda x: x not in stop_words, toks))


# Sentence tokenization.

sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
max_short_len = 3


def sent_tokenize(text):
    # Basic preprocessing.
    text = text.lower()

    # Step 1: generate sentences and spans.
    sents = [tokenize(x) for x in sent_detector.tokenize(text)]
    spans = sent_detector.span_tokenize(text)

    # Step 2: concatenate short sentences.
    out_sentences = []
    out_spans = []
    short_sentence = []
    short_span = None
    for sent, span in zip(sents, spans):
        if short_span:
            short_sentence += sent
            short_span = (short_span[0], span[1])
        elif len(sent) <= max_short_len:
            short_sentence = sent
            short_span = span
        if len(sent) > max_short_len:
            if short_span:
                out_sentences.append(short_sentence)
                out_spans.append(short_span)
                short_sentence = []
                short_span = None
            else:
                out_sentences.append(sent)
                out_spans.append(span)
    if short_span:
        out_sentences.append(short_sentence)
        out_spans.append(short_span)

    return out_sentences, out_spans


lemmatizer = WordNetLemmatizer()
nltk.download("wordnet")

add_unparsed = True
gr_regexp = regex.compile("[^\w]")


def lemmatize(toks):
    return list(map(lemmatizer.lemmatize, toks))


def intersects(a1, b1, a2, b2):
    return b1 >= a2 and b2 >= a1


xmls_paths = list(filter(lambda f: f.endswith(".xml"), os.listdir(tasks_dir)))

train_data = []

for xml_path in xmls_paths:
    train_xml_path = os.path.join(tasks_dir, xml_path)
    xml_root = etree.parse(train_xml_path)
    suspicious_path = xml_root.getroot().attrib["reference"]
    suspicious_text = open(os.path.join(data_dir, "susp", suspicious_path)).read()
    for feature in xml_root.xpath("feature"):
        train_row = {}
        if feature.attrib["name"] != "plagiarism":
            continue
        # Read raw data
        suspicious_offset = int(feature.attrib["this_offset"])
        suspicious_length = int(feature.attrib["this_length"])
        source_offset = int(feature.attrib["source_offset"])
        source_length = int(feature.attrib["source_length"])
        source_path = feature.attrib["source_reference"]
        # Set df values
        train_row["obfuscation"] = feature.attrib["type"]
        train_row["suspicious_path"] = suspicious_path
        train_row["suspicious_offset"] = suspicious_offset
        train_row["suspicious_length"] = suspicious_length
        train_row["source_path"] = source_path
        train_row["source_offset"] = source_offset
        train_row["source_length"] = source_length
        # Read texts and set values
        source_text = open(os.path.join(data_dir, "src", source_path)).read()
        train_row["suspicious_text"] = suspicious_text[suspicious_offset:suspicious_offset + suspicious_length]
        train_row["source_text"] = source_text[source_offset:source_offset + source_length]
        train_data.append(train_row)

train_df = pd.DataFrame(train_data)
train_df.to_csv("pan_train_texts.csv")

pairs_dict = collections.defaultdict(set)
positions_dict = collections.defaultdict(set)
sentences_ids = collections.defaultdict(list)

with open("pan_train_texts.csv") as texts_file:
    header = texts_file.readline().strip().split(",")
    texts_reader = csv.DictReader(texts_file, header)
    for row in tqdm(texts_reader, desc="reading annotations"):
        susp_path = "susp/" + row["suspicious_path"]
        src_path = "src/" + row["source_path"]
        susp_start = int(row["suspicious_offset"])
        susp_end = susp_start + int(row["suspicious_length"]) - 1
        src_start = int(row["source_offset"])
        src_end = src_start + int(row["source_length"]) - 1
        positions_dict[susp_path].add((susp_start, susp_end))
        positions_dict[src_path].add((src_start, src_end))
        pairs_dict[susp_path, src_path].add((susp_start, susp_end, src_start, src_end))

for path, positions in tqdm(positions_dict.items(), desc="parsing annotations"):
    with open(data_dir + path) as fin:
        fin_text = fin.read()
        for i, (tokens, (sent_start, sent_end)) in enumerate(zip(*sent_tokenize(fin_text))):
            if len(tokens) > 0:
                for train_start, train_end in positions:
                    if intersects(train_start, train_end, sent_start, sent_end):
                        sentences_ids[path, train_start, train_end].append(i)

with open("pan_train_texts.csv") as texts_file, open("pan_train_sentences.csv", "w") as sentences_file:
    header = texts_file.readline().strip().split(",")
    texts_reader = csv.DictReader(texts_file, header)
    #         sentences_file.write("obfuscation,suspicious_path,source_path,suspicious_sentence_id,source_sentence_id\n")
    sentences_file.write("obfuscation,suspicious_path,source_path,suspicious_start_sentence_id,")
    sentences_file.write("suspicious_end_sentence_id,source_start_sentence_id,source_end_sentence_id\n")
    sentences_writer = csv.writer(sentences_file)
    for row in tqdm(texts_reader, desc="writing sentences"):
        obfuscation = row["obfuscation"]
        susp_path = "susp/" + row["suspicious_path"]
        src_path = "src/" + row["source_path"]
        susp_start = int(row["suspicious_offset"])
        susp_end = susp_start + int(row["suspicious_length"]) - 1
        src_start = int(row["source_offset"])
        src_end = src_start + int(row["source_length"]) - 1
        susp_ids = sentences_ids[susp_path, susp_start, susp_end]
        src_ids = sentences_ids[src_path, src_start, src_end]
        if len(susp_ids) == 0 or len(src_ids) == 0:
            continue
        if True:
            sentences_writer.writerow((obfuscation, susp_path, src_path, min(susp_ids), max(susp_ids),
                                       min(src_ids), max(src_ids)))
        if False and (len(susp_ids) == len(src_ids)):
            for susp_id, src_id in zip(susp_ids, src_ids):
                sentences_writer.writerow((obfuscation, susp_path, src_path, susp_id, src_id))
        if False and (len(susp_ids) <= 1 or len(src_ids) <= 1):
            for susp_id in susp_ids:
                for src_id in src_ids:
                    sentences_writer.writerow((obfuscation, susp_path, src_path, susp_id, src_id))

sentences = {}
sentences_dict = {}
files_sentences_ids = {}
sentences_files_ids = []
sents_cnt = 0
paths = collections.OrderedDict()

with open(tasks_dir + "pairs") as fin:
    for line in fin:
        susp_name, src_name = line.strip().split()
        paths["susp/" + susp_name] = 1
        paths["src/" + src_name] = 1

paths = list(paths.keys())

for path in tqdm(paths, desc="reading sentences"):
    lines = []
    with open(data_dir + path) as fin:
        fin_text = fin.read()
        for i, (tokens, (sent_start, sent_end)) in enumerate(zip(*sent_tokenize(fin_text))):
            lines.append(tokens)
    sentences[path] = []
    sentences_dict[path] = {}
    files_sentences_ids[path] = {}
    for i, line in enumerate(lines):
        if line:
            lemmas = lemmatize(line)
            files_sentences_ids[path][i] = sents_cnt
            sentences_files_ids.append((path, i))
            sentences[path].append(lemmas)
            sentences_dict[path][i] = lemmas
            sents_cnt += 1

# Похожесть предложения и текста.

pre_detections_list = []

test_paths = []

with open(tasks_dir + "pairs") as fin:
    for line in tqdm(fin, desc="calculating similarity"):
        susp_name, src_name = line.strip().split()
        susp_path, src_path = "susp/" + susp_name, "src/" + src_name
        test_paths.append((susp_path, src_path))
        susp_sents_is = list(files_sentences_ids[susp_path].keys())
        susp_sents_ids = list(files_sentences_ids[susp_path].values())
        src_sents_is = list(files_sentences_ids[src_path].keys())
        src_sents_ids = np.array(list(files_sentences_ids[src_path].values()))
        cands_sents_ids = collections.defaultdict(set)
        features, indices, pre_detections = [], [], []
        embed_dists = []
        for t1, (susp_sent_i, susp_sent_id) in enumerate(zip(susp_sents_is, susp_sents_ids)):
            susp_lemmas = sentences_dict[susp_path][susp_sent_i]
            susp_lemmas_set = set(susp_lemmas)
            # Шаг 1: вычисление расстояния по всем эмбеддингам
            top_dists = []
            for dists in embed_dists:
                top_dists.append(dists[t1])
            src_lemmas = sum(sentences_dict[src_path].values(), [])
            src_lemmas_set = set(src_lemmas)
            intersection = susp_lemmas_set & src_lemmas_set
            union = susp_lemmas + src_lemmas
            left_num = sum(map(intersection.__contains__, susp_lemmas))
            right_num = sum(map(intersection.__contains__, src_lemmas))
            iou_num = sum(map(intersection.__contains__, union))
            left_incl_dist = left_num / len(susp_lemmas)
            right_incl_dist = 1 - right_num / len(src_lemmas)
            iou_incl_dist = iou_num / len(union)
            susp_lens_dist = len(susp_lemmas)
            src_lens_dist = len(src_lemmas)
            min_lens_dist = min(len(susp_lemmas), len(src_lemmas))
            top_dists.append([left_incl_dist])
            top_indices = [(susp_sent_id, 0)]
            # Шаг 2: запись признаков
            for dists, ix in zip(zip(*top_dists), top_indices):
                features.append(dists)
                indices.append(ix)
        features = np.array(features)
        probas = features[:, 0]
        for (susp_sent_id, src_sent_id), p in zip(indices, probas):
            _, susp_sent_i = sentences_files_ids[susp_sent_id]
            _, src_sent_i = sentences_files_ids[src_sent_id]
            pre_detections.append((p, susp_sent_i, src_sent_i))
        pre_detections_list.append(pre_detections)

files_lens = {}

for susp_path, src_path in tqdm(test_paths, desc="calculating offsets"):
    if susp_path not in files_lens:
        files_lens[susp_path] = sent_tokenize(open(data_dir + susp_path).read())[1]
    if src_path not in files_lens:
        files_lens[src_path] = sent_tokenize(open(data_dir + src_path).read())[1]

threshold = 0.750

detections_list = []

for pre_detections in pre_detections_list:
    pre_detections = [(b, c, a) for a, b, c in pre_detections if a > threshold]
    detections_list.append(pre_detections)

# Уменьшение granularity на summary 2.0

out_detections_list = []

susp_dets_lens = []
src_dets_lens = []

for (susp_path, src_path), detections in zip(test_paths, detections_list):
    out_detections = []
    susp_lens = files_lens[susp_path]
    src_lens = files_lens[src_path]
    if len(detections) and min(detections, key=lambda pr: pr[0]) < max(detections, key=lambda pr: pr[0]):
        max_detection = max(detections, key=lambda pr: pr[2])[0]
        if model_type == "m1":
            susp_poss = [(susp_lens[max_detection][0], susp_lens[max_detection][1])]
        else:
            if max_detection == 0:
                susp_poss = [(susp_lens[1][0], susp_lens[-1][1] + 1)]
            elif max_detection == len(susp_lens) - 1:
                susp_poss = [(susp_lens[0][0], susp_lens[-2][1] + 1)]
            else:
                susp_pos1 = (susp_lens[0][0], susp_lens[max_detection - 1][1])
                susp_pos2 = (susp_lens[max_detection + 1][0], susp_lens[-1][1] + 1)
                susp_poss = [susp_pos1, susp_pos2]
        src_pos = (min(src_lens)[0], max(src_lens)[1] + 1)
        for susp_pos in susp_poss:
            susp_dets_lens.append(susp_pos[1] - susp_pos[0])
            src_dets_lens.append(src_pos[1] - src_pos[0])
            out_detections.append((src_pos, susp_pos))
    out_detections_list.append(out_detections)

with open(out_det_dir, "wb") as fout:
    pickle.dump(out_detections_list, fout)

print("done")

"""
Script filtrage et nettoyage du dataset
"""
import os
import re
import pandas as pd
import mwparserfromhell as mw
import tqdm

from transformers import AutoTokenizer

PARQUETS_DIR = "./data/parquets"
CSV_OUTPUT = "./data/csv/wikipedia-small.csv"

LINK = re.compile(r'\[\[(.*?)\]\]')

def links(string):
    links = [(a.group(), a.group(1), a.start(), a.end()) for a in LINK.finditer(string)]

    segments = []

    pointer_wiki = 0
    pointer_raw = 0

    raw = ""
    for wikicode, inner, start, end in links:
        wikicode_len = len(wikicode)
        inner_split = inner.split('|')

        if len(inner_split) == 1:
            _, value = None, inner_split[0]
        elif len(inner_split) == 2:
            _, value = inner_split
        else:
            raise Exception("mauvais tag wikimedia")


        before_text = string[pointer_wiki:start]
        before = (pointer_raw, pointer_raw + len(before_text), before_text, False)

        balise = (
            pointer_raw + len(before_text),
            pointer_raw + len(before_text) + len(value),
            value,
            True
        )

        segments += [before, balise]
        raw += before_text + value

        pointer_wiki = end
        pointer_raw = pointer_raw + len(before_text) + len(value)

    last_text = string[pointer_wiki:]
    last = (pointer_wiki, pointer_wiki + len(last_text), last_text, False)

    segments.append(last)

    assert pointer_wiki + len(last_text) == len(string)

    return raw, segments

def get_in_out_sets(segments):
    in_set = set()
    out_set = set()
    for start, end, _, en in segments:
        if en:
            in_set.update(range(start, end))
        else:
            out_set.update(range(start, end))
    return in_set, out_set

# 0 : Out
# 1 : B-Link
# 2 : I-Link
def make_output(input_ids, offset_mapping, sets):
    in_sets, _ = sets

    mirror = []
    last = 0
    for id, mapping in zip(input_ids, offset_mapping):
        map_start, map_end = mapping


        if map_start == map_end == 0:
            mirror.append(0)
            continue

        map_set = set(range(map_start, map_end))

        if map_set.intersection(in_sets):
            if last == 0:
                mirror.append(1)
                last = 1
            else:
                mirror.append(2)
                last = 2
        else:
            mirror.append(0)
            last = 0

    return mirror


def parquet_to_csv():
    os.makedirs("./data/csv", exist_ok=True)
    parquets = pd.read_parquet(PARQUETS_DIR)
    parquets.to_csv(CSV_OUTPUT, index=None)

def make_dataset():
    tokenizer = AutoTokenizer.from_pretrained("almanach/camembertv2-base")

    df = pd.read_csv(CSV_OUTPUT)

    final_texts = []
    outputs = []
    inputs = []
    attention_masks = []
    offset_mapping = []

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        text = row["text"]

        try:
            raw, segments = links(text)
        except Exception as e:
            print("erreur")
            continue

        final_texts.append(text)

        sets = get_in_out_sets(segments)

        #print(segments)

        tokens = tokenizer(
            raw,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_offsets_mapping=True
        )

        output = make_output(tokens["input_ids"], tokens["offset_mapping"], sets)

        inputs.append(tokens["input_ids"])
        outputs.append(output)
        attention_masks.append(tokens["attention_mask"])
        offset_mapping.append(tokens["offset_mapping"])


    ndf = pd.DataFrame({
        "text": final_texts,
        "input": inputs,
        "attention_mask": attention_masks,
        "offset_mapping": offset_mapping,
        "output": outputs,
    })

    ndf.to_parquet("./data/parquet/dataset-small.parquet")
    ndf.to_csv("./data/parquet/dataset-small.csv")

make_dataset()

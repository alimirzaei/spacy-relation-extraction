import json

import typer
from pathlib import Path

from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer
from spacy.lang.en import English
import random
nlp = English()
msg = Printer()


def get_doc_bin_on_file(file_path):
    data = list(file_path.open())
    samples = []
    relation_labels = set()
    for index in range(0, len(data), 4):
        sample = {}
        sentence = "\"".join(data[index].split("\"")[1:-1])
        relation = data[index+1].split('(')[0]
        if(len(data[index+1].split('(')) > 1): # there are relationship
            first_entity = data[index+1].split('(')[1].split(',')[0]
        else:
            first_entity = None
            relation = "Other"
        e1_start_index = sentence.find("<e1>")
        e1_end_index = sentence.find("</e1>")-4
        sentence = sentence.replace("<e1>", "")
        sentence = sentence.replace("</e1>", "")
        
        e2_start_index = sentence.find("<e2>")
        e2_end_index = sentence.find("</e2>")-4
        sentence = sentence.replace("<e2>", "")
        sentence = sentence.replace("</e2>", "")
        doc= nlp(sentence)
        try:
            e1 = doc.char_span(e1_start_index, e1_end_index, label="e1")
            e2 = doc.char_span(e2_start_index, e2_end_index, label="e2")
            doc.set_ents([e1, e2])
        except:
            print("error")
            continue
        if(first_entity):
            doc._.rel = {
                (e1.start, e2.start): {
                    relation: 1.0 if first_entity=='e1' else 0
                },
                (e2.start, e1.start): {
                    relation: 1.0 if first_entity=='e2' else 0
                }
            }
        else:
            doc._.rel = {
                (e1.start, e2.start): {
                    relation: 1
                },
                (e2.start, e1.start): {
                    relation: 1
                }
            }
        samples.append(doc)
        relation_labels.add(relation)
    for doc in samples:
        for key in doc._.rel:
            for r in relation_labels:
                if(r not in doc._.rel[key]):
                    doc._.rel[key][r] = 0 
    random.shuffle(samples)
    train_samples = samples[0:int(0.7*len(samples))]
    val_samples = samples[int(0.7*len(samples)): int(0.9*len(samples))]
    test_samples = samples[int(0.9*len(samples)):]
    docbin_train = DocBin(docs=train_samples, store_user_data=True)
    docbin_val = DocBin(docs=val_samples, store_user_data=True)
    docbin_test = DocBin(docs=test_samples, store_user_data=True)
    return docbin_train, docbin_val, docbin_test
    

def main(json_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    """Creating the corpus from the Prodigy annotations."""
    Doc.set_extension("rel", default={})

    train_data_path = Path('./assets/sem-eval/TRAIN_FILE.TXT')
    train_doc_bin, val_doc_bin, test_doc_bin = get_doc_bin_on_file(train_data_path)
    train_doc_bin.to_disk(train_file)
    msg.info(
        f"{len(train_doc_bin)} training sentences "
    )


    val_doc_bin.to_disk(dev_file)
    msg.info(
        f"{len(val_doc_bin)} dev sentences "
    )

    test_doc_bin.to_disk(test_file)
    msg.info(
        f"{len(test_doc_bin)} dev sentences "
    )


if __name__ == "__main__":
    typer.run(main)

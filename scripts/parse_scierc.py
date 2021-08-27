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
    docs = []
    for paper in json_file:
        offset = 0
        for s,rs,ner in zip(paper["sentences"], paper["relations"], paper["ner"]):
            text = ' '.join(s)
            doc = Doc(vocab, words=s, spaces=[True for i in s])
            ents = []
            for e in ner:
                try: 
                    ent = Span(doc, e[0]-offset, e[1]-offset+1, label=e[2])
                    doc.ents = list(doc.ents) + [ent]
                except:
                    print("err")
                    continue
            doc._.rel = [ {(-offset+r[0], -offset+r[2]): {
                                r[4] : 1.0 
                }} for r in rs if(-offset+r[0] in [e.start for e in doc.ents] and -offset+r[2] in [e.start for e in doc.ents])]
            offset += len(s)
            docs.append(doc)
    docbin = DocBin(docs=docs, store_user_data=True)
    return docbin
    

def main(json_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    """Creating the corpus from the Prodigy annotations."""
    Doc.set_extension("rel", default={})

    train_data_path = Path('./assets/SciErc/train.json')
    train_doc_bin = get_doc_bin_on_file(train_data_path)
    train_doc_bin.to_disk(train_file)
    msg.info(
        f"{len(train_doc_bin)} training sentences "
    )

    dev_data_path = Path('./assets/SciErc/dev.json')
    dev_doc_bin = get_doc_bin_on_file(dev_data_path)
    dev_doc_bin.to_disk(dev_file)
    msg.info(
        f"{len(dev_doc_bin)} training sentences "
    )

    test_data_path = Path('./assets/SciErc/test.json')
    test_doc_bin = get_doc_bin_on_file(test_data_path)
    dev_doc_bin.to_disk(test_file)
    msg.info(
        f"{len(test_doc_bin)} training sentences "
    )


if __name__ == "__main__":
    typer.run(main)

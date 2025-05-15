# Masters Thesis Code

This program is the practical component of my masters thesis, _"In Search of
Serendipity: Applying Actor-Network Theory to Zettelkasten and How Natural
Language Processing Can Help Manage Difficulties Arising from Zettelkasten's
Bottom-Up Structure"_

It aims to manage some of the difficulties arising from the bottom-up structure
of the Zettelkasten method by using machine learning processes, namely
similarity learning ([SBERT](https://sbert.net)) and topic modelling
([BERTopic](https://maartengr.github.io/BERTopic/index.html)).

The technical scope is (currently) somewhat limited, in that it is assumed the
user can output their Zettelkasten data as json
([zk](https://github.com/zk-org/zk) can do this).

A typical workflow may look as follows:

A user is currently working on a note, \_"Zettelkasten is an interface for
thought".

```bash
zk list --format json > data/zk.json # output data to json

./main.py sl train # train the similarity learning model
./main.py tm train # train the topic modelling model

# Compare the given note against all other notes in terms of semantic similarity.
# Data is saved to `data/simdiss.json`
./main.py sl compare --title "Zettelkasten is an interface for thought"

# list notes topically related to given search term
./main.py tm list --related "Zettelkasten is an interface for thought"

# list topics of corpus and their topic ids.
./main.py tm list --topics

# list documents belonging to given topic.
./main.py tm list --docs-for-topic <topic-id>

# search for topics related to given term.
./main.py tm list --topic-search <string>
```


## Installation

```bash
git clone git@git.sr.ht:~tjex/thesis-code
cd thesis-code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

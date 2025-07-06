# Masters Thesis Code

This program is the practical component of my master's thesis, _"In Search of
Serendipity: Applying Actor-Network Theory to Zettelkasten and How Natural
Language Processing Can Help Manage Difficulties Arising from Zettelkasten's
Bottom-Up Structure"_ written in conclusion of the Master of Creative
Technologies degree at Filmuniversität Babelsberg KONRAD WOLF.

It aims to manage some of the difficulties arising from the bottom-up structure
of the Zettelkasten method by using machine learning processes, namely
similarity learning ([SBERT](https://sbert.net)) and topic modelling
([BERTopic](https://maartengr.github.io/BERTopic/index.html)).

The `demo.cast` in the root of this repository is a terminal recording of this
program's functionality. It can be played back within the terminal using
[asciinema](https://asciinema.org/) (requires installation) and is available to
[view online](https://asciinema.org/a/lMz4rISK6pUD0LdJHic6jywnK).

## Requirements

- [`zk`](https://github.com/zk-org/zk) (required to export notes to JSON)
- Zettelkasten notes as local markdown files
- Python3.13 or higher

## Installation and Setup

### Program

The following block of commands will clone the repo and set up the project. It
will install all python packages in a virtual environment, BUT it will download
NLTK assets. NLTK will print the destination of where it's writing to, but you
can set it to a specific destination with (for example):

```bash
export NLTK_DATA="$XDG_DATA_HOME/nltk_data"
```

Note: This project has only been tested with python version 3.13

```bash
git clone git@git.sr.ht:~tjex/thesis-code
cd thesis-code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup.py
python main.py --help
```

### Zettelkasten

If the user does not already use `zk` as their Zettelkasten program, they should
copy their current Zettelkasten (e.g, Obsidian vault) to a new location. Enter
that directory and execute `zk init` and follow the prompts.

In `.zprofile` / `.profile` set
`export ZK_NOTEBOOK_DIR="${HOME}/path/to/copied/vault"` and refresh the shell or
close and open a new one.

Now `zk` commands can be run from anywhere as the global notebook has been set.
Navigate to this clone repository and implement the below workflow example.

Alternatively, there is a pre-exported demo data set,
`./data/ideaverse-lite-1.5.json` that can be used to test this program's
functionality without exporting your own data. It's the publicly available
[LYT Kit](https://www.linkingyourthinking.com/myideaverse/treasure).

Simply rename this file to `zk.json` and run the example as below.

Note: the title used in the examples below will not work with the demo data.

## Workflow Example

A user is currently working on a note, \_"Zettelkasten is an interface for
thought".

```bash
zk list --format json > ./data/zk.json # output data to json

python ./main.py sl train # train the similarity learning model
python ./main.py tm train # train the topic modelling model

# Compare the given note against all other notes in terms of semantic similarity.
# Data is saved to `data/simdiss.json`
python ./main.py sl compare --title "Zettelkasten is an interface for thought"

# Cluster notes by similarity (default: 10)
python ./main.py sl cluster --clusters <n>

# list topics of corpus and their topic ids.
python ./main.py tm list --topics

# list documents belonging to given topic.
python ./main.py tm list --docs-for-topic <topic-id>

# list notes topically related to given search term
python ./main.py tm list --related "Zettelkasten is an interface for thought"

# search for topics related to given term.
python ./main.py tm search <search term>

```

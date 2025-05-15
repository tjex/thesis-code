simdiss-test:
	python main.py sl compare --title "Zettelkasten is an interface for thought"

simdiss-random:
	zk list --quiet --format "{{title}}" --sort random --limit 1 --no-input \
		| xargs -I{} python main.py simdiss "{}"

export-note-data:
	zk list --format json > data/zk.json


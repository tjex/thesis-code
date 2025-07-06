#!/usr/bin/env bash

# Generate a test corpus of 100 cloned notes.
# Useful for varifying that the similarity learning models are performing
# correctly. All notes should be 100% similar with each other across the entire
# corpus.

title="foo"
body="Addressing the question [what connections are there between the Zettelkasten\nsystem, creative ideation and network graphs?](../../it61). Trying to think\nabout how zk and graphs are complex systems, and how creative ideation is the\nresult of those systems. How does that happen? That [consciousness occurs when\ncolonies of cells are large enough in number and connections](../../k8pg) is an\ninteresting angle, but in the case of zk, the notes aren't 'interacting' with\nthemselves per se. Rather, it is the interaction within the mind, of the\ninformation they hold, that somehow comes together to create an idea.\n\nIdea's are generated from borrowing parts of pre-existing information."
file="100_clones.json"

rm $file
touch $file
echo "[" >>$file
for i in {0..99}; do
    echo "{\"title\": \"${title}-${i}\",\"body\": \"${body}\"}," >>$file
done
echo "{\"title\": \"bar\",\"body\": \"${body}\"}" >>$file
echo "]" >>$file

Data here was generated by calculating similarity on the note:

"Zettelkasten is an interface for thought"

Generated at this commit: 6bd4584

## Files

no-cleaning: was just passing note bodys as they were.

manual: was with regex based removal of markdown symbols and markdown
link punctuation and filepath.

```python
md_symbols_patt = r"(#)|(>)|(\*)"
md_link_patt = r"\[(.*?)\]\(.*?\)"
```

manual-and-stop-words: was using manual cleaning (above) and nltk stop words

word-stop: using nltk stop words and word tokenization.

sent-stop: using nltk stop words and sentence tokenization.


## Findings

sent-stop provided the best results in that it returned fewer "most similar"
notes. The other methods progressively returned more, some returned notes were
clearly not that related in the no-cleaning method.

the manual-and-stop-words was similar in performance, but returned a few extra
notes that were arguably not that relevant. It would seem that the manual
cleaning negatively effects semantic similarity in this use  case.

An method that produces fewer most similar notes is preferable as with a
Zettelkasten with thousans of notes, the amount of notes that can be returned
may be very large. Being more rigorous with the most_similar creates less input
for the user to have to sift through.

the method used to evaluate how the returned values differed was a diff
comparrison:

`diff --side-by-side simdiss-stop-words-most.json simdiss-nltk-most.json | less -S`

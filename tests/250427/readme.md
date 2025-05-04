simdiss dbase: no cleaning at all. passing note bodies as is.
simdiss manual: remove newlines, remove md syntax, remove md link syntax

diff shows more accurate and relevant results being attributed to the segments
in simidss-manual.json. 

E.g "Communicating with slip boxes" moved from "moderatly relevant" to "Somewhat
relevant" (an increase in relevance by one segment). 

Shows that manual cleaning is beneficial in semantic analysis by SBERT.

similarity tensor comparrison:

Mean absolute difference: 0.023315
Max absolute difference: 0.402765

diff checker: https://www.diffchecker.com/FiB9R0y0/

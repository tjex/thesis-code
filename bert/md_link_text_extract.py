# md link extract

import re

link = (
    "[target text](../file.md) some foo more text foo, with foo [more text](../file.md)"
)


regex1 = r"\[(.*?)\]\(.*?\)"
regex2 = r"\[.*?\]"


link = re.sub(regex1, r"\1", link)

print(link)

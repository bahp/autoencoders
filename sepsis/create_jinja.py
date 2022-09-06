# Libraries
from pathlib import Path

path = './objects/results/baseline'

# Create index
content = ""
for path in Path(path).glob('**/*.jpg'):
   content += """
        <img src="%s" width=200, height=200/>
   """ % path

print(content)

html = "<html><head></head><body>" + content + "</body></html>"

with open('index.html', 'w') as f:
    f.write(html)
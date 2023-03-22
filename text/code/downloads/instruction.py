'''
https://github.com/mahnazkoupaee/WikiHow-Dataset
'''
import json
import csv
from pathlib import Path

import gdown
from tqdm import tqdm

from utils import root


_DATA_URL = "https://public.boxcloud.com/d/1/b1!mySdjRKDp9_FKZZlB7fvwq3YZZ9J2O5m_QGmE4MY1PlocarH7KvxgnV9p5NOF0aYw8lGAISLGL98Dblr-w607ciSG4JxzL7XQ0dP-NmIvn1NgGnLPw4-MTJUPlzCNKKUx-38QhkQJAByyliVFw-_Fni27eJkgyhIOHiLlpNrE8kNTT-v_kJT7oLySRNQaMDKlvAldTr5-vA9adXJy4N7I6BgiXnoM1BU9tJRxB3IHKjifn70bzzB8Mw1BnFBxB-JYZSt3pI-RFPdDcsL2zZlpMfMgKAkrnU2kuAdK_1JS65YSEElKKlbyH_Z6tVZ5kGkuoR3rJY-hwcsVkb4W9IM27eGgVxWABVkrcRhjIuYsWeRp2pEWDDyx_56czIEBesugsDZb5PLq8xQ8zoOcnFqT5mUwpIrANCVqtxjBHZ6phWb7stJ1P51wI0HT8Zs8xxj71cD20ndc36kZuMfpESo6EVudQE9rf6dFwz12WvWnBma6-c0-_Ordfb-6RoIJvLQi4j3NduC9UCFeMgDIErg2B_AsHiFf-w3to_CI1AAEX-N-OB6gtlMfHFSOkM3OCA8gZYmRVQj4JBCKk79QG_wLJdRfiA5SB0BoNHeNB3XXzIxkRFf2MKf7ZjDGn3gZhOvSxN4fSCR9gMB2GMfZxHh7VpsN6S72SDH9Dp3AQheDtxsZmlQAEfaUvp6XDO0VK0Jq6MHO-0-DtWly_6sC7FUfQDiB-5bMfbBS8Zz3Wi1Mrxri-IJO5Uj_GR0Ok6-uU4ZJzhQI8jo3-Iy9wP_RAihRfJRlXdFUvhFauHLfOHbbEBG2bQTIz3qUoRG0kSMigXEwrG07wd5y4dbnrVaaYup3pGLA3JEeSxmK663FmiNDf_XrpY-NR0kQQ9eIy6U089xDyz27zJiiKAJn6LY9xstgRgA6tvaXjEGiek9L7WupiIOCLLnAg1wYRVycJRnp7vOAeJNeKuBM7uMEZPSlViTBUItW2Pi4Ot4l9N5YkSmB3o4qmP51Mm0S0DjA-CbZhBWYT48PzD90m8nqV5VAD38YvnKKhROoAOUoRWbjDP7NrFjRDP60WRg0xniJNDK840UqDizUZcd5xfDump6JPkaEOLS8FLx4sTR_VdMccX2E9NRp8jSXG21Q0jj1jm4o8oHXAbCiZfQwjW3melfn8mL1Pts-hXE7yUwYS0xVASATub3juKJag9b3IAGv1oeMYWeH6CFl0ShRc5lmiM733vqrTlNkSm9PtW90mGLBXlNsDx6aQzPcvC16r_V/download"


data_dir = root / 'data/raw/instruction'
out_dir = root / 'data/texts'
data_dir.mkdir(exist_ok=True)
Path(out_dir).mkdir(exist_ok=True)

data_path = data_dir / 'wikihowAll.csv'

if not data_path.is_file():
    gdown.download(_DATA_URL, str(data_path), quiet=False)

data = []
with open(data_path) as f:
    reader = csv.reader(f)
    for row in tqdm(reader):
        txt = row[0].strip()
        txt = '.'.join(txt.split('.,'))
        data.append(txt.replace('\n', ' '))

print(f"{len(data)} lines in total")
with open(Path(out_dir) / 'instruction.txt', 'w') as f:
    for line in data:
        f.write(f'{line}\n')

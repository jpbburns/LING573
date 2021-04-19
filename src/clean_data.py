# cli usage: clean_data.py [csv_file.csv]

import pandas as pd
from sys import argv

#csv_path = 'train.csv'
csv_path = argv[1]

df = pd.read_csv(csv_path)
df_rows = len(df.axes[0])

class Headline:
    
    def __init__(self, data_frame, row_num, edited_version=True):
        
        # Gather raw fields
        
        self.id =           df.iloc[row_num].id
        self.unigrams =     df.iloc[row_num].original.split()
        self.edit =         df.iloc[row_num].edit
        self.grades =       df.iloc[row_num].grades # we may not actually need this
        self.meanGrade =    float(df.iloc[row_num].meanGrade)
        
        ### build unigrams ###
        
        # if we want the edited headline, find the tagged word and replace it
        # with the edit
        
        # otherwise, find the tagged word and clean it, and set meanGrade to 0
        # (we assume the original headlines are 'not funny')
        
        for i in range(len(self.unigrams)):
            if self.unigrams[i][0] == '<' and self.unigrams[i][-2:] == '/>':
                if edited_version:
                    self.unigrams[i] = self.edit
                else:
                    self.unigrams[i] = self.unigrams[i][1:-2]
                    self.meanGrade = 0.0
                    self.id = -self.id  
                    # this is to force unique dict keys for the two versions
                    # of each headline: if id = x, it's the edited version; if
                    # id = -x, it's the original version of the same headline


all_headlines = {}

# from each row, add the edited and unedited versions to the dict with their
# id's as keys: positive id's are edited, negative id's are original
for row in range(df_rows):
    hl_edited = Headline(df, row, edited_version=True)
    hl_orig = Headline(df, row, edited_version=False)
    all_headlines[hl_edited.id] = hl_edited
    all_headlines[hl_orig.id] = hl_orig
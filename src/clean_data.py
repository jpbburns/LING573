import pandas as pd
from sys import argv

csv_path = 'train.csv'
#csv_path = argv[1]

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
        
        for i in range(len(self.unigrams)):
            if self.unigrams[i][0] == '<' and self.unigrams[i][-2:] == '/>':
                if edited_version:
                    self.unigrams[i] = self.edit
                else:
                    self.unigrams[i] = self.unigrams[i][1:-2]
                    self.meanGrade = 0.0
                    self.id = -self.id


all_headlines = {}

for row in range(df_rows):
    hl_edited = Headline(df, row, edited_version=True)
    hl_orig = Headline(df, row, edited_version=False)
    all_headlines[hl_edited.id] = hl_edited
    all_headlines[hl_orig.id] = hl_orig


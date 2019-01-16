# Fake News Detection
The latest hot topic in the news is fake news and many are wondering what data scientists can do to detect it and stymie its viral spread. This dataset is only a first step in understanding and tackling this problem.

__Dataset__
- The data source used for this project is dataset which contains 3 files with .tsv format for test, train and validation. Below is some description about the data files used for this project.

- Column 1: the ID of the statement ([ID].json).
- Column 2: the label. (Label class contains: True, Mostly-true, Half-true, Barely-true, FALSE, Pants-fire)
- Column 3: the statement.
- Column 4: the subject(s).
- Column 5: the speaker.
- Column 6: the speaker's job title.
- Column 7: the state info.
- Column 8: the party affiliation.
Column 9-13: the total credit history count, including the current statement.
- Column 9: barely true counts.
- Column 10: false counts.
- 11: half true counts.
- 12: mostly true counts.
- 13: pants on fire counts.
- Column 14: the context (venue / location of the speech or statement).

__Prerequisites__  
- Python3.6
- Sklearn
- Numpy
- Pandas
- Scipy
- NLTK
- Matplotlib

### Project by : Rashid and Trisha
### Mentor : Sarah Masud
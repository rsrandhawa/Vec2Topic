# Vec2Topic
A Topic Modeling algorithm that extracts the core topics from a text corpus. 

Implements the algorithm as described in R. S. Randhawa, P. Jain, and G. Madan, [Topic Modeling Using Distributed Word Embeddings] (http://arxiv.org/abs/1603.04747)

You can try an online working version at http://topic.randhawa.us 
## Requirements: 
Python 2.7, and other packages listed in requirements.txt 

## Usage
Download all files into a single folder. The code uses Wikipedia trained vectors to augment its learning on the provided corpus. You can download these files [here](https://www.dropbox.com/sh/e0t37fpq9j226yw/AAD4O_4SZ6jB5jpu9QCiJ9PJa?dl=0): it is about 10 Gigs. Place these files (*wiki.pkl* and *wiki.shelve*) within a sub-directory called wikimodel.  Then run the following command:

```python
python vec2topic.py -i corpus.txt -g /path/to/wikimodel/ -K num_topics -s stopwords.txt
```

### Inputs
1. -i: *input text file*. The script takes as input a single text file
2. -g: *path to global vectors*. There should be a trailing “/“ in the /path/to/wikimodel/ argument
3. -K: *num_topics*, *(Optional)*: the number of topics to generate. Defaults to 10 if not provided.
4. -s: *stopwords.txt*, *(Optional)*. If it is provided then stopwords.txt file is loaded and the contained words are removed from analysis. If no argument is given then no stop words are used.

### Outputs
Three *csv* files are generated:

1. *corpus_topics.csv*: contains the topics sorted by score: Topic 1>Topic 2>Topic 3>..., and within each topic, words are sorted by their score
2. *corpus_score.csv*: sorted list of words in the corpus based on the score
3. *corpus_depth.csv*: sorted list of words in the corpus based on depth


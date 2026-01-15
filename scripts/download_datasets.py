"""
Download and prepare datasets for keyboard suggestion model training.

This script downloads necessary datasets and prepares them for training:
1. Word frequency list (for word completion)
2. Text corpus (for next-word prediction)
3. Common words (for typo generation)

Run this before training the model.
"""

import os
import urllib.request
import zipfile
from pathlib import Path


def download_word_frequencies(output_dir: str = "./data/keyboard"):
    """
    Download word frequency list.
    
    Using: Google Books N-grams (top 50K English words)
    Alternative: COCA word frequency list
    """
    print("Downloading word frequency list...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # For now, create a sample word list
    # TODO: Replace with actual download from Google Books N-grams
    sample_words = """
the
be
to
of
and
a
in
that
have
i
it
for
not
on
with
he
as
you
do
at
this
but
his
by
from
they
we
say
her
she
or
an
will
my
one
all
would
there
their
what
so
up
out
if
about
who
get
which
go
me
when
make
can
like
time
no
just
him
know
take
people
into
year
your
good
some
could
them
see
other
than
then
now
look
only
come
its
over
think
also
back
after
use
two
how
our
work
first
well
way
even
new
want
because
any
these
give
day
most
us
hello
help
helping
world
thank
thanks
please
sorry
yes
no
maybe
okay
great
good
bad
nice
love
hate
like
want
need
have
make
take
give
show
tell
ask
try
call
keep
hold
turn
start
stop
run
walk
talk
work
play
live
die
buy
sell
read
write
hear
see
feel
think
know
understand
remember
forget
learn
teach
study
practice
""".strip().split('\n')
    
    output_path = os.path.join(output_dir, "word_freq.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in sample_words:
            word = word.strip()
            if word:
                f.write(f"{word}\n")
    
    print(f"✓ Word frequency list saved to: {output_path}")
    print(f"  Words: {len(sample_words)}")
    print("\n⚠ NOTE: This is a sample list. For production, download:")
    print("  - Google Books N-grams: https://storage.googleapis.com/books/ngrams/books/datasetsv3.html")
    print("  - Or COCA word frequency: https://www.wordfrequency.info/")
    
    return output_path


def download_text_corpus(output_dir: str = "./data/keyboard"):
    """
    Download text corpus for next-word prediction.
    
    Using: OpenSubtitles or similar
    """
    print("\nDownloading text corpus...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # For now, create sample sentences
    # TODO: Replace with actual corpus download
    sample_sentences = """
how are you doing today
i am going to the store
thank you for your help
have a great day
see you later tonight
what time is it now
where are you going
can you help me please
i don't know what to do
let me know if you need anything
that sounds good to me
i will be there soon
how was your day today
what are you doing tonight
i am so happy to see you
thank you so much for everything
have a wonderful weekend
see you tomorrow morning
i love you very much
take care of yourself
good morning everyone
good night sleep well
how is it going
what do you think about this
i think that is a great idea
let me check on that
i will get back to you
sounds good to me
no problem at all
you are welcome
nice to meet you
pleased to meet you
how have you been
long time no see
what have you been up to
not much just working
same here nothing new
that is interesting to know
i did not know that
tell me more about it
i would love to hear more
that makes sense to me
i understand what you mean
i see what you are saying
that is a good point
you are absolutely right
i completely agree with you
i think so too
me too same here
i feel the same way
that is exactly what i think
""".strip().split('\n')
    
    output_path = os.path.join(output_dir, "corpus.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sample_sentences:
            sentence = sentence.strip()
            if sentence:
                f.write(f"{sentence}\n")
    
    print(f"✓ Text corpus saved to: {output_path}")
    print(f"  Sentences: {len(sample_sentences)}")
    print("\n⚠ NOTE: This is a sample corpus. For production, download:")
    print("  - OpenSubtitles: https://opus.nlpl.eu/OpenSubtitles.php")
    print("  - Or Reddit comments: https://files.pushshift.io/reddit/")
    
    return output_path


def prepare_all_datasets(output_dir: str = "./data/keyboard"):
    """
    Download and prepare all datasets.
    """
    print("="*60)
    print("Preparing Keyboard Suggestion Training Datasets")
    print("="*60)
    
    # Download word frequencies
    word_freq_path = download_word_frequencies(output_dir)
    
    # Download text corpus
    corpus_path = download_text_corpus(output_dir)
    
    print("\n" + "="*60)
    print("✓ Dataset preparation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run data preparation script to generate training pairs")
    print("2. Train model in Colab")
    print("\nFiles created:")
    print(f"  - {word_freq_path}")
    print(f"  - {corpus_path}")
    
    return {
        "word_freq": word_freq_path,
        "corpus": corpus_path
    }


if __name__ == "__main__":
    prepare_all_datasets()

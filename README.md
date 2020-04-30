# TextSummarizer
This is a command line text summarization tool that builds off of Skip-Thought Networks

### Dependencies
* python 2.7
* Theano 0.7
* numpy 
* scipy
* click

### Skip-Thoughts Code
Available at https://github.com/ryankiros/skip-thoughts
* clone repository into the TextSummarization folder
* download the pre-trained models specified in the skip-thoughts repository
* if you would like, train your own model


### Setting up necessary paths
* in skip-thoughts.py change the paths to where the pre-trained models are on your machine
* in training/tools.py edit the paths to reflect where the models you trained are on your machine

### How to use on the command line
Usage: Summarizer.py [OPTIONS]

Options:
  --trained / --pre-trained  specify if you want to use trained or pre-trained
                             model

  --file TEXT                set path to a tokenized file you want to
                             transcribe

  --folder TEXT              set path to a folder of tokenized files you want
                             to transcribe

  --help                     Show this message and exit.

### References
* https://github.com/jatana-research/email-summarization/blob/master/email_summarization.py
* https://github.com/ryankiros/skip-thoughts

autohl
======

This project is attempting to create an auto-highlighter via text extraction.

Currently, autohl.py will process a text file and attempt to split it into sentences.  The sentences are compared via the overlap of stemmed words and then ranked via the PageRank algorithm (TextRank: Rada Mihalcea and Paul Tarau, 2004: TextRank: Bringing Order into Texts, Department of Computer Science University of North Texas).  The output is written to an html file with a slider to display the level of highlighting corresponding to percentage of sentences highlighted.

The next iteration of this project will be a port to pure JavaScript to allow a Chrome extension.
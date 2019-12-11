# CABR
Chris' Auditory Brainstem Response (Tools) is a small package which can analyze and export ABR based experiments.

The program demands a certain file structure to run. In a large experiment folder, there should be folders for each experimental condition. In each condition folder are folders for each animal. Within these folders are files containing analyzed abr data for a single tone pip ABR.  
```
Experiment
├── Condition 1
│   ├── Animal 1
│   ├── Animal 2
│   └── Animal 3
└── Condition 2
    ├── Animal 4
    ├── Animal 5
    └── Animal 6
```

There are two operations: **write** and **plot**
within each, you can choose to display either threshold or amplitude growth functions (agf)

For example, a usage may be:

```
import parser
import CABR3

abr = CABR3.ABR('./Experiment', ParsingClass = parser.EPL)

abr.write.threshold()
abr.write.agf()
abr.plot.threshold()
abr.plot.agf()
```
**parser** is an included library which is used to import the data files. By default it resorts to the analyzed ABR formatting of the Eaton Peabody Laboratories of Mass Eye and Ear. 

CABR can accept any custom parsing class as long as it has properties:
```
*treatment*
*frequency*
*threshold*
*levels*
*amplitudes*
```
parser.Parser is the parent class of any specific parser class. 


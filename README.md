# Multi-document Summarization Evaluation
This repository contains the summaries and evaluations from the paper 'A Dataset for the Analysis of Text Quality Dimensions in Summarization Evaluation'. 

###The Summaries
In the 'summaries' folder you find the summaries generated by ten different automatic systems. Each system summarized 49 topics, each containing a number of heterogenous documents. The topics are named 1001 to 1050, note that there is no topic 1009. The corpus used was presented by this paper: [Beyond Generic Summarization: A Multi-faceted Hierarchical Summarization Corpus of Large Heterogeneous Data](http://www.lrec-conf.org/proceedings/lrec2018/pdf/252.pdf)

###The Annotations
For more details have a look at the paper.

#LikertAnno 
'LikertAnno.csv' contains annotations of the summaries of seven systems: H1-5, MMR and LeadFirst. A likert-scale of 1 to 5 is used, with 5 representing the best rating. The annotations jugde the summaries on eleven quality criteria: 
Non-Redundancy
Referential Clarity
Grammaticality
Focus
Structure
Coherence
Readability
Information Content
Spelling
Length
Overall Quality

#PairAnnno
'PairAnnno.csv' contains the pairwise comparisons of the summaries of six systems: H1-4, MMR\*, PG-MMR and Submodular. The annotations were done by Amazon Mechanical Turk workers. Each annotation/row marks either the left summary or the right summary as better regarding the criterion (exactly 0 or 1).
Only the following criteria were used:
Non-Redundancy
Referential Clarity
Structure
Readability
Information Content
Overall Quality

# License
* The annotations and summaries are licensed under the [Creative Commons CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
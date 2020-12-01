# Identification of Human CYP450 Inhibition Using Multitask Convolutional Neural Networks

#### T-H Nguyen-Vo, Q. H. Trinh, L. Nguyen, P-U. Nguyen-Hoang, T-N. Nguyen, [L. Le](http://cbc.bio.hcmiu.edu.vn/)∗, and [B. P. Nguyen](https://homepages.ecs.vuw.ac.nz/~nguyenb5/about.html)∗


![alt text](https://github.com/mldlproject/2020-CYP450-mCNN/blob/main/CYP450_abs.svg)

**Motivation**: Human cytochrome P450 (CYP) superfamily holds responsibilities for the metabolism
of both endogenous and exogenous compounds such as drugs, cellular metabolites, and toxins. Inhibition 
of CYP450 isoforms is closely associated with adverse drug reactions which may cause metabolic failures 
and even induce serious side effects. In modern drug discovery and development, identification of potential 
CYP isoforms’inhibitors is highly essential. Besides experimental approaches, numerous computational 
frameworks have been recently developed to address this biological issue. In our study, we propose robust, 
stable, and effective prediction models to virtually screen for five CYP isoforms’ inhibitors, including 
CYP1A2, CYP2C9, CYP2C19, CYP2D6, and CYP3A4. Our proposed method employs multitask learning combining with 
the connection-based feature to significantly boost the predictive power. For a particular task, the 
connection-based feature of a compound presents its class responses to other related tasks.

**Results**: The obtained results show that multitask learning has remarkably leverage useful information 
contained in various related tasks to promote generalization performance for all tasks. Besides, the 
utilization of connective relations among related tasks improved the performance of specific tasks. 
Additionally, the combination of multitask learning and connection-based features came up with outstanding 
results for the five tasks with all ROC-AUC and PR-AUC values of higher than 0.90.

**Availability and implementation** Source code and data are available on [GitHub](https://github.com/mldlproject/2020-CYP450-mCNN)


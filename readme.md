# iCYP-MFE: Indentifying Human Cytochrome P450 Inhibitors using Multi-task Learning and Molecular Fingerprint-embedded Encoding

#### T-H Nguyen-Vo, Q. H. Trinh, L. Nguyen, P-U. Nguyen-Hoang, T-N. Nguyen, D. T. Nguyen, [B. P. Nguyen](https://homepages.ecs.vuw.ac.nz/~nguyenb5/about.html)∗ and [L. Le](http://cbc.bio.hcmiu.edu.vn/)∗

![alt text](https://github.com/mldlproject/2020-CYP450-mCNN/blob/main/CYP450_abs0.svg)

## Motivation
The human cytochrome P450 (CYP) superfamily holds responsibilities for the metabolism of both endogenous and exogenous compounds such as drugs, cellular metabolites, 
and toxins. The inhibition exerted on the CYP enzymes is closely associated with adverse drug reactions encompassing metabolic failures and induced side effects. 
In modern drug discovery, identification of potential CYP inhibitors is, therefore, highly essential. Alongside experimental approaches, numerous computational models 
have been proposed to address this biochemical issue. In this study, we introduce iCYP-MFE, a computational framework for virtual screening on CYP inhibitors toward 1A2, 
2C9, 2C19, 2D6, and 3A4 isoforms. iCYP-MFE contains a set of five robust, stable, and effective prediction models developed using multitask learning incorporated with 
molecular fingerprint-embedded features. 

## Results
The results show that multitask learning can remarkably leverage useful information from related tasks to promote global performance. Comparative analysis indicates that iCYP-MFE achieves three predominant tasks, one equivalent task, and one less effective task compared to state-of-the-art methods. The area under the receiver operating characteristic curve (AUC-ROC) and the area under the precision-recall curve (AUC-PR) were two decisive metrics used for model evaluation. The prediction task for CYP2D6-inhibition achieves the highest AUC-ROC value of 0.93 while the prediction task for CYP1A2-inhibition obtains the highest AUC-PR value of 0.92. The substructural analysis preliminarily explains the nature of the CYP-inhibitory activity of compounds. An online web server for iCYP-MFE with a user-friendly interface was also deployed to support scientific communities in identifying CYP inhibitors.


## Availability and implementation
Source code and data are available on [GitHub](https://github.com/mldlproject/2021-iCYP-MFE)

## Web-based Application
[Click here](http://13.238.182.15:8888/)

## Citation
Thanh-Hoang Nguyen-Vo, Quang H. Trinh, Loc Nguyen, Phuong-Uyen Nguyen-Hoang, Thien-Ngan Nguyen, Dung T. Nguyen, Binh P. Nguyen, and Ly Le. iCYP-MFE: Identifying Human Cytochrome P450 Inhibitors Using Multitask Learning and Molecular Fingerprint-Embedded Encoding. *Journal of Chemical Information and Modeling* (2021). [DOI: 10.1021/acs.jcim.1c00628](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00628).

## Contact 
[Go to contact information](https://homepages.ecs.vuw.ac.nz/~nguyenb5/contact.html)

## Papers
- https://arxiv.org/pdf/2206.04792: explians context drivt and similar problems then ours 
- https://link.springer.com/content/pdf/10.1007/s10462-024-10995-w.pdf: explians differnt anonly detectors 
- https://arxiv.org/pdf/2206.09426 DL-based unsupervised methods are surprisingly worse than shallow methods
- https://dl.acm.org/doi/pdf/10.1145/3606274.3606277 The paper is written a little odd but it describs the problem quite well
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9537291 use easy domains to identify anomaly and only present hard domain.
- https://zenodo.org/records/6627050: DingDongData, syscall order malicious and not

## Problems
I want to measure the accuracy of my models. How do i label my dataset/find a ground truth? 
Currently i used as a baseline freq1d measuring the occurrents of certain states per feature and calculating the likelihood of the the current. I then label all one-off events (events that are totally unique, differ at least in one property for all others) and run my freq1d dector, it labels near non of them, meaning that might be unique but they contain very common features. Should these one-offs be considered anomaly? What even is an anomaly?

- Task isn't clear:
    - Naming is chaos, nothing is uniformly defined: 
        - Novelty: Event that did not occur before
        - Outlier: Event that is very far from the other (on what space?)
        - Anomaly: Event that is unlikely to happen
        - Malicious: Event that is caused by an attack
    - Problem is that there are differences and paper just assume something fitting to there goal without tell us.
    - Twin Freak problem: Is a repeated anomaly still an anomaly? Many pares say yes, i would say no. 
    
- Model selection isn't clear
    - https://dl.acm.org/doi/pdf/10.1145/3606274.3606277 The paper is written a little odd but it describes the problem quite well
    - Is a not well defined problem: 
    > Given an unlabeled outlier detection task, which detection algorithm and associated hyper parameter (HP) settings should one use? [...] the problem is notoriously hard in the absence of any labeled data, any well-accepted objective or loss function, and potentially very large model space especially for deep outlier detectors with many HPs.
    - Selecting models at random is a valid strategies, therefore are models are equally week:
    > Our study reports a striking finding, that none of the existing and adapted strategies would be practically useful: stand-alone ones are not significantly different from random, and consensus-based ones do not out perform iForest (w/ default HPs) while being more expensive 

- No good comparable datasets
    - most datasets papers where tested on are useless: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9537291
    - datasets have small feature and event sizes. 
    - datasets are mostly labled accouring to some hidden outer information. Examples: 
        - Number of Taxi orders on Holidays is abnormal. 

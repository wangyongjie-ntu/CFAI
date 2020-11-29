[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/wangyongjie-ntu/Awesome-explainable-AI/graphs/commit-activity)
![versions](https://img.shields.io/pypi/pyversions/pybadges.svg)
![](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg)

# About Counterfacutal Explanations (Pytorch)

Counterfactual explanations mainly target to find an explanation of a given instance, such that the explanation can change the prediction(Ususally from an undesirable outcome to an ideal one). The explanation itself is also a valid instance, or changing of current instance(can reduce to another instance implicitly) in the feature space. It has broad applications, E.g., finance, education etc.

This task is popular recently. However, I only find few released repositories on Pytorch platform. Hence, I create this project and plan to implement some famous algorithms continously. 
I appreciate it very much if anyone can contribute to this project together with me.  For the implementation details, I am glad to discuss with all of you together.

A collection of methods on counterfactual explanation to add into this repository. Continously updating.

- [ ] PlainCF, [Counterfactual explanations without opening the black box: Automated decisions and the GDPR](https://arxiv.org/pdf/1711.00399.pdf) [1]

- [ ] DiCE, [Explaining machine learning classifiers through diverse counterfactual explanations]((https://arxiv.org/pdf/1905.07697.pdf)[2]

- [ ] TBD,  

- [ ] TBD,

# Installation

**Installation Requirements**
- Pytorch >= 1.0+
- Python >= 3.6

CF only supports Python3+. If you want to install the latest version, run the command in the root folder
```
git clone https://github.com/wangyongjie-ntu/Counterfactual-Explanations-Pytorch

cd Counterfactual-Explanations-Pytorch

pip install -e .
```

For the stable version installation, you can directly install it from PyPI via

```
pip install cf-explanations
```


# Getting Start

To be done.

# References

[1] Wachter, Sandra, Brent Mittelstadt, and Chris Russell. "Counterfactual explanations without opening the black box: Automated decisions and the GDPR." Harv. JL & Tech. 31 (2017): 841. [https://arxiv.org/pdf/1711.00399.pdf](https://arxiv.org/pdf/1711.00399.pdf)

[2] Mothilal, Ramaravind K., Amit Sharma, and Chenhao Tan. "Explaining machine learning classifiers through diverse counterfactual explanations." Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency. 2020.
[https://arxiv.org/pdf/1905.07697.pdf](https://arxiv.org/pdf/1905.07697.pdf)

# Acknowledge

If any questions on this repo, please kindly let me know by email(yongjie.wang@ntu.edu.sg)


# SEC-former
SEC-former is a transformer to perform Document Format Reconstruction of 10-K SEC filings

The analysis of financial reports is a crucial task for investors and regulators, especially the mandatory annual reports (10-K) required by the SEC (Securities and Exchange Commission) that provide crucial information about a public company in the American stock market. Although SEC suggests a specific document format to uniform and simplifies the analysis, in recent years, several companies have introduced their own format and organization of the contents making human-based and automatic knowledge extraction inherently more difficult.

We propose SEC-former, an Autoregressive language model that has been fine-tuned to classify paragraphs into one of the items suggested by the SEC guidelines.
The model has been fine-tuned with a bidirectional mechanism and using thousands of 10-Ks in the years between 2011-201.

# Document Format Reconstruction with Neural Language Models for Automatic Analysis of Financial SEC Filings
 | **Autoregressive Transformer** | XLNet Large |
|:---:|:---:|
| **BiLSTM's architecture** | Two stacked LSTMs with 1024 units for each one,\\and a dropout of 0.2 between them. |
| **DNN's architecture** | A neural network with two hidden layers of 512 and\\256 units, and a dropout of 0.2 after these layers. |
| **Hidden and\\Attention dropout** | 0.1 |
| **CLS token dropout** | 0.0 |
| **Batch size** | 32 |
| **Learning rate** | 5e-5 |

# Download the model
Please, find the h5 version of the Keras model at http://www.ce.unipr.it/people/lombardo/SEC-Former.h5


## Pre-requisites
In order to use this model, you need four python libraries, which are **numpy**, **tensorflow**, **transformers** and **sentencepiece**. They can be installed with the following commands.
```
pip install numpy
pip install tensorflow
pip install transformers
pip install sentencepiece
```

## Usage 
Here is how to simply use our model for inference, i.e. to classify paragraphs based on standard 10-Ks items provided by SEC.
First of all, import ```inference_utils``` file and load the model:
```python
from inference_utils import get_model, inference

model = get_model(path='full_model/XLNet_BiLSTM_DNN.h5')
```
The next step simply is the inference:
```python
text = """We aim to deliver open software and hardware platforms with industry-defining standards.
        Around the globe, companies are building their networks, systems, and solutions on open 
        standards-based platforms. Intel has helped set the stage for this movement, with our 
        historic contributions in developing standards such as CXL, Thunderbolt, and PCle. We also 
        contributed to the design, build, and validation of new open-source products in the industry 
        such as Linux, Android, and others. The world's developers constantly innovate and expand the 
        capabilities of these open platforms while increasing their stability, reliability, and security. 
        In addition, microservices have enabled the development of flexible, loosely coupled services 
        that are connected via APIs to create end-to-end processes. We use industry collaboration, 
        co-engineering, and open-source contributions to accelerate software innovation. Through our 
        oneAPI initiative, developers use a unified language across CPUs, GPUs, and FPGAs to cut down on 
        development time and to enhance productivity. We also deliver a steady stream of open-source code 
        and optimizations for projects across virtually every platform and usage model. We are committed 
        to co-engineering and jointly designing, building, and validating new products with software 
        industry leaders to accelerate mutual technology advancements and help new software and hardware 
        work better together. Our commitment extends to developers through our developer-first approach 
        based on openness, choice, and trust"""
inference(text=text, model=model)
```
>Predicted item: Item 1.
>
>Scores on each item:
>
>Item 1: 0.9128&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 7: 0.0504&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 1a: 0.0200&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 6: 0.0086&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 2: 0.0015&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 5: 0.0010
>
>Item 8: 0.0010&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 15: 0.0009&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 7a: 0.0006&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 9: 0.0006&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 9b: 0.0005&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 13: 0.0005
>
>Item 12: 0.0004&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 9a: 0.0003&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 11: 0.0003&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 3: 0.0003&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 4: 0.0002&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 10: 0.0001

```python
text = """We are subject to the risks of product defects, errata, or other product issues.
        From time to time, we identify product defects, errata (deviations from published
        specifications), and other product issues which can result from problems in our
        product design or our manufacturing and assembly and test processes.
        Components and products we purchase or license from third-party suppliers, or
        gain through acquisitions, can also contain defects. Product issues also
        sometimes result from the interaction between our products and third-party
        products and software. We face risks if products that we design, manufacture, or
        sell, or that include our technology, cause personal injury or property damage,
        even where the cause is unrelated to product defects or errata. These risks may
        increase as our products are introduced into new devices, market segments,
        technologies, or applications, including transportation, autonomous driving,
        healthcare, communications, financial services, and other industrial, critical
        infrastructure. These costs could be large and may increase expenses and lower 
        gross margin, and/or result in delay or loss of revenue"""

inference(text=text, model=model)
```
>Predicted item: Item 1a.
>
>Scores on each item:
>
>Item 1a: 0.9740&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 1: 0.0078&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 6: 0.0042&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 3: 0.0038&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 2: 0.0023&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 9b: 0.0022
>
>Item 7: 0.0019&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 7a: 0.0010&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 15: 0.0009&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 9: 0.0008&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 5: 0.0002&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 8: 0.0002
>
>Item 9a: 0.0002&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 12: 0.0001&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 13: 0.0001&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 10: 0.0001&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 11: 0.0001&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 4: 0.0001
```python
text = """Operating income increased $37 million, driven by higher revenue due to recovery 
        in the embedded and communications market segments from COVID-19 lows, partially 
        offset by a decrease in the cloud market segment. Our total revenue grew from 
        $62.8 billion in 2017 to $79.0 billion in 2021, representing 6% CAGR. In 2021, revenue 
        was $79.0 billion, up $1.2 billion, or 1%, from 2020. CCG revenue grew 1% due to continued
        strength in notebook demand and recovery in desktop demand, partially offset by lower 
        notebook ASPs due to strength in the consumer and education market segments. CCG adjacent 
        revenue decreased primarily due to the continued ramp down from the exit of our 5G smartphone 
        modem and Home Gateway Platform businesses. Our effective tax rate decreased in 2021 compared 
        to 2020, primarily driven by one-time tax benefits due to the restructuring of certain non-US 
        subsidiaries as well as a higher proportion of our income in non-US jurisdictions. As a result 
        of the restructuring, we established deferred tax assets and released the valuation allowances 
        of certain foreign deferred tax assets."""

inference(text=text, model=model)
```
>Predicted item: Item 7.
>
>Scores on each item:
>
>Item 7: 0.9239&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 6: 0.0307&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 1: 0.0147&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 7a: 0.0074&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 5: 0.0048&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 4: 0.0030
>
>Item 1a: 0.0026&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 9b: 0.0025&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 8: 0.0023&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 11: 0.0019&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 3: 0.0018&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 2: 0.0018
>
>Item 13: 0.0007&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 15: 0.0005&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 9: 0.0005&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 12: 0.0004&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 10: 0.0003&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Item 9a: 0.0003



## BibTeX entry and citation info

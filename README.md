


HP# SeqControl

SeqControl is a Python library for in-silico data-driven control of dynamical systems. It contains utilities for training, testing, and validation of sequence models, including Reservoir Computing approaches and State-Space Model Approaches.

## Resources

### Documentation

You can find the documentation of the library's features [here](https://iaa-master-thesis-dpe.readthedocs.io/en/latest/).

### Thesis

You can find information about the background and application of SeqControl in my thesis.


## Installing SeqControl

This tutorial guides you through installing **SeqControl** on your local machine, whether you want to use it as a dependency in another project or contribute to its development.

---

### Prerequisites

Before installing, ensure you have the following installed on your system:

* **Python 3.8** or higher
* **Git** command-line tool
* **pip** (Python package manager)

---

### Option 1: Quick Install via Git & pip

If you simply want to use `SeqControl` in your Python scripts without modifying the source code, you can install it directly from GitHub using `pip`.

1. Open your terminal or command prompt.
2. Run the following command:


```
pip install git+[https://github.com/ilyes0702/IAA_Master_Thesis_DPE.git](https://github.com/ilyes0702/IAA_Master_Thesis_DPE.git)
```
To install a specific version or release tag in the future, append @tag_name to the URL:
```
pip install git+https://github.com/ilyes0702/IAA_Master_Thesis_DPE.git@v1.0.0
```



### Option 2: Developer / Local Installation
If you want to modify the source code, build new features, or run tests, set up an editable local installation.


Step 1: Clone the Repository
Download a local copy of the repository using git:

```
git clone https://github.com/ilyes0702/IAA_Master_Thesis_DPE.git
cd IAA_Master_Thesis_DPE
```



Step 2: Install in Editable Mode


Install the package using the -e flag. This links your environment to the source files in src/seqControl so that any changes you make to the code take effect immediately:

```
pip install -e .
```


Tip: If you plan to run the test suite or build documentation locally, install the developer dependencies:
bash
Kopieren

```pip install -r requirements.txt```





Verifying Your Installation
To verify that SeqControl is correctly installed:
Launch a Python interactive shell:


```
import seqControl
print("SeqControl successfully installed!")
```
## How to use SeqControl

## Contact

If there are any questions, feel free to contact me via e-mail.
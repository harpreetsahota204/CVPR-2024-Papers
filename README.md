# 📝📊 CVPR 2024 Accepted Papers Dataset

This code creates a `fiftyone` dataset contains the accepted papers for the 2024 Conference on Computer Vision and Pattern Recognition (CVPR). 

The CVPR 2024 conference received 11,532 valid paper submissions, out of which only 2,719 were accepted. 

This results in an overall acceptance rate of about 23.6%. However, the dataset currently includes 2,379 papers, which represent those for which we were able to easily find papers.

If you're going to be at CVPR 2024, be sure to come say "Hi!". Here's where you can find me 👇🏼

<img src="4.10.24_CVPR24_Social_AV.png" width="50%">

## 📄 Dataset Details

### 📚 Dataset Description

- **Curated by:** [Harpreet Sahota, Hacker-in-Residence at Voxel51](https://huggingface.co/harpreetsahota)
- **Language(s):** English (en)
- **License:** [CC-BY-ND-4.0](https://spdx.org/licenses/CC-BY-ND-4.0)

### 🗂️ Dataset Structure

The dataset includes the following information for each paper:

- **🖼️ Image of the first page of the paper**
- **📌 `title`:** The title of the paper
- **👨‍🔬👩‍🔬 `authors_list`:** The list of authors
- **📄 `abstract`:** The abstract of the paper
- **🔗 `arxiv_link`:** Link to the paper on arXiv
- **🔗 `other_link`:** Link to the project page, if available
- **🏷️ `category_name`:** The primary category of the paper according to the [arXiv taxonomy](https://arxiv.org/category_taxonomy)
- **🏷️ `all_categories`:** All categories the paper falls into, as per the arXiv taxonomy
- **🔑 `keywords`:** Keywords extracted using GPT-4o

## 🎯 Uses

This dataset can be used for various purposes, including:

- Analyzing trends in research presented at CVPR 2024

- Studying the distribution of topics and methods in computer vision research

- Developing new machine learning models and techniques using the provided abstracts and keywords

## 🛠️ Dataset Creation

The dataset was created using the following steps:

1. **Scrape the CVPR 2024 website** for the list of accepted papers.

2. **Search for each paper's abstract on arXiv** using DuckDuckGo.

3. **Extract abstracts, categories, and download PDFs** using arXiv.py, a Python wrapper for the arXiv API.

4. **Convert the first page of each paper to an image** using `pdf2image`.

5. **Extract keywords from each abstract** using GPT-4o.

### 🚀 Code Execution Order

To build this dataset, run the following scripts in order:

1. `scrape_cvpr_for_papers.py`
2. `get_arxiv_data.py`
3. `pre_process_csv.py`
4. `create_fiftyone_dataset.py`

## 🙏 Acknowledgements

We gratefully acknowledge the support from the CVPR conference organizers, arXiv for providing an accessible API, and the contributors who made this dataset possible.

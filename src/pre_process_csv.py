import ast
import os

import re
import pandas as pd
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

taxonomy_map = {
    'cs.AI': 'Artificial Intelligence',
    'cs.AR': 'Hardware Architecture',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering, Finance, and Science',
    'cs.CG': 'Computational Geometry',
    'cs.CL': 'Computation and Language',
    'cs.CR': 'Cryptography and Security',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.DB': 'Databases',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata Theory',
    'cs.GL': 'General Literature',
    'cs.GR': 'Graphics',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LG': 'Machine Learning',
    'cs.LO': 'Logic in Computer Science',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.MS': 'Mathematical Software',
    'cs.NA': 'Numerical Analysis',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.OH': 'Other Computer Science',
    'cs.OS': 'Operating Systems',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SC': 'Symbolic Computation',
    'cs.SD': 'Sound',
    'cs.SE': 'Software Engineering',
    'cs.SI': 'Social and Information Networks',
    'cs.SY': 'Systems and Control',
    'eess.AS': 'Audio and Speech Processing',
    'eess.IV': 'Image and Video Processing',
    'eess.SP': 'Signal Processing',
    'eess.SY': 'Systems and Control',
    'math.OC': 'Optimization and Control'
 }

template = """Given the title, abstract of a paper accepted to the 2024 Computer Vision and Pattern Recognition (CVPR) conference and a list of provided and approved keywords, your task is to select the appropriate keywords for an abstract based on the title and abstract of the paper presented at the 2024 Computer Vision and Pattern Recognition (CVPR) conference.

The title for this paper is: {title}

The abstract for this paper is: {abstract}

Return a list consisting of the most appropriate keywords for this paper. A paper can have multiple keywords. Select ONLY keywords from the following list of provided and approved keywords.

The provided and approved keywords are:
```
- 3D vision (multi-view, sensors, and single images)
- Autonomous systems (driving, robotics, and embodied vision)
- Deep learning architectures and techniques
- Self-supervised, unsupervised, and semi-supervised learning
- Meta-learning, transfer learning, and continual learning
- Efficient and scalable vision
- Datasets, benchmarks, and evaluation methods
- Low-level vision
- Object recognition, detection, and segmentation
- Scene analysis and understanding
- Generative AI (synthetic datasets, GANs, LLMs, and diffusion models)
- Image and video generation and manipulation
- Multimodal models and vision-language models
- Large multimodal models and prompting techniques
- Medical imaging and biological vision
- Remote sensing and photogrammetry
- Document analysis and understanding
- Biometrics and human analysis
- Computational imaging and physics-based vision
- Vision applications for social good and ethics
- Vision systems and graphics integration
```

Remember you MUST ONLY select keywords from the provided and approved keywords above. Your selected keywords MUST be based on the title and abstract of a paper. Under no circumstances are you allowed to select a keyword that does not appear 
in the list of provided and approved keywords. A paper can have multiple keywords.

Think step-by-step and understand the title and abstract before selecting the keywords.

\n{format_instructions}
"""

valid_keyword_list = [
    "3D vision (multi-view, sensors, and single images)",
    "Autonomous systems (driving, robotics, and embodied vision)",
    "Deep learning architectures and techniques",
    "Self-supervised, unsupervised, and semi-supervised learning",
    "Meta-learning, transfer learning, and continual learning",
    "Datasets, benchmarks, and evaluation methods",
    "Low-level vision",
    "Object recognition, detection, and segmentation",
    "Scene analysis and understanding",
    "Generative AI (synthetic datasets, GANs, LLMs, and diffusion models)",
    "Image and video generation and manipulation",
    "Multimodal models and vision-language models",
    "Large multimodal models and prompting techniques",
    "Deep learning architectures and techniques",
    "Medical imaging and biological vision",
    "Remote sensing and photogrammetry",
    "Efficient and scalable vision",
    "Document analysis and understanding",
    "Biometrics and human analysis",
    "Image and video synthesis and generation",
    "Medical and biological vision",
    "Vision applications for social good and ethics",
    "Photogrammetry and remote sensing",
    "Computational imaging and physics-based vision",
    "Vision systems and graphics integration",
]


# Define the format instructions
format_instructions = CommaSeparatedListOutputParser().get_format_instructions()

# Create the prompt template
prompt = PromptTemplate(
    template="{title}\n\n{abstract}\n\n{format_instructions}",
    input_variables=["title", "abstract"],
    partial_variables={"format_instructions": format_instructions},
)

# Instantiate the model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Create the output parser
output_parser = CommaSeparatedListOutputParser()

# Define the chain
chain = prompt | model | output_parser

# Define the extract_keywords function
def extract_keywords(row):
    try:
        result = chain.invoke({
            "title": row['Title'],
            "abstract": row['summary']
        })
        return result
    except Exception as e:
        print(f"Error processing row: {e}")
        return []

def clean_keywords(keywords):
    # Convert valid_keyword_list to lowercase for case-insensitive comparison
    valid_keywords_lower = [keyword.lower() for keyword in valid_keyword_list]
    
    # Remove unwanted characters, convert to sentence case, and filter invalid keywords
    cleaned_keywords = [
        keyword.replace('```', '').replace('python\n', '').strip().capitalize() 
        for keyword in keywords 
        if keyword.lower() in valid_keywords_lower
    ]
    return cleaned_keywords

# Define the main function
def main(input_csv, output_csv):
    # Load the CSV file
    cvpr_papers = pd.read_csv(input_csv)
    # Regular expression pattern to find URLs, excluding trailing periods
    url_pattern = r'(https?://[^\s]+)\.?'

    # Extract the first URL found in each summary and set it in the 'other_link' column
    cvpr_papers['other_link'] = cvpr_papers['summary'].str.extract(url_pattern, expand=False)

    # Assuming taxonomy_map and map_categories are defined elsewhere
    cvpr_papers['category_name'] = cvpr_papers['primary_category'].map(taxonomy_map)
    cvpr_papers['category_name'] = cvpr_papers['category_name'].fillna('')

    # Apply the map_categories function to the 'categories' column
    cvpr_papers['all_categories'] = cvpr_papers['categories'].apply(map_categories)

    # Process authors
    cvpr_papers['authors_list'] = cvpr_papers['Authors'].apply(lambda x: [name.strip() for name in x.split('·')])

    # Generate image path
    cvpr_papers['image_path'] = cvpr_papers['pdf_path'].apply(lambda x: "arxiv_pdfs/first_page_images/" + x.replace(".pdf", ".png") if pd.notnull(x) else None)

    # Filter rows with non-null image paths
    cvpr_papers = cvpr_papers[cvpr_papers['image_path'].notnull()]

    # Fill null values in 'other_link'
    cvpr_papers['other_link'] = cvpr_papers['other_link'].fillna('')

    # Filter rows where the image file exists
    cvpr_papers = cvpr_papers[cvpr_papers['image_path'].apply(lambda x: os.path.exists(x))]

    # Extract keywords
    cvpr_papers['keywords'] = cvpr_papers.apply(extract_keywords, axis=1)
    cvpr_papers['keywords'] = cvpr_papers.apply(extract_keywords, axis=1)

    # Apply the cleaning function to the 'keywords' column
    cvpr_papers['keywords'] = cvpr_papers['keywords'].apply(clean_keywords)

    # Convert columns to literal list
    columns_to_convert = ['all_categories', 'authors_list', 'keywords']
    for column in columns_to_convert:
        cvpr_papers[column] = cvpr_papers[column].apply(ast.literal_eval)

    # Save the processed CSV
    cvpr_papers.to_csv(output_csv, index=False)
    print(f"Processed CSV saved to {output_csv}")

# Execute the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CVPR papers CSV.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_csv", help="Path to save the processed CSV file")
    args = parser.parse_args()

    main(args.input_csv, args.output_csv)
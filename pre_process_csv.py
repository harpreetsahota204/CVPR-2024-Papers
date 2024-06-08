import pandas as pd
import ast
import os
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


template = """Extract a maximum of ten specific keywords from the title and abstract of a 2024 Computer Vision and Pattern Recognition (CVPR) conference paper, 
related to deep learning topics such as computer vision, multimodal models, large language models, vision-language models, diffusion, transformers, and other relevant areas.

Include any referenced artificial intelligence, computer vision, or deep learning techniques; such as datasets, or models as keywords.

These keywords MUST be recognizable by current deep learning researchers, engineers, practitioners, and students.

The title for this paper is: {title}

The abstract for this paper is: {abstract}

\n{format_instructions}
"""

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

# Define the main function
def main(input_csv, output_csv):
    # Load the CSV file
    cvpr_papers = pd.read_csv(input_csv)

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

    # Convert columns to list
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
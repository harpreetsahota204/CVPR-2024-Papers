import fiftyone as fo
import fiftyone.core.fields as fof
import os

def create_cvpr_fiftyone_dataset(name) -> fo.Dataset:
	"""
	Creates schema for a COYO-Tiny FiftyOne dataset.
	"""
	dataset = fo.Dataset(name=name, persistent=True, overwrite=True)

	dataset.add_sample_field(
		'arXiv_link', 
		fof.StringField,
		description='Link to paper abstract on arxiv'
		)

	dataset.add_sample_field(
		'other_link', 
		fof.StringField,
		description='Link to other resource found for paper'
		)
	
	dataset.add_sample_field(
		'title', 
		fof.StringField,
		description='The title of the paper'
		)

	dataset.add_sample_field(
		'abstract', 
		fof.StringField,
		description='The abstract of the paper'
		)

	dataset.add_sample_field(
		'authors_list', 
		fof.ListField,
		subfield=fof.StringField,
		description='The authors listed on the paper'
		)

	dataset.add_sample_field(
		'keywords', 
		fof.ListField,
		subfield=fof.StringField,
		description='Keywords associated with this paper, extracted using GPT-4o'
		)
	
	dataset.add_sample_field(
		'category_name', 
		fof.StringField,
		description='The category of the paper'
		)

	dataset.add_sample_field(
		'all_categories', 
		fof.ListField,
		subfield=fof.StringField,
		description='The authors listed on the paper'
		)
	
	return dataset


def create_fo_sample(image: dict) -> fo.Sample:
    """
    Creates a FiftyOne Sample from a given image entry with metadata and custom fields.

    Args:
        image (dict): A dictionary containing image data including the path and other properties.

    Returns:
        fo.Sample: The FiftyOne Sample object with the image and its metadata.
    """
    filepath = image.get('image_path')
    
    if not filepath:
        return None

    arXiv_link = image.get('arXiv_link')
    other_link = image.get('other_link')
    title = image.get('Title')
    abstract = image.get('summary')
    keywords = image.get('keywords')
    authors_list = image.get('authors_list')
    category_name = image.get('category_name')
    all_categories = image.get('all_categories')

    sample = fo.Sample(
        filepath=filepath,
        arXiv_link=arXiv_link,
        other_link=other_link,
        title=title,
        abstract=abstract,
        keywords=keywords,
        authors_list=authors_list,
        category_name=category_name,
        all_categories=all_categories
    )

    return sample

def add_samples_to_fiftyone_dataset(
	dataset: fo.Dataset,
	samples: list
	):
	"""
	Creates a FiftyOne dataset from a list of samples.

	Args:
		samples (list): _description_
		dataset_name (str): _description_
	"""
	dataset.add_samples(samples, dynamic=True)
	dataset.add_dynamic_sample_fields()

def main(input_csv, dataset_name):
    # Load the CSV file
    cvpr_papers = pd.read_csv(input_csv)

    # Convert DataFrame to list of dictionaries
    images = cvpr_papers.to_dict(orient='records')

    # Create the FiftyOne dataset
    dataset = create_cvpr_fiftyone_dataset(dataset_name)

    # Create FiftyOne samples
    samples = [create_fo_sample(image) for image in images]

    # Add samples to the dataset
    add_samples_to_fiftyone_dataset(dataset, samples)

    print(f"Dataset '{dataset_name}' created with {len(samples)} samples.")

# Execute the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a CVPR FiftyOne dataset from a CSV file.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("dataset_name", help="Name of the FiftyOne dataset to create")
    args = parser.parse_args()

    main(args.input_csv, args.dataset_name)
	
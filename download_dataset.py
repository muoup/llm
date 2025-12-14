
import argparse
from datasets import load_dataset
from tqdm import tqdm

def download_and_format_dataset(num_stories, output_file):
    """
    Downloads the TinyStories dataset, formats it, and saves it to a file.

    Args:
        num_stories (int): The number of stories to download.
        output_file (str): The path to the output file.
    """
    print("Loading TinyStories dataset from HuggingFace...")
    # Load the dataset with streaming to avoid downloading the whole dataset at once
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    print(f"Downloading and formatting {num_stories} stories...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Use tqdm for a progress bar
        progress_bar = tqdm(total=num_stories, desc="Processing stories")
        
        count = 0
        for story in dataset:
            if count >= num_stories:
                break
            
            text = story["text"]
            f.write(text)
            f.write("<|endoftext|>")
            
            count += 1
            progress_bar.update(1)
            
    progress_bar.close()
    print(f"\nSuccessfully downloaded and formatted {num_stories} stories.")
    print(f"Dataset saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and format the TinyStories dataset.")
    parser.add_argument(
        "--num_stories",
        type=int,
        default=100000,
        help="The number of stories to download."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="tinystories_100k.txt",
        help="The path to the output file."
    )
    
    args = parser.parse_args()
    
    download_and_format_dataset(args.num_stories, args.output_file)

import os
import json
from google.cloud import storage
from google.protobuf.json_format import MessageToDict

# Set up GCP credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\ocr\vision_key.json"

# Storage Client setup
storage_client = storage.Client()
bucket_name = "lms_ocr"
destination_prefix = "pdf_res"


def clean_text(raw_text):
    """Clean text by removing unnecessary characters."""
    return raw_text.replace("\n", " ").strip()


def extract_text_from_json(json_data):
    """Extract text from the JSON response."""
    extracted_texts = []
    if "responses" in json_data:
        for response in json_data["responses"]:
            full_text_annotation = response.get("fullTextAnnotation", {})
            extracted_texts.append(full_text_annotation.get("text", ""))
    return clean_text(" ".join(extracted_texts))


def process_json_files():
    """Process all JSON files from a GCS directory and extract text."""
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=destination_prefix)

    combined_texts = []

    for blob in blobs:
        if blob.name.endswith(".json"):
            try:
                json_data = blob.download_as_bytes()
                json_content = json_data.decode("utf-8")
                json_parsed = json.loads(json_content)
                text_output = extract_text_from_json(json_parsed)
                if text_output:
                    combined_texts.append(text_output)
            except Exception as e:
                print(f"Error parsing {blob.name}: {e}")
                continue

    # Join and print the final result
    full_text = " ".join(combined_texts)
    print("Final Extracted Text:\n", full_text)


if __name__ == "__main__":
    process_json_files()

"""
Generate eval_pairs.json using GPT-4o-mini vision.

Selects 200 images from data/holdout/, sends each to GPT-4o-mini, and saves
a natural language search query per image. Results are written incrementally
so progress is not lost if the script is interrupted.

Usage:
    python scripts/generate_eval_pairs.py

Requires:
    OPENAI_API_KEY in .env
"""

import base64
import json
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from utils.find_root import find_project_root

load_dotenv()
PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "data" / "index"
OUTPUT_FILE = PROJECT_ROOT / "data" / "eval_pairs.json"
SEED = 42
N_PAIRS = 200
MODEL = "gpt-4o-mini"

PROMPT = """Look at this image carefully.

Write a single natural language search query that a real user would type into an image search box to find this specific image.

Rules:
- Be specific — the query should match this image and ideally no other image
- Write it the way a person would actually search, not as a technical description
- Include key subjects, actions, setting, and any distinctive details
- Do NOT use generic words like "photo", "image", "picture"
- Return ONLY the query string, nothing else

Example of a bad query: "dog"
Example of a good query: "golden retriever sitting next to a red fire hydrant on a suburban street"
"""


def encode_image(image_path: Path) -> str:
    """
    Read an image file and return it as a base64-encoded string.

    Args:
        image_path (Path): Absolute path to the JPEG image file.

    Returns:
        str: Base64-encoded contents of the image, ready to embed in an API request.
    """
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def generate_query(client: OpenAI, image_path: Path) -> str:
    """
    Send an image to GPT-4o-mini and return a natural language search query describing it.

    Args:
        client (OpenAI): Authenticated OpenAI client instance.
        image_path (Path): Absolute path to the JPEG image file.

    Returns:
        str: A single search query string, e.g. "two dogs playing in a snow-covered park".
    """
    b64 = encode_image(image_path)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        max_tokens=60,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    images = sorted(DATA_DIR.glob("*.jpg"))
    if not images:
        print(f"ERROR: No images in {DATA_DIR}. Run split_data.py first.")
        sys.exit(1)

    random.seed(SEED)
    selected = random.sample(images, N_PAIRS)

    # Load existing progress if the script was interrupted
    existing: dict[str, dict] = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for entry in json.load(f):
                existing[entry["image_id"]] = entry
        print(f"Resuming — {len(existing)} pairs already done.\n")

    client = OpenAI()
    pairs = list(existing.values())
    errors = 0

    for image_path in tqdm(selected, unit="img"):
        image_id = image_path.stem  # e.g. "000000012345"

        if image_id in existing:
            continue  # already generated, skip

        try:
            query = generate_query(client, image_path)
            pairs.append(
                {
                    "query": query,
                    "image_id": image_id,
                    "image_path": f"index/{image_path.name}",
                    "source": "coco_val2017",
                }
            )
            # Write after every image so progress is never lost
            with open(OUTPUT_FILE, "w") as f:
                json.dump(pairs, f, indent=2)

        except Exception as e:
            tqdm.write(f"[WARN] {image_path.name}: {e}")
            errors += 1

        # Avoid hitting rate limits
        time.sleep(0.3)

    print(f"\nDone — {len(pairs)} pairs written to {OUTPUT_FILE}")
    if errors:
        print(f"  {errors} errors — re-run the script to retry failed images.")
    print("\nNext step: review eval_pairs.json and edit any queries that feel wrong.")


if __name__ == "__main__":
    main()

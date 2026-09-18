from pathlib import Path
import csv
import json

wsasl = "data/WLASL_v0.3.json"
out_dir = "data/combined_dict.json"
csv_path = "data/ASL_Citizen/splits/test.csv"
csv_matches = "data/aslc_matches.csv"

def create_json():
    data = []

    with open(csv_matches, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        curr_obj = {"gloss" : "", "instances" : []}

        for row in reader:
            if row["wlasl_gloss"] != curr_obj["gloss"]:
                data.append(curr_obj)
                curr_obj = {"gloss" : row["wlasl_gloss"], "instances" : []}

            curr_obj["instances"].append({
                "split" : row["aslc_split"],
                "video_id" : row["aslc_video_file"][:-4]
            })

    with open(out_dir, "w") as file:
        json.dump(data, file, indent=4)

def combine_wlasl():
    combined_words = set()

    with open(csv_matches, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            combined_words.add(row["wlasl_gloss"])

    to_combine = {}

    with open(wsasl, mode = 'r', encoding='utf-8') as f:
        data = json.load(f)

        for entry in data:
            if entry["gloss"] in combined_words:
                to_combine[entry["gloss"]] = [
                    {"split" : video["split"], "video_id" : video["video_id"]}
                    for video in entry["instances"]
                ]

    with open(out_dir, 'r+', encoding='utf-8') as file:
        data = json.load(file)

        for i, entry in enumerate(data):
            if entry["gloss"] in to_combine:
                data[i]["instances"] += to_combine[entry["gloss"]]

        json.dump(data, file, indent=4)


    

if __name__ == "__main__":
    combine_wlasl()

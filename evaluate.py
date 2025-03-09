import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse

captions = []
ground_truth = []
predicted = []
confidence_scores = []
inference_time_list = []
candidates = []
entities = []

# result_dir = 'result_new_ic_dont_use_img_matched_tags'
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='result')
parser.add_argument('--skip_non_candidates', '-s', action='store_true')
args = parser.parse_args()
result_dir = args.result_dir

result_dir2 = 'result_4o_gemini_94'
result_json_list = os.listdir(result_dir)

incorrect_index = []
empty_evidence_count = 0
for item in result_json_list:
    try:
        result_json_dir = os.path.join(result_dir, item)
        with open(result_json_dir, 'r', encoding='utf-8') as f:
            result_json = json.load(f)
            
            if result_json['internal_check']['visual_candidates'] == []:
                empty_evidence_count += 1
                if args.skip_non_candidates:
                    continue
                if result_json['ground_truth'] != result_json['final_result']['OOC']:
                    print(result_json_dir)
            # else:
            #     continue
            
            # if result_json['external_check']['text_evidences'] == []:
            #     empty_evidence_count += 1
            #     if args.skip_non_candidates:
            #         continue
            #     if result_json['ground_truth'] != result_json['final_result']['OOC']:
            #         print(result_json_dir)
            # else:
            #     continue

            
            captions.append(result_json['caption'])
            ground_truth.append(result_json['ground_truth'])
            predicted.append(1 if result_json['final_result']['OOC'] else 0)
            confidence_scores.append(result_json['final_result']['confidence_score'])
            inference_time_list.append(result_json['inference_time'])
            # candidates.append(result_json['external_check']['text_evidences'])
            # entities.append(result_json['internal_check']['visual_entities'])
            if result_json['ground_truth'] != result_json['final_result']['OOC']:
                # print(result_json_dir)
                # print(os.path.join(result_dir2, item))
                
                # extract index from result_json_dir
                index = result_json_dir.split('_')[-1].split('.')[0]
                incorrect_index.append(int(index))
    except Exception as e:
        print(item)
        print(f"ERROR: {e}")
        raise

# save incorrect_index to a file
incorrect_index.sort()
print(incorrect_index)
with open('src/incorrect_index.txt', 'w') as f:
    for index in incorrect_index:
        f.write(str(index) + '\n')

print(f"Empty evidence count: {empty_evidence_count}")
print(len(captions))
print(len(ground_truth))
print(len(predicted))
print(len(inference_time_list))
# print(len(candidates))
# print(len(entities))

df = pd.DataFrame({
    'caption': captions,
    'ground_truth': ground_truth,
    'predicted': predicted,
    'confidence_score': confidence_scores,
    'inference_time': inference_time_list,
    # 'candidates': candidates,
    # 'entities': entities
})
df['adjusted_predicted'] = df.apply(
    lambda row: row['predicted'] if row['confidence_score'] >= 8 else 1 - row['predicted'], axis=1
)
adjusted_predicted = df['adjusted_predicted'].tolist()
df[(df['ground_truth'] != df['predicted'])]

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

class_names = ["NOOC", "OOC"]

# Generate Classification Report
report = classification_report(ground_truth, predicted, target_names=class_names)
print("\nClassification Report:")
print(report)
print("################################")

# calculate per-class accuracy
cm = confusion_matrix(ground_truth, predicted)
class_accuracies = cm.diagonal() / cm.sum(axis=1)
print("Per-Class Accuracy:")
for class_name, acc in zip(class_names, class_accuracies):
    print(f"{class_name}: {acc:.4f}")
print("################################")

# Calculate average inference time
average_time = sum(inference_time_list) / len(inference_time_list)
print(f"Average Inference Time: {average_time:.6f} seconds")


# Generate Classification Report
report = classification_report(ground_truth, adjusted_predicted, target_names=class_names)
print("\nClassification Report:")
print(report)
print("################################")

# calculate per-class accuracy
cm = confusion_matrix(ground_truth, adjusted_predicted)
class_accuracies = cm.diagonal() / cm.sum(axis=1)
print("Per-Class Accuracy:")
for class_name, acc in zip(class_names, class_accuracies):
    print(f"{class_name}: {acc:.4f}")
print("################################")
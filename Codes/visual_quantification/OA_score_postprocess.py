import argparse
import math
from collections import Counter
import datasets

def compute_tfidf(tags):
    total_instances = len(tags)

    tag_document_frequency = Counter()
    for tag_list in tags:
        unique_tags = set(tag_list)  # 每个实例的标签去重
        for tag in unique_tags:
            tag_document_frequency[tag] += 1

    tag_idf = {}
    for tag, df in tag_document_frequency.items():
        tag_idf[tag] = math.log(total_instances / df)

    return tag_idf


def compute_instance_score(tag_list, tag_idf):
    # 计算实例的tag分数
    tag_counts = Counter(tag_list)
    score = 0
    for tag, count in tag_counts.items():
        score += count * tag_idf.get(tag, 0)
    return score


def compute_tag_richness(instance):
    global  tag_idf
    score = compute_instance_score(instance["dino_labels"], tag_idf)
    instance['OA score'] = score
    return instance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    data = datasets.load_from_disk(args.data_path)
    tags = data["dino_labels"]
    tag_idf = compute_tfidf(tags)
    data = data.map(compute_tag_richness)
    data.save_to_disk(args.save_path)


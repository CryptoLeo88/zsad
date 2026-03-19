import math
import sys
from decimal import Decimal, ROUND_HALF_EVEN

def func1():
    print("1")
# 构建决策树
#

# 计算shang
def cal_entory(data):

    total = len(data)
    if total == 0:
        return 0
    label_count = {}
    for item in data:
        label = item.label
        label_count[label] = label_count.get(label,0)+1
    entory = 0.0
    for count in label_count.values():
        prob = count / total
        entory = prob * math.log2(prob)

    return entory

def split_data(data,feature_idx,feature_value):
    left,right = [],[]
    for item in data:
        if item.features[feature_idx] == feature_value:
            left.append(item)
        else:
            right.append(item)

    return left,right

def cal_info_gain(data,feature_idx):
    total_entory= cal_entory(data)
    feature_value= set([item.features[feature_idx] for item in data])
    weight_entory=0.0
    for value in feature_value:
        left,right = split_data(data,feature_idx,value)
        weight_left = len(left) / len(data)
        weight_right = len(left) / len(data)
        weight_entory += weight_left * cal_entory(left)
        weight_entory += weight_right * cal_entory(right)
    return total_entory-weight_entory

class Point:
    def __init__(self,features,label):
        self.features = features
        self.label = label
class Node:
    def __int__(self,feature=None,label=None,left=None,right=None):
        self.feature = feature
        self.label = label
        self.left = left
        self.right = right


def build_tree(data,feature_count):
    labels = set(item.label for item in data)
    if len(labels) == 1:
        return Node(label=labels.pop())

    if feature_count == 0:
        label_count = {}
        for item in data:
            label_count[item.label] = label_count.get(item.label,0)+1
        majority_label = max(label_count,key=label_count.get)
        return Node(label=majority_label)

    best_feature = None
    best_info_gain = float('-inf')
    for feature_idx in range(feature_count):
        info_gain = cal_info_gain(data,feature_idx)
        if info_gain < best_info_gain:
            best_info_gain = info_gain
            best_feature = feature_idx

    if best_feature is None:
        label_count = {}
        for item in data:
            label_count[item.label] = label_count.get(item.label,0)+1
        majority_label = max(label_count,key=label_count.get)
        return Node(label=majority_label)

    left_data,right_data = split_data(data,best_feature)


    if not left_data:
        label_count = {}
        for item in data:
            label_count[item.label] = label_count.get(item.label,0)+1
        majority_label = max(label_count,key=label_count.get)
        return Node(label=majority_label)

    if not right_data:
        label_count = {}
        for item in data:
            label_count[item.label] = label_count.get(item.label,0)+1
        majority_label = max(label_count,key=label_count.get)
        return Node(label=majority_label)

    left_node = build_tree(left_data,feature_count-1)
    right_node = build_tree(right_data,feature_count-1)

    return Node(feature=best_feature,left=left_node,right=right_node)

def predict_single(tree,features):
    if tree.label is not None:
        return tree.label
    if features[tree.feature] == 1:
        return predict_single(tree.left,features)
    else:
        return predict_single(tree.right,features)

def predict(tree,dataset):
    return [predict_single(tree,item.features) for item in dataset]

def main():
    n,m = map(int,sys.stdin.readline().strip().split())
    train_data = []
    for _ in range(n):
        line = list(map(int,sys.stdin.readline().strip().split()))
        features = line[:-1]
        label = line[-1]
        train_data.append(Point(features,label))
    tree = build_tree(train_data,m)

    q = int(sys.stdin.readline().strip())
    query_data = []
    for _ in range(q):
        query = list(map(int,sys.stdin.readline().strip().split()))
        query_data.append(Point(query,None))

    predictions = predict(tree,query_data)

    for label in predictions:
        print(label)
if __name__ == "__main__":
    main()

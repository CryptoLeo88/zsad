import sys
from decimal import Decimal, ROUND_HALF_EVEN


def func():
    n, k = map(int, input().split())

    point = []

    for _ in range(n):
        x, y = map(int, input().split())
        point.append(Point(x, y))

    center = kmeans(point, k)

    # worse cluster
    worst_score, worse_clister = float("inf"), -1
    for i in range(k):
        score = silhouetee(point, center, k, i)
        if score < worst_score:
            worst_score, worse_clister = score, i

    # print
    bx = Decimal(str(center[worse_clister].x)).quantize(Decimal("0.00"), rounding=ROUND_HALF_EVEN)
    by = Decimal(str(center[worse_clister].y)).quantize(Decimal("0.00"), rounding=ROUND_HALF_EVEN)
    print(f"{bx},{by}")
    # please define the python3 input here. For example: a,b = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


def dist(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def kmeans(points, k, max_iter=100, tol=1e-6):
    center = [Point(p.x, p.y) for p in points[:k]]
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        # 分配到最近的簇
        for p in points:
            best, best_dist = 0, dist(p, center[0])
            for i in range(1, k):
                d = dist(p, center[i])
                if d < best_dist:
                    best, best_dist = i, d
            clusters[best].append(p)

        # 更新簇中心
        converged = True
        for i in range(k):
            if not clusters[i]:
                continue
            avgx = sum(p.x for p in clusters[i]) / len(clusters[i])
            avgy = sum(p.y for p in clusters[i])/ len(clusters[i])
            new_center = Point(avgx, avgy)

            if dist(new_center, center[i]) > tol:
                converged = False
            center[i] = new_center

        if converged:
            break
    return center


def silhouetee(point, center, k, cluster_idx):
    # 找到簇内点
    cluster = []
    assignment = []
    for p in point:
        nearest, best = 0, dist(p, center[0])

        for i in range(1, k):
            d = dist(p, center[i])
            if d < best:
                nearest, best = i, d
        assignment.append(nearest)
        if nearest == cluster_idx:
            cluster.append(p)
    if not cluster:
        return 0.0

    total_s = 0.0

    for p in cluster:
        a = 0.0  # 平均距离
        if len(cluster) > 1:
            for q in cluster:
                if q != p:
                    a += dist(p, q)
                a /= (len(cluster) - 1)

        b = float("inf")  # 到其他的最小距离
        for i in range(k):
            if i == cluster_idx:
                continue
            other = [point[j] for j in range(len(point)) if assignment[j] == i]
            if other:
                avgd = sum(dist(p, q) for q in other) / len(other)
                b = min(b, avgd)

        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        total_s += s

    return total_s / len(cluster)


if __name__ == "__main__":
    func()

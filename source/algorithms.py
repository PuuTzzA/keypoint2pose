import numpy as np
import math
import random
import itertools

def dis(a, b):
    sum = 0
    try:
        for i in range(len(a)):
            sum += (a[i] - b[i]) * (a[i] - b[i])
    except:
        print("dimension mismach ", len(a), " is not ", len(b))
    return math.sqrt(sum)

# Direct Linear Transform Estimator model
def direct_linear_transform(sample):
    A = []
    b = []

    p_matrix = sample["projection_matrix"]

    c1 = p_matrix[0][0]
    c2 = p_matrix[1][1]
    c3 = p_matrix[2][2]
    c4 = p_matrix[2][3]

    for s in sample["points"]:
        x, y, z = s["3d"]
        x_, y_ = s["2d"]

        A.append([-c1 * x, -c1 * y, -c1 * z, -c1, 0, 0, 0, 0, x * x_, y * x_, z * x_, x_])
        A.append([0, 0, 0, 0, -c2 * x, -c2 * y, -c2 * z, -c2, x * y_, y * y_, z * y_, y_])
        b.append(0)
        b.append(0)

    A = np.array(A)
    b = np.array(b)

    U, S, Vt = np.linalg.svd(A)

    model_matrix_predicted = Vt[-1].reshape(3, 4)
    model_matrix_predicted /= model_matrix_predicted[-1][-1]
    model_matrix_predicted = np.vstack([model_matrix_predicted, [0, 0, 0, 1]])

    # get the translation and rotation part out of the prediciton
    R = [[model_matrix_predicted[i][j] for j in range(0, 3)] for i in range(0, 3)]
    T = [model_matrix_predicted[i][3] for i in range(0, 3)]
    R = np.array(R)
    T = np.array(T)

    # force the scale to be 1
    U, _, Vt = np.linalg.svd(R)
    R_orthogonal = U @ Vt

    # rescale the translation by the same amount
    s = R_orthogonal[0][0] / R[0][0]
    T *= s

    model_matrix_predicted = np.eye(4)

    # put the rotation and translation back together
    for i in range(0, 4):
        for j in range(0, 4):
            if i < 3 and j < 3:
                model_matrix_predicted[i][j] = R_orthogonal[i][j]
            elif i < 3:
                model_matrix_predicted[i][j] = T[i]
    
    return model_matrix_predicted

def dlt_loss(model, point, projection_matrix):
    x, y, z = point["3d"]
    x_, y_ = point["2d"]

    projected = projection_matrix @ model @ np.array([x, y, z, 1])
    projected /= projected[3]

    return dis([x_, y_], projected[:2])

class RANSAC:
    def __init__(self, n=8, k=600, t=0.07, d=6, model=None, loss=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: function that takes a sample and predicts a model
        self.loss = loss        # `loss`: function that calculates the error of one point
        self.best_model = None
        self.best_error = np.inf

    def reset(self):
        self.best_model = None
        self.best_error = np.inf

    def fit(self, data): #algorithm like described in the slides (a bit modified to be determernistic [checks all subsets of size n])
        self.reset()

        if(len(data["points"]) < self.d):
            print("RANSAC failed, not enough datapoints")
            return self.model(data), [], np.inf
        
        best_inliers = []
        projection_matrix = data["projection_matrix"]

        for sample in itertools.combinations(data["points"], self.n): # we can use this since 14 choose 8 is only 3003
            estimation = self.model({"points" : sample, "projection_matrix" : projection_matrix})

            inliers = []
            for point in data["points"]:
                error = self.loss(estimation, point, projection_matrix)
                if error < self.t:
                    inliers.append(point)

            if len(inliers) > len(best_inliers):
                total_error = sum(self.loss(estimation, p, projection_matrix) for p in data["points"])

                if total_error < self.best_error:
                    self.best_model = estimation
                    self.best_error = total_error
                    best_inliers = inliers
                    
        if self.best_error == np.inf:
            print("RANSAC failed, no good Subset found")
            return self.model(data), [], np.inf

        return self.best_model, best_inliers, self.best_error

    def fit_wikipedia(self, data): # RANSAC as described on Wikipedia
        self.reset()
        best_inliers = []
        projection_matrix = data["projection_matrix"]

        if(len(data["points"]) < self.n):
            print("RANSAC failed, not enough datapoints")
            return self.model(data), []

        for _ in range(self.k):
            sample = random.sample(data["points"], self.n)

            estimation = self.model({"points" : sample, "projection_matrix" : projection_matrix})

            inliers = []
            for point in data["points"]:
                error = self.loss(estimation, point, projection_matrix)
                if error < self.t:
                    inliers.append(point)

            if len(inliers) >= self.d:
                better_estimation = self.model({"points" : inliers, "projection_matrix" : projection_matrix})
                better_estimation = estimation
                total_error = sum(self.loss(better_estimation, p, projection_matrix) for p in data["points"])
                
                if total_error < self.best_error:
                    self.best_model = better_estimation
                    self.best_error = total_error
                    best_inliers = inliers

        if self.best_error == np.inf:
            print("RANSAC failed, no good Subset found")
            return self.model(data), []

        return self.best_model, best_inliers
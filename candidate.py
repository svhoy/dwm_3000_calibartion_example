import itertools
import numpy as np

global perturbation_limit

devices = ["test1", "test2", "test3"]
distances = [0, 162.1613, 162.2531, 162.1720, 0, 162.2449, 162.2155, 162.2582, 0]
real_distances = [0 , 7.914, 7.914,  7.914, 0 ,7.914, 7.914, 7.914, 0]

iterations = 100
inital_delay = 513e-9

num_candidates = 1000

tof_distances = [] 
tof_real_distances = []



def convert_distance_to_tof(distances: float) -> float:
        return (1 / 299702547) * distances

def convert_list_to_matrix(list) -> np.ndarray:
    len_devices = len(devices)
    matrix = np.empty((len_devices, len_devices))
    zeile = 0
    for list_index in range(0, len(list)):
        spalte = list_index % len_devices
        if spalte == 0 and list_index != 0:
            zeile += 1

        matrix[zeile, spalte] = list[list_index]

    return matrix

def populate_cadidates(iteration, candidates: np.ndarray, perturbation_limit) -> np.ndarray:
        if iteration == 0:
            for index in range(0, num_candidates):
                arr = inital_delay + np.random.uniform(
                    -6e-9,
                    6e-9,
                    3, 
                )
                arr = np.append(arr, 0)
                #print(f"Arr: {arr}")
                candidates[index] = arr
            
        else:
            copied_elements = int(candidates.shape[0] / 4)
            best_25 = candidates[:copied_elements]
            randomized_arrays = []
            for _ in range(3): 
                for candidate in best_25:
                    randomized_shifts = np.random.uniform(
                        -1 * perturbation_limit,
                        perturbation_limit,
                        candidate.shape,
                    )
                    rnd_candidate = np.add(candidate, randomized_shifts)
                    randomized_arrays.append(rnd_candidate)

            candidates = np.concatenate([best_25, randomized_arrays])


        if iteration % 20 == 0 and iteration != 0:
            perturbation_limit = perturbation_limit / 2
        
        #print(f"Candidates: {candidates}")
        return candidates, perturbation_limit


def evaluate_candidates(candidates):
        row, column = measured_edm.shape
        edm_candidate = np.empty((row, column))
        for index, candidate in enumerate(candidates):
            for i in range(0, row):
                for j in range(0, column): 
                        if measured_edm[i, j] != 0: 
                            edm_candidate[i, j] = (((4*measured_edm[i, j]) - ((2*candidate[i]) + (2*candidate[j])))/4.0)
                        else: 
                            edm_candidate[i, j] = 0
            #print(f"Kandidaten Matrix {edm_candidate}")
            #print(real_edm-edm_candidate)
            norm_diff = np.linalg.norm(real_edm-edm_candidate)
            candidates[index, 3] = norm_diff
            #print(f"Differnz Norm: {norm_diff}")
            #print(f"Candidate: {candidate}")
        
        sorted_indices = np.argsort(candidates[:,3])
        sorted_candidates = candidates[sorted_indices]
        print(f"Sorted Candidate: {sorted_candidates}")
        return sorted_candidates

for distance in distances:
    tof_distance = convert_distance_to_tof(distance)
    tof_distances.append(tof_distance)
for distance in real_distances:
    real_distance = convert_distance_to_tof(distance)
    tof_real_distances.append(real_distance)


measured_edm = convert_list_to_matrix(tof_distances)
real_edm = convert_list_to_matrix(tof_real_distances)

print(measured_edm)
print(real_edm)

candidates = np.zeros((num_candidates, 4))
best_canidate = 0
perturbation_limit = 0.2e-9
for i in range(iterations):
    candidates, perturbation_limit = populate_cadidates(i, candidates, perturbation_limit)
    candidates = evaluate_candidates(candidates)
    print(i)

# return find_best_canidate()
print(candidates)
best_canidate = candidates[0]
print(f"Best: {best_canidate}" )

import os

import numpy as np
from scipy import spatial

def calculate_cos_score(boundrary_1, boundrary_2):
    return 1 - spatial.distance.cosine(boundrary_1, boundrary_2)

if __name__ == '__main__':

    # attr_change = "pose"
    # attr_change = "Male"
    # attr_change = "Smiling"
    attr_change = "gender"
    # attr_change = "pose"
    # attr_change = "Wavy_Hair"
    # attr_change = "Blond_Hair"
    # attr_change = "pose"
    # attr_change = "Bangs"

    space_list = ["p","pz","z"]
    method_list = ["stylegan2-pytorch","StyleMapGAN", "DiagonalGAN", "Controllable-Face-Generation",]
    test_attribute_list = ["Male", "Smiling", "pose"]
    list_1 = []
    list_2 = []

    load_dict = {}
    for method in method_list:
        eval_base_dir = os.path.join("/mnt/lustre/xuyanbo",method, "editing_evaluation")
        eval_dir = os.path.join(eval_base_dir, attr_change)
        load_dict[method]=np.load(os.path.join(eval_dir, "test_dict_softmax.npy"), allow_pickle=True)


    for attr_interest in test_attribute_list:
        for method in method_list:
            if method in ["stylegan2-pytorch", "StyleMapGAN"]:
                space_list = ["z"]
            else:
                space_list = ["p","pz","z"]

            result_list = load_dict[method]

            for space in space_list:
                attr_change_score = []
                attr_interest_score = []
                delta_sum_change_pos = 0
                delta_sum_change_neg = 0
                delta_sum_interest_pos = 0
                delta_sum_interest_neg = 0
            
                for i in range(len(result_list)):
                    delta_sum_change_pos += np.sum(np.array(result_list[i][attr_change][space][4:7])-np.array(result_list[i][attr_change][space][3:6]))
                    delta_sum_interest_pos += np.sum(np.array(result_list[i][attr_interest][space][4:7])-np.array(result_list[i][attr_interest][space][3:6]))
                    delta_sum_change_neg += np.sum(np.array(result_list[i][attr_change][space][0:3])-np.array(result_list[i][attr_change][space][1:4]))
                    delta_sum_interest_neg += np.sum(np.array(result_list[i][attr_interest][space][0:3])-np.array(result_list[i][attr_interest][space][1:4]))
                    attr_change_score.append(result_list[i][attr_change][space])
                    attr_interest_score.append(result_list[i][attr_interest][space])
                    
                delta_sum_change_pos /= len(result_list)
                delta_sum_interest_pos /= len(result_list)

                delta_sum_change_neg /= len(result_list)
                delta_sum_interest_neg /= len(result_list)

                attr_change_score = np.concatenate(attr_change_score)
                attr_interest_score =  np.concatenate(attr_interest_score)
                corralation = np.corrcoef(attr_change_score, attr_interest_score)

        
                result = (abs(delta_sum_interest_pos/delta_sum_change_pos) + abs(delta_sum_interest_neg/delta_sum_change_neg))/2
                print(method, attr_change, attr_interest, space, result)
            